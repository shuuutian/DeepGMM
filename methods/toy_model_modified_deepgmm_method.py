from __future__ import annotations

from typing import Optional

import torch

from data.data_class_mar import PVTrainDataSetMARTorch
from methods.pci_deepgmm_method import PCIDeepGMMMethod
from methods.toy_model_selection_method import ToyModelSelectionMethod


class ToyModelModifiedDeepGMMMethod:
    """
    Unified oracle / naive / modified comparison on toy DGP data.

    Oracle / naive  → ToyModelSelectionMethod (original DeepGMM pipeline).
                      ToyModelSelectionMethod expects g: input_dim=1, f: input_dim=2,
                      which matches the toy DGP directly (x scalar, z 2-D).
    Modified        → PCIDeepGMMMethod with W = x as the outcome proxy (MAR mask
                      applied).  h: input_dim=2 ([W, A]), f: input_dim=3 ([A, Z]).

    All fit() inputs are torch double tensors:
        x_train       (n, 1)  treatment
        z_train       (n, 2)  instruments / treatment proxy
        y_train       (n, 1)  outcome
        delta_w_train (n, 1)  MAR indicator  (1 = W observed, 0 = missing)

    ATE estimation
    --------------
    Oracle / naive  : point evaluation  β̂(a) = g(a)          (single forward pass)
    Modified        : cross-fit average β̂(a) = (1/n) Σ h(Wᵢ, a)
    Both converge to g_true(a) if the estimator is consistent.
    """

    def __init__(
        self,
        mode: str = "modified",
        n_folds: int = 5,
        missing_rate: float = 0.3,
        enable_cuda: bool = False,
        max_num_epochs: int = 300,
        batch_size: int = 256,
    ):
        if mode not in {"oracle", "naive", "modified"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.n_folds = n_folds
        self.missing_rate = missing_rate
        self.enable_cuda = enable_cuda
        self.max_num_epochs = max_num_epochs
        self.batch_size = batch_size

        self._toy_method: Optional[ToyModelSelectionMethod] = None
        self._pci_method: Optional[PCIDeepGMMMethod] = None

        # diagnostics — empty for oracle/naive (ToyModelSelectionMethod does not
        # expose per-epoch losses); populated for modified via PCIDeepGMMMethod
        self.loss_history: list = []
        self.dev_moment_history: list = []
        self.best_checkpoint_epoch: int = -1

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _obs_mask(delta_w: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(delta_w) > 0.5

    @staticmethod
    def _filter(mask: torch.Tensor, *tensors):
        return tuple(t[mask] for t in tensors)

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        x_train: torch.Tensor,
        z_train: torch.Tensor,
        y_train: torch.Tensor,
        delta_w_train: torch.Tensor,
        x_dev: Optional[torch.Tensor] = None,
        z_dev: Optional[torch.Tensor] = None,
        y_dev: Optional[torch.Tensor] = None,
        delta_w_dev: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> None:
        if self.mode in {"oracle", "naive"}:
            self._fit_toy(
                x_train, z_train, y_train, delta_w_train,
                x_dev, z_dev, y_dev, delta_w_dev, verbose,
            )
        else:
            self._fit_pci(
                x_train, z_train, y_train, delta_w_train,
                x_dev, z_dev, y_dev, delta_w_dev, verbose,
            )

    def _fit_toy(
        self,
        x_train, z_train, y_train, delta_w_train,
        x_dev, z_dev, y_dev, delta_w_dev, verbose,
    ) -> None:
        if self.mode == "naive":
            keep = self._obs_mask(delta_w_train)
            x_tr, z_tr, y_tr = self._filter(keep, x_train, z_train, y_train)
        else:  # oracle: use all rows
            x_tr, z_tr, y_tr = x_train, z_train, y_train

        x_dv = z_dv = y_dv = None
        if x_dev is not None:
            if self.mode == "naive" and delta_w_dev is not None:
                keep_d = self._obs_mask(delta_w_dev)
                x_dv, z_dv, y_dv = self._filter(keep_d, x_dev, z_dev, y_dev)
            else:
                x_dv, z_dv, y_dv = x_dev, z_dev, y_dev

        method = ToyModelSelectionMethod(enable_cuda=self.enable_cuda)
        method.fit(x_tr, z_tr, y_tr, x_dv, z_dv, y_dv, verbose=verbose)
        self._toy_method = method

    def _fit_pci(
        self,
        x_train, z_train, y_train, delta_w_train,
        x_dev, z_dev, y_dev, delta_w_dev, verbose,
    ) -> None:
        # Outcome proxy W = x (treatment); zero-fill for missing units
        obs_mask = self._obs_mask(delta_w_train)
        w_train = x_train.clone()
        w_train[~obs_mask] = 0.0

        train_data = PVTrainDataSetMARTorch(
            treatment=x_train,
            treatment_proxy=z_train,
            outcome_proxy=w_train,
            outcome=y_train,
            backdoor=None,
            delta_w=delta_w_train,
        )

        dev_data = None
        if x_dev is not None:
            w_dev = x_dev.clone()
            dw_dev = delta_w_dev if delta_w_dev is not None else torch.ones_like(x_dev)
            w_dev[~self._obs_mask(dw_dev)] = 0.0
            dev_data = PVTrainDataSetMARTorch(
                treatment=x_dev,
                treatment_proxy=z_dev,
                outcome_proxy=w_dev,
                outcome=y_dev,
                backdoor=None,
                delta_w=dw_dev,
            )

        pci = PCIDeepGMMMethod(
            mode="modified",
            n_folds=self.n_folds,
            missing_rate=self.missing_rate,
            enable_cuda=self.enable_cuda,
            max_num_epochs=self.max_num_epochs,
            batch_size=self.batch_size,
        )
        pci.fit(train_data, dev_data, verbose=verbose)
        self._pci_method = pci

        # expose diagnostics from learner
        self.loss_history = pci.learner.loss_history
        self.dev_moment_history = pci.learner.dev_moment_history
        self.best_checkpoint_epoch = pci.learner.best_checkpoint_epoch

    # ── ATE estimation ────────────────────────────────────────────────────────

    def beta_hat(self, a_value: float) -> float:
        if self.mode in {"oracle", "naive"}:
            if self._toy_method is None:
                raise AttributeError("fit must be called before beta_hat")
            x_test = torch.tensor([[a_value]], dtype=torch.float64)
            if self.enable_cuda and torch.cuda.is_available():
                x_test = x_test.cuda()
            with torch.no_grad():
                pred = self._toy_method.predict(x_test)
            return float(pred.mean())
        else:
            if self._pci_method is None:
                raise AttributeError("fit must be called before beta_hat")
            return self._pci_method.beta_hat(a_value)

    def predict_ate(self, a1: float, a0: float) -> float:
        return float(self.beta_hat(a1) - self.beta_hat(a0))
