"""
Oracle and naive baselines using the ORIGINAL DeepGMM objective
(OptimalMomentObjective from simple_moment_objective.py) with proper
model selection (checkpoint + dev-set eval) and Polyak averaging.

Purpose: isolate whether bad results come from the DGP/data vs our PCI
learning code. These baselines are deliberately close to the original
published DeepGMM machinery.

Data format adapter
-------------------
Original DeepGMM:  g(x), f(z)
PCI demand DGP:    h(W, A),   f(A, Z)

We set:
    x  ←  concat(W, A)   input to h  [dim 2]  (raw, unnormalised)
    z  ←  concat(A, Z)   input to f  [dim 3]  (raw, unnormalised)
    y  ←  Y  (outcome)

LR grid search
--------------
If lr_grid is provided (list of h_lr values), fit() runs a short
selection phase for each entry (f_lr = 5 * h_lr, matching the toy
baseline's ratio), picks the config with lowest dev MSE, then does
a full training run with the winning config.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from data.data_class_mar import PVTrainDataSetMARTorch
from game_objectives.simple_moment_objective import OptimalMomentObjective
from models.mlp_model import MLPModel
from optimizers.oadam import OAdam
from optimizers.optimizer_factory import OptimizerFactory


class OriginalDeepGMMBaseline:
    """
    Oracle / naive baseline using the original OptimalMomentObjective.

    mode='oracle' : use all W (delta_w forced to 1), run standard DeepGMM.
    mode='naive'  : keep only complete-case rows (delta_w == 1), run standard
                    DeepGMM on the filtered subset — reproduces the biased
                    complete-case estimator.

    Key differences from our PCIDeepGMMMethod:
    - Uses OptimalMomentObjective (epsilon = h - Y, same sign as original paper)
    - No cross-fitting imputation (no MAR correction)
    - Uses raw (unnormalised) inputs — same as PCIDeepGMMMethod
    - Saves h checkpoint at each eval and restores the best dev-set one

    LR grid search (lr_grid parameter):
    - If lr_grid=[lr1, lr2, ...], a selection phase of selection_epochs is run
      for each h_lr (with f_lr = 5 * h_lr).  The config with lowest dev MSE
      wins and is used for the full max_num_epochs training run.
    - If lr_grid=None, a single run with h_lr / f_lr is used.
    """

    def __init__(
        self,
        mode: str = "oracle",
        max_num_epochs: int = 6000,
        batch_size: int = 1024,
        h_lr: float = 5e-4,
        f_lr: float = 2.5e-3,
        lr_grid: Optional[List[float]] = None,
        selection_epochs: int = 2000,
        eval_freq: int = 20,
        burn_in: int = 1000,
        enable_cuda: bool = False,
    ):
        if mode not in {"oracle", "naive"}:
            raise ValueError(f"OriginalDeepGMMBaseline only supports 'oracle' and 'naive', got '{mode}'")
        self.mode = mode
        self.max_num_epochs = max_num_epochs
        self.batch_size = batch_size
        self.h_lr = h_lr
        self.f_lr = f_lr
        self.lr_grid = lr_grid          # list of h_lr values; f_lr = 5 * h_lr each
        self.selection_epochs = selection_epochs
        self.eval_freq = eval_freq
        self.burn_in = burn_in
        self.enable_cuda = enable_cuda

        self.h: Optional[nn.Module] = None
        self.f: Optional[nn.Module] = None
        self._train_w: Optional[torch.Tensor] = None   # raw W (outcome_proxy) for beta_hat
        self._best_h_state: Optional[dict] = None
        self._best_dev_loss: float = float("inf")
        self.loss_history: list = []
        self.dev_moment_history: list = []   # list of (epoch, dev_moment)
        self.best_checkpoint_epoch: int = -1

    # ── data preparation ─────────────────────────────────────────────────────

    def _filter_data(self, data: PVTrainDataSetMARTorch) -> PVTrainDataSetMARTorch:
        """For naive: keep only complete-case rows."""
        if self.mode == "naive":
            keep = torch.squeeze(data.delta_w) > 0.5
            return data.subset(torch.where(keep)[0].tolist())
        return data   # oracle: use all rows as-is

    def _to_xz(self, data: PVTrainDataSetMARTorch) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([data.outcome_proxy, data.treatment], dim=1)
        z = torch.cat([data.treatment, data.treatment_proxy], dim=1)
        return x, z

    # ── training utilities ───────────────────────────────────────────────────

    def _make_models(self, device: torch.device) -> Tuple[nn.Module, nn.Module]:
        h = MLPModel(input_dim=2, layer_widths=[64, 64], activation=nn.LeakyReLU).double()
        f = MLPModel(input_dim=3, layer_widths=[64, 64], activation=nn.LeakyReLU).double()
        h.initialize(); f.initialize()
        return h.to(device), f.to(device)

    def _dev_mse(self, h: nn.Module, x_dev: torch.Tensor, y_dev: torch.Tensor) -> float:
        """MSE on dev set — same criterion as SGDLearningPCI for comparability."""
        h.eval()
        with torch.no_grad():
            residual = torch.squeeze(y_dev) - torch.squeeze(h(x_dev))
            mse = float((residual ** 2).mean())
        h.train()
        return mse

    # ── single training run ──────────────────────────────────────────────────

    def _train_one_config(
        self,
        x_train: torch.Tensor,
        z_train: torch.Tensor,
        y_train: torch.Tensor,
        x_dev: Optional[torch.Tensor],
        y_dev: Optional[torch.Tensor],
        h_lr: float,
        f_lr: float,
        epochs: int,
        burn_in: int,
    ) -> Tuple[float, Optional[dict], int, list, list, nn.Module, nn.Module]:
        """
        Train fresh h and f for `epochs` epochs with given LRs.

        Returns:
            best_dev_loss, best_h_state, best_epoch,
            loss_history, dev_moment_history,
            h_model, f_model
        """
        device = x_train.device
        h, f = self._make_models(device)
        objective = OptimalMomentObjective()
        h_opt = OptimizerFactory(OAdam, lr=h_lr, betas=(0.5, 0.9))(h)
        f_opt = OptimizerFactory(OAdam, lr=f_lr, betas=(0.5, 0.9))(f)

        n = x_train.shape[0]
        best_dev_loss = float("inf")
        best_h_state: Optional[dict] = None
        best_epoch = -1
        loss_history: list = []
        dev_moment_history: list = []

        for epoch in range(epochs):
            h.train(); f.train()
            perm = torch.randperm(n, device=device)
            num_batches = int(np.ceil(n / self.batch_size))
            epoch_h = 0.0; epoch_f = 0.0

            for b in range(num_batches):
                idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                xb = x_train[idx]; zb = z_train[idx]; yb = y_train[idx]
                h_obj, f_obj = objective.calc_objective(h, f, xb, zb, yb)

                h_opt.zero_grad(); f_opt.zero_grad()
                h_obj.backward(retain_graph=True)
                f_obj.backward()
                h_opt.step(); f_opt.step()

                epoch_h += float(h_obj.detach().cpu())
                epoch_f += float(f_obj.detach().cpu())

            loss_history.append((epoch_h / num_batches, epoch_f / num_batches))

            if epoch >= burn_in and epoch % self.eval_freq == 0 and x_dev is not None:
                dev_loss = self._dev_mse(h, x_dev, y_dev)
                dev_moment_history.append((epoch, dev_loss))
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_h_state = copy.deepcopy(h.state_dict())
                    best_epoch = epoch

        return best_dev_loss, best_h_state, best_epoch, loss_history, dev_moment_history, h, f

    # ── main fit ─────────────────────────────────────────────────────────────

    def fit(
        self,
        train_data: PVTrainDataSetMARTorch,
        dev_data: Optional[PVTrainDataSetMARTorch] = None,
        verbose: bool = False,
    ) -> None:
        train_data = self._filter_data(train_data)
        x_train, z_train = self._to_xz(train_data)
        y_train = train_data.outcome

        x_dev_t = y_dev_t = None
        if dev_data is not None:
            dev_filtered = self._filter_data(dev_data)
            x_dev_t, _ = self._to_xz(dev_filtered)
            y_dev_t = dev_filtered.outcome

        self._train_w = train_data.outcome_proxy.detach().cpu()

        if self.enable_cuda and torch.cuda.is_available():
            x_train = x_train.cuda(); z_train = z_train.cuda(); y_train = y_train.cuda()
            if x_dev_t is not None:
                x_dev_t = x_dev_t.cuda(); y_dev_t = y_dev_t.cuda()

        # ── LR selection phase ───────────────────────────────────────────────
        grid = self.lr_grid if self.lr_grid and len(self.lr_grid) > 1 else None
        if grid is not None:
            best_sel_loss = float("inf")
            best_h_lr = grid[0]
            if verbose:
                print(f"  [LR selection] trying {len(grid)} configs for {self.selection_epochs} epochs each")
            for h_lr_cand in grid:
                f_lr_cand = 5.0 * h_lr_cand
                sel_loss, _, sel_epoch, _, _, _, _ = self._train_one_config(
                    x_train, z_train, y_train, x_dev_t, y_dev_t,
                    h_lr_cand, f_lr_cand,
                    self.selection_epochs, burn_in=200,
                )
                if verbose:
                    print(f"    h_lr={h_lr_cand:.1e}  f_lr={f_lr_cand:.1e}  "
                          f"best_dev_mse={sel_loss:.5f}  (epoch {sel_epoch})")
                if sel_loss < best_sel_loss:
                    best_sel_loss = sel_loss
                    best_h_lr = h_lr_cand
            winning_h_lr = best_h_lr
            winning_f_lr = 5.0 * best_h_lr
            if verbose:
                print(f"  [LR selection] winner: h_lr={winning_h_lr:.1e}  f_lr={winning_f_lr:.1e}")
        else:
            # Single config — use constructor h_lr / f_lr (or the only grid entry)
            winning_h_lr = self.lr_grid[0] if self.lr_grid else self.h_lr
            winning_f_lr = 5.0 * winning_h_lr if self.lr_grid else self.f_lr

        # ── Full training run with winning LR ────────────────────────────────
        if verbose:
            print(f"  [Full training] h_lr={winning_h_lr:.1e}  f_lr={winning_f_lr:.1e}  "
                  f"epochs={self.max_num_epochs}  burn_in={self.burn_in}")

        (self._best_dev_loss, self._best_h_state, self.best_checkpoint_epoch,
         self.loss_history, self.dev_moment_history,
         self.h, self.f) = self._train_one_config(
            x_train, z_train, y_train, x_dev_t, y_dev_t,
            winning_h_lr, winning_f_lr,
            self.max_num_epochs, self.burn_in,
        )

        if self._best_h_state is not None:
            self.h.load_state_dict(self._best_h_state)

    # ── ATE estimation ───────────────────────────────────────────────────────

    def beta_hat(self, a_value: float, data: Optional[PVTrainDataSetMARTorch] = None) -> float:
        """β̂(a) = (1/n) Σ_i h(W_i, a) using raw (unnormalised) inputs."""
        if data is None:
            w_col = self._train_w
        else:
            w_col = data.outcome_proxy.cpu()
        a_col = torch.full((w_col.shape[0], 1), float(a_value), dtype=w_col.dtype)
        x = torch.cat([w_col, a_col], dim=1)

        self.h.eval()
        with torch.no_grad():
            dev = next(self.h.parameters()).device
            pred = torch.squeeze(self.h(x.to(dev))).detach().cpu()
        return float(pred.mean())

    def predict_ate(self, a1: float, a0: float) -> float:
        return float(self.beta_hat(a1) - self.beta_hat(a0))
