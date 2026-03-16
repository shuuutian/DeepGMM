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
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

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
    """

    def __init__(
        self,
        mode: str = "oracle",
        max_num_epochs: int = 1500,
        batch_size: int = 256,
        h_lr: float = 2e-4,
        f_lr: float = 8e-4,
        eval_freq: int = 20,
        burn_in: int = 200,
        enable_cuda: bool = False,
    ):
        if mode not in {"oracle", "naive"}:
            raise ValueError(f"OriginalDeepGMMBaseline only supports 'oracle' and 'naive', got '{mode}'")
        self.mode = mode
        self.max_num_epochs = max_num_epochs
        self.batch_size = batch_size
        self.h_lr = h_lr
        self.f_lr = f_lr
        self.eval_freq = eval_freq
        self.burn_in = burn_in
        self.enable_cuda = enable_cuda

        self.h: Optional[nn.Module] = None
        self.f: Optional[nn.Module] = None
        self._train_w: Optional[torch.Tensor] = None   # raw W (outcome_proxy) for beta_hat
        self._h_pred_history: list = []                # Polyak snapshots
        self._best_h_state: Optional[dict] = None
        self._best_dev_loss: float = float("inf")
        self.loss_history: list = []
        # diagnostics
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

    def _dev_moment(self, x_dev: torch.Tensor, z_dev: torch.Tensor, y_dev: torch.Tensor) -> float:
        """MSE on dev set — same criterion as SGDLearningPCI for comparability."""
        self.h.eval()
        with torch.no_grad():
            residual = torch.squeeze(y_dev) - torch.squeeze(self.h(x_dev))
            mse = float((residual ** 2).mean())
        self.h.train()
        return mse

    def _snapshot(self, x_train: torch.Tensor) -> None:
        """Record h(x) predictions on training set for Polyak averaging."""
        self.h.eval()
        with torch.no_grad():
            pred = torch.squeeze(self.h(x_train)).detach().cpu()
        self.h.train()
        self._h_pred_history.append(pred)

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

        x_dev_t = z_dev_t = y_dev_t = None
        if dev_data is not None:
            dev_filtered = self._filter_data(dev_data)
            x_dev_t, z_dev_t = self._to_xz(dev_filtered)
            y_dev_t = dev_filtered.outcome
        # store raw W for beta_hat
        self._train_w = train_data.outcome_proxy.detach().cpu()

        device = y_train.device
        if self.enable_cuda and torch.cuda.is_available():
            x_train = x_train.cuda(); z_train = z_train.cuda(); y_train = y_train.cuda()
            if x_dev_t is not None:
                x_dev_t = x_dev_t.cuda(); z_dev_t = z_dev_t.cuda(); y_dev_t = y_dev_t.cuda()

        h = MLPModel(input_dim=2, layer_widths=[64, 64], activation=nn.LeakyReLU).double()
        f = MLPModel(input_dim=3, layer_widths=[64, 64], activation=nn.LeakyReLU).double()
        if self.enable_cuda and torch.cuda.is_available():
            h = h.cuda(); f = f.cuda()
        h.initialize(); f.initialize()

        h_opt = OptimizerFactory(OAdam, lr=self.h_lr, betas=(0.5, 0.9))(h)
        f_opt = OptimizerFactory(OAdam, lr=self.f_lr, betas=(0.5, 0.9))(f)
        objective = OptimalMomentObjective()

        self.h = h; self.f = f
        n = x_train.shape[0]

        for epoch in range(self.max_num_epochs):
            h.train(); f.train()
            perm = torch.randperm(n, device=x_train.device)
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

            self.loss_history.append((epoch_h / num_batches, epoch_f / num_batches))

            # ── model selection + Polyak snapshot ───────────────────────────
            if epoch >= self.burn_in and epoch % self.eval_freq == 0:
                self._snapshot(x_train)
                if x_dev_t is not None:
                    dev_loss = self._dev_moment(x_dev_t, z_dev_t, y_dev_t)
                    self.dev_moment_history.append((epoch, dev_loss))
                    if dev_loss < self._best_dev_loss:
                        self._best_dev_loss = dev_loss
                        self._best_h_state = copy.deepcopy(h.state_dict())
                        self.best_checkpoint_epoch = epoch

            if verbose and epoch % 50 == 0:
                h_l, f_l = self.loss_history[-1]
                print(f"  epoch={epoch:04d}  h_loss={h_l:.5f}  f_loss={f_l:.5f}")

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
