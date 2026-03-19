from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data.data_class_mar import PVTrainDataSetMARTorch, create_k_folds


@dataclass
class FoldLinearModel:
    weight: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]
    fallback: torch.Tensor

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None or self.bias is None:
            return self.fallback.expand(x.shape[0], 1)
        return x @ self.weight + self.bias


class SGDLearningPCI:
    def __init__(
        self,
        game_objective,
        h,
        f,
        h_optimizer,
        f_optimizer,
        n_folds: int = 5,
        max_num_epochs: int = 500,
        batch_size: int = 256,
        ema_alpha: float = 0.05,
        ridge: float = 1e-3,
        eval_freq: int = 20,        # evaluate dev set every N epochs
        burn_in: int = 200,         # epochs before checkpoint selection starts
    ):
        self.game_objective = game_objective
        self.h = h
        self.f = f
        self.h_optimizer = h_optimizer
        self.f_optimizer = f_optimizer
        self.n_folds = n_folds
        self.max_num_epochs = max_num_epochs
        self.batch_size = batch_size
        self.ema_alpha = ema_alpha
        self.ridge = ridge
        self.eval_freq = eval_freq
        self.burn_in = burn_in
        self.loss_history: List[Tuple[float, float]] = []
        self.fold_indices: Optional[List[torch.Tensor]] = None
        self._last_fold_train_indices: Dict[int, torch.Tensor] = {}
        self.theta_bar = None
        # Polyak averaging: accumulate h(W,A) predictions on training set
        self._h_pred_history: List[torch.Tensor] = []  # each entry shape (n,)
        self._best_h_state: Optional[dict] = None
        self._best_dev_loss: float = float("inf")
        # diagnostics
        self.dev_moment_history: List[Tuple[int, float]] = []  # (epoch, dev_moment)
        self.best_checkpoint_epoch: int = -1

    def _lplus(self, data: PVTrainDataSetMARTorch) -> torch.Tensor:
        pieces = [data.treatment, data.treatment_proxy, data.outcome]
        if data.backdoor is not None:
            pieces.append(data.backdoor)
        return torch.cat(pieces, dim=1)

    def _fit_linear_model(self, x: torch.Tensor, y: torch.Tensor) -> FoldLinearModel:
        if x.shape[0] == 0:
            return FoldLinearModel(weight=None, bias=None, fallback=torch.zeros(1, 1, device=y.device, dtype=y.dtype))
        x_aug = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)], dim=1)
        eye = torch.eye(x_aug.shape[1], device=x.device, dtype=x.dtype)
        beta = torch.linalg.solve(x_aug.T @ x_aug + self.ridge * eye, x_aug.T @ y)
        weight = beta[:-1]
        bias = beta[-1:].reshape(1, 1)
        return FoldLinearModel(weight=weight, bias=bias, fallback=y.mean().reshape(1, 1))

    def _cross_fit_models(
        self,
        data: PVTrainDataSetMARTorch,
        residual: torch.Tensor,
        residual_sq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fold_indices is None:
            self.fold_indices = create_k_folds(data, self.n_folds, seed=123)
        features = self._lplus(data)
        delta = torch.squeeze(data.delta_w) > 0.5
        m_all = torch.zeros_like(residual)
        v_all = torch.zeros_like(residual_sq)
        for k, val_idx in enumerate(self.fold_indices):
            train_idx = torch.cat(
                [self.fold_indices[j] for j in range(len(self.fold_indices)) if j != k], dim=0
            )
            self._last_fold_train_indices[k] = train_idx.detach().cpu()
            obs_train = train_idx[delta[train_idx]]
            x_train = features[obs_train]
            m_model = self._fit_linear_model(x_train, residual[obs_train].reshape(-1, 1))
            v_model = self._fit_linear_model(x_train, residual_sq[obs_train].reshape(-1, 1))
            x_val = features[val_idx]
            m_all[val_idx] = torch.squeeze(m_model.predict(x_val))
            v_all[val_idx] = torch.squeeze(torch.clamp(v_model.predict(x_val), min=1e-8))
        return m_all, v_all

    def _compute_observed_residual(self, data: PVTrainDataSetMARTorch, model) -> torch.Tensor:
        y = torch.squeeze(data.outcome)
        delta = torch.squeeze(data.delta_w) > 0.5
        residual = torch.zeros_like(y)
        if delta.any():
            h_in = torch.cat([data.outcome_proxy[delta], data.treatment[delta]], dim=1)
            pred = torch.squeeze(model(h_in))
            residual[delta] = y[delta] - pred
        return residual

    def _update_theta_bar(self) -> None:
        with torch.no_grad():
            if self.theta_bar is None:
                self.theta_bar = {k: v.detach().clone() for k, v in self.h.state_dict().items()}
                return
            current = self.h.state_dict()
            for key in self.theta_bar.keys():
                self.theta_bar[key] = (1.0 - self.ema_alpha) * self.theta_bar[key] + self.ema_alpha * current[key].detach()

    def _load_theta_bar_model(self):
        h_bar = copy.deepcopy(self.h)
        if self.theta_bar is not None:
            h_bar.load_state_dict(self.theta_bar)
        h_bar = h_bar.to(next(self.h.parameters()).device).double()
        h_bar.eval()
        return h_bar

    def _dev_moment(self, dev_data: PVTrainDataSetMARTorch) -> float:
        """
        Observed-unit MSE on dev set — smaller is better, f-independent.

        We use MSE instead of the f-weighted moment for two reasons:
        1. The old criterion evaluated E[f·r·δ_W] (complete-case), which is a
           different identification equation from the PCI target E[f·r̃] and can
           be spuriously small for checkpoints that satisfy the wrong (biased)
           CMR.
        2. When s̃ is large early in training the f_reg term pushes f→0
           (f-collapse), making the f-weighted moment ≈ 0 for all h, selecting
           the earliest checkpoint regardless of h quality.
        MSE is f-independent, stable from epoch 1, and monotonically decreases
        as h converges toward the structural bridge function.
        """
        self.h.eval()
        with torch.no_grad():
            delta = torch.squeeze(dev_data.delta_w) > 0.5
            if not delta.any():
                return float("inf")
            h_in = torch.cat([dev_data.outcome_proxy[delta], dev_data.treatment[delta]], dim=1)
            residual = torch.squeeze(dev_data.outcome)[delta] - torch.squeeze(self.h(h_in))
            mse = float((residual ** 2).mean())
        self.h.train()
        return mse

    def _snapshot_h_pred(self, train_data: PVTrainDataSetMARTorch) -> None:
        """Record h(W_i, A_i) on training observed units for Polyak averaging."""
        self.h.eval()
        with torch.no_grad():
            delta = torch.squeeze(train_data.delta_w) > 0.5
            pred = torch.zeros(train_data.num_data(), dtype=train_data.outcome.dtype,
                               device=train_data.outcome.device)
            if delta.any():
                h_in = torch.cat([train_data.outcome_proxy[delta], train_data.treatment[delta]], dim=1)
                pred[delta] = torch.squeeze(self.h(h_in))
        self.h.train()
        self._h_pred_history.append(pred.detach().cpu())

    def fit(
        self,
        train_data: PVTrainDataSetMARTorch,
        dev_data: Optional[PVTrainDataSetMARTorch] = None,
        verbose: bool = False,
    ) -> None:
        n = train_data.num_data()
        self.fold_indices = create_k_folds(train_data, self.n_folds, seed=123)
        self._update_theta_bar()

        for epoch in range(self.max_num_epochs):
            h_bar = self._load_theta_bar_model()
            residual = self._compute_observed_residual(train_data, self.h).detach()
            residual_bar = self._compute_observed_residual(train_data, h_bar).detach()
            m_all, v_all = self._cross_fit_models(train_data, residual, residual_bar.pow(2))
            m_all = m_all.detach()
            v_all = v_all.detach()

            perm = torch.randperm(n, device=train_data.outcome.device)
            num_batches = int(np.ceil(n / self.batch_size))
            epoch_h_loss = 0.0
            epoch_f_loss = 0.0
            for b in range(num_batches):
                idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                batch = train_data.subset(idx.tolist())
                h_obj, f_obj = self.game_objective.calc_objective(
                    self.h,
                    self.f,
                    batch.treatment,
                    batch.treatment_proxy,
                    batch.outcome_proxy,
                    batch.outcome,
                    batch.delta_w,
                    m_theta=m_all[idx],
                    v_theta_bar=v_all[idx],
                )
                self.h_optimizer.zero_grad()
                self.f_optimizer.zero_grad()
                h_obj.backward(retain_graph=True)
                f_obj.backward()
                self.h_optimizer.step()
                self.f_optimizer.step()

                epoch_h_loss += float(h_obj.detach().cpu())
                epoch_f_loss += float(f_obj.detach().cpu())

            self._update_theta_bar()
            self.loss_history.append((epoch_h_loss / num_batches, epoch_f_loss / num_batches))

            # ── model selection + Polyak averaging ──────────────────────────
            if epoch >= self.burn_in and epoch % self.eval_freq == 0:
                self._snapshot_h_pred(train_data)
                if dev_data is not None:
                    dev_loss = self._dev_moment(dev_data)
                    self.dev_moment_history.append((epoch, dev_loss))
                    if dev_loss < self._best_dev_loss:
                        self._best_dev_loss = dev_loss
                        self._best_h_state = copy.deepcopy(self.h.state_dict())
                        self.best_checkpoint_epoch = epoch

            if verbose and epoch % 20 == 0:
                h_l, f_l = self.loss_history[-1]
                print(f"epoch={epoch:04d} h_loss={h_l:.6f} f_loss={f_l:.6f}")

        # ── restore best checkpoint ──────────────────────────────────────────
        if self._best_h_state is not None:
            self.h.load_state_dict(self._best_h_state)
        if verbose:
            print(f"Best dev moment: {self._best_dev_loss:.6f}")

    def _polyak_h_at(self, a_value: float, data: PVTrainDataSetMARTorch) -> Optional[torch.Tensor]:
        """
        Return Polyak-averaged h predictions (h averaged over snapshot history).
        Returns None if no snapshots recorded.
        Returns a 1-D tensor of shape (n,) on CPU.
        """
        if not self._h_pred_history:
            return None
        return torch.stack(self._h_pred_history, dim=0).mean(dim=0)

    def estimate_beta(self, data: PVTrainDataSetMARTorch, a_value: float) -> float:
        a_col = torch.full_like(data.treatment, float(a_value))
        delta = torch.squeeze(data.delta_w) > 0.5
        y_hat = torch.zeros(data.num_data(), device=data.treatment.device, dtype=data.treatment.dtype)
        if delta.any():
            obs_input = torch.cat([data.outcome_proxy[delta], a_col[delta]], dim=1)
            y_hat[delta] = torch.squeeze(self.h(obs_input))

        if (~delta).any():
            if self.fold_indices is None:
                self.fold_indices = create_k_folds(data, self.n_folds, seed=123)
            lplus = self._lplus(data)
            target_obs = torch.zeros_like(y_hat)
            target_obs[delta] = y_hat[delta]
            for k, val_idx in enumerate(self.fold_indices):
                train_idx = torch.cat(
                    [self.fold_indices[j] for j in range(len(self.fold_indices)) if j != k], dim=0
                )
                obs_train = train_idx[delta[train_idx]]
                q_model = self._fit_linear_model(lplus[obs_train], target_obs[obs_train].reshape(-1, 1))
                val_missing = val_idx[~delta[val_idx]]
                if len(val_missing) > 0:
                    y_hat[val_missing] = torch.squeeze(q_model.predict(lplus[val_missing]))
        return float(y_hat.mean().detach().cpu())

    def get_fold_training_indices(self) -> Dict[int, torch.Tensor]:
        return self._last_fold_train_indices
