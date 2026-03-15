from __future__ import annotations

from typing import Optional, Tuple

import torch

from game_objectives.abstract_objective import AbstractObjective


class PCIOptimalMomentObjective(AbstractObjective):
    """
    MAR-adapted DeepGMM game objective with imputed residuals.
    """

    def __init__(self, lambda_1: float = 0.25):
        super().__init__()
        self.lambda_1 = float(lambda_1)

    @staticmethod
    def _as_1d(tensor: torch.Tensor) -> torch.Tensor:
        return torch.squeeze(tensor)

    def calc_objective(
        self,
        h,
        f,
        treatment: torch.Tensor,
        treatment_proxy: torch.Tensor,
        outcome_proxy: torch.Tensor,
        outcome: torch.Tensor,
        delta_w: torch.Tensor,
        m_theta: Optional[torch.Tensor] = None,
        v_theta_bar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        l = torch.cat([treatment, treatment_proxy], dim=1)
        f_l = self._as_1d(f(l))

        y = self._as_1d(outcome)
        delta = self._as_1d(delta_w).to(dtype=y.dtype)
        obs_mask = delta > 0.5

        residual = torch.zeros_like(y)
        if obs_mask.any():
            h_input_obs = torch.cat([outcome_proxy[obs_mask], treatment[obs_mask]], dim=1)
            h_obs = self._as_1d(h(h_input_obs))
            residual_obs = y[obs_mask] - h_obs
            residual[obs_mask] = residual_obs
        else:
            residual_obs = torch.zeros(0, dtype=y.dtype, device=y.device)

        if (~obs_mask).any():
            if m_theta is None:
                raise ValueError("m_theta must be provided when there are missing outcome proxies")
            residual[~obs_mask] = self._as_1d(m_theta)[~obs_mask]

        s_tilde = torch.zeros_like(y)
        if obs_mask.any():
            s_tilde[obs_mask] = residual_obs.pow(2)
        if (~obs_mask).any():
            if v_theta_bar is None:
                raise ValueError("v_theta_bar must be provided when there are missing outcome proxies")
            s_tilde[~obs_mask] = self._as_1d(v_theta_bar)[~obs_mask]

        moment = (f_l * residual).mean()
        f_reg = self.lambda_1 * (f_l.pow(2) * s_tilde).mean()
        return moment, -moment + f_reg

