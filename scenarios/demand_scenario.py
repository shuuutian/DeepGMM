from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from data.data_class_mar import PVTrainDataSetMAR
from scenarios.abstract_scenario import AbstractScenario


def _psi(t: np.ndarray) -> np.ndarray:
    return 2.0 * (((t - 5.0) ** 4) / 600.0 + np.exp(-4.0 * (t - 5.0) ** 2) + t / 10.0 - 2.0)


@dataclass
class DemandTestData:
    treatment_grid: np.ndarray
    structural_outcome: np.ndarray


class DemandScenario(AbstractScenario):
    """PCI demand DGP with oracle/MAR/naive modes."""

    VALID_MODES = {"oracle", "mar_modified", "mar_naive"}

    def __init__(self, mode: str = "mar_modified"):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self._splits_mar: Dict[str, PVTrainDataSetMAR] = {
            "train": None,
            "dev": None,
        }
        self._test_data: DemandTestData | None = None

    def _generate_raw(self, num_data: int, seed: int) -> Tuple[np.ndarray, ...]:
        rng = np.random.default_rng(seed)
        demand = rng.uniform(0.0, 10.0, size=(num_data, 1))
        eps1 = rng.normal(0.0, 1.0, size=(num_data, 1))
        eps2 = rng.normal(0.0, 1.0, size=(num_data, 1))
        eps3 = rng.normal(0.0, 1.0, size=(num_data, 1))
        eps4 = rng.normal(0.0, 1.0, size=(num_data, 1))
        eps5 = rng.normal(0.0, 1.0, size=(num_data, 1))

        cost1 = 2.0 * np.sin(demand * 2.0 * np.pi / 10.0) + eps1
        cost2 = 2.0 * np.cos(demand * 2.0 * np.pi / 10.0) + eps2
        treatment = 35.0 + (cost1 + 3.0) * _psi(demand) + cost2 + eps3
        outcome_proxy = 7.0 * _psi(demand) + 45.0 + eps4
        outcome = np.clip(np.exp((outcome_proxy - treatment) / 10.0), 0.0, 5.0) * treatment
        outcome = outcome - 5.0 * _psi(demand) + eps5
        treatment_proxy = np.concatenate([cost1, cost2], axis=1)
        return treatment, treatment_proxy, outcome_proxy, outcome

    @staticmethod
    def _mar_delta(
        treatment: np.ndarray,
        treatment_proxy: np.ndarray,
        outcome: np.ndarray,
        missing_rate: float,
        seed: int,
    ) -> np.ndarray:
        l_plus = np.concatenate([treatment, treatment_proxy, outcome], axis=1)
        l_plus = (l_plus - l_plus.mean(0, keepdims=True)) / (l_plus.std(0, keepdims=True) + 1e-8)
        alpha = np.array([1.6, 0.8, -0.8, 1.2], dtype=np.float64).reshape(-1, 1)
        score = l_plus @ alpha

        target_obs = 1.0 - missing_rate
        lo, hi = -15.0, 15.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            probs = 1.0 / (1.0 + np.exp(-(score + mid)))
            if probs.mean() > target_obs:
                hi = mid
            else:
                lo = mid
        probs = 1.0 / (1.0 + np.exp(-(score + 0.5 * (lo + hi))))

        rng = np.random.default_rng(seed + 717)
        delta = (rng.uniform(size=probs.shape) < probs).astype(np.float64)
        return delta

    def generate_data(
        self, num_data: int, missing_rate: float = 0.3, seed: int = 42
    ) -> PVTrainDataSetMAR:
        treatment, treatment_proxy, outcome_proxy, outcome = self._generate_raw(num_data, seed=seed)

        if self.mode == "oracle":
            delta_w = np.ones((num_data, 1), dtype=np.float64)
            observed_w = outcome_proxy.copy()
        else:
            delta_w = self._mar_delta(
                treatment=treatment,
                treatment_proxy=treatment_proxy,
                outcome=outcome,
                missing_rate=missing_rate,
                seed=seed,
            )
            observed_w = outcome_proxy.copy()
            observed_w[delta_w < 0.5] = 0.0

        return PVTrainDataSetMAR(
            treatment=treatment.astype(np.float64),
            treatment_proxy=treatment_proxy.astype(np.float64),
            outcome_proxy=observed_w.astype(np.float64),
            outcome=outcome.astype(np.float64),
            backdoor=None,
            delta_w=delta_w.astype(np.float64),
        )

    def _estimate_structural_curve(
        self, treatment_grid: np.ndarray, num_mc: int = 20000, seed: int = 123
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        demand = rng.uniform(0.0, 10.0, size=(num_mc, 1))
        eps4 = rng.normal(0.0, 1.0, size=(num_mc, 1))
        eps5 = rng.normal(0.0, 1.0, size=(num_mc, 1))
        views = 7.0 * _psi(demand) + 45.0 + eps4
        out = []
        for a in treatment_grid.flatten():
            y = np.clip(np.exp((views - a) / 10.0), 0.0, 5.0) * a - 5.0 * _psi(demand) + eps5
            out.append(float(y.mean()))
        return np.asarray(out, dtype=np.float64).reshape(-1, 1)

    def setup(
        self,
        num_train: int,
        num_dev: int = 0,
        num_test: int = 10,
        missing_rate: float = 0.3,
        seed: int = 42,
    ) -> None:
        self._splits_mar["train"] = self.generate_data(
            num_data=num_train, missing_rate=missing_rate, seed=seed
        )
        if num_dev > 0:
            self._splits_mar["dev"] = self.generate_data(
                num_data=num_dev, missing_rate=missing_rate, seed=seed + 1
            )

        treatment_grid = np.linspace(20.0, 60.0, num=max(num_test, 2)).reshape(-1, 1)
        structural_outcome = self._estimate_structural_curve(treatment_grid=treatment_grid, seed=seed + 2)
        self._test_data = DemandTestData(
            treatment_grid=treatment_grid.astype(np.float64),
            structural_outcome=structural_outcome.astype(np.float64),
        )
        self.initialized = True

    def get_train_data(self):
        if self._splits_mar["train"] is None:
            raise LookupError("Scenario is not set up")
        return self._splits_mar["train"]

    def get_dev_data(self):
        if self._splits_mar["dev"] is None:
            raise LookupError("Dev split unavailable")
        return self._splits_mar["dev"]

    def get_test_data(self):
        if self._test_data is None:
            raise LookupError("Scenario is not set up")
        return self._test_data

