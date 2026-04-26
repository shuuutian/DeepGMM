from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch


class PVTrainDataSetMAR(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: Optional[np.ndarray]
    delta_w: np.ndarray

    def validate(self) -> None:
        n = self.outcome.shape[0]
        arrays = [
            self.treatment,
            self.treatment_proxy,
            self.outcome_proxy,
            self.outcome,
            self.delta_w,
        ]
        if self.backdoor is not None:
            arrays.append(self.backdoor)
        for arr in arrays:
            if arr.shape[0] != n:
                raise ValueError("All arrays must share first dimension")


class PVTrainDataSetMARTorch(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: Optional[torch.Tensor]
    delta_w: torch.Tensor

    @classmethod
    def from_numpy(cls, data: PVTrainDataSetMAR) -> "PVTrainDataSetMARTorch":
        data.validate()
        backdoor = None
        if data.backdoor is not None:
            backdoor = torch.as_tensor(data.backdoor).double()
        return cls(
            treatment=torch.as_tensor(data.treatment).double(),
            treatment_proxy=torch.as_tensor(data.treatment_proxy).double(),
            outcome_proxy=torch.as_tensor(data.outcome_proxy).double(),
            outcome=torch.as_tensor(data.outcome).double(),
            backdoor=backdoor,
            delta_w=torch.as_tensor(data.delta_w).double(),
        )

    def to_gpu(self) -> "PVTrainDataSetMARTorch":
        if not torch.cuda.is_available():
            return self
        backdoor = None
        if self.backdoor is not None:
            backdoor = self.backdoor.cuda()
        return PVTrainDataSetMARTorch(
            treatment=self.treatment.cuda(),
            treatment_proxy=self.treatment_proxy.cuda(),
            outcome_proxy=self.outcome_proxy.cuda(),
            outcome=self.outcome.cuda(),
            backdoor=backdoor,
            delta_w=self.delta_w.cuda(),
        )

    def subset(self, idx: Sequence[int]) -> "PVTrainDataSetMARTorch":
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.treatment.device)
        backdoor = None
        if self.backdoor is not None:
            backdoor = self.backdoor[idx_t]
        return PVTrainDataSetMARTorch(
            treatment=self.treatment[idx_t],
            treatment_proxy=self.treatment_proxy[idx_t],
            outcome_proxy=self.outcome_proxy[idx_t],
            outcome=self.outcome[idx_t],
            backdoor=backdoor,
            delta_w=self.delta_w[idx_t],
        )

    def num_data(self) -> int:
        return int(self.outcome.shape[0])


def create_k_folds(
    data: PVTrainDataSetMARTorch, n_folds: int, seed: int
) -> List[torch.Tensor]:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    n = data.num_data()
    if n < n_folds:
        raise ValueError("n_folds cannot exceed number of observations")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    splits = np.array_split(perm, n_folds)
    device = data.outcome.device
    return [torch.as_tensor(split, dtype=torch.long, device=device) for split in splits]


def get_train_val_split(
    data: PVTrainDataSetMARTorch, fold_indices: List[torch.Tensor], val_fold: int
) -> Tuple[PVTrainDataSetMARTorch, PVTrainDataSetMARTorch]:
    if val_fold < 0 or val_fold >= len(fold_indices):
        raise IndexError("val_fold out of range")
    val_idx = fold_indices[val_fold]
    train_parts = [fold_indices[i] for i in range(len(fold_indices)) if i != val_fold]
    train_idx = torch.cat(train_parts, dim=0)
    return data.subset(train_idx.tolist()), data.subset(val_idx.tolist())

