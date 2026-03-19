from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from data.data_class_mar import PVTrainDataSetMARTorch
from game_objectives.pci_moment_objective import PCIOptimalMomentObjective
from learning.learning_pci import SGDLearningPCI
from models.mlp_model import MLPModel
from optimizers.oadam import OAdam
from optimizers.optimizer_factory import OptimizerFactory


class PCIDeepGMMMethod:
    def __init__(
        self,
        mode: str = "modified",
        n_folds: int = 5,
        missing_rate: float = 0.3,
        enable_cuda: bool = False,
        max_num_epochs: int = 300,
        batch_size: int = 256,
    ):
        if mode not in {"oracle", "modified", "naive"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.n_folds = n_folds
        self.missing_rate = missing_rate
        self.enable_cuda = enable_cuda
        self.max_num_epochs = max_num_epochs
        self.batch_size = batch_size
        self.h: Optional[torch.nn.Module] = None
        self.f: Optional[torch.nn.Module] = None
        self.learner: Optional[SGDLearningPCI] = None
        self.train_data_: Optional[PVTrainDataSetMARTorch] = None

    def _prepare_data(self, data: PVTrainDataSetMARTorch) -> PVTrainDataSetMARTorch:
        if self.mode == "oracle":
            return PVTrainDataSetMARTorch(
                treatment=data.treatment,
                treatment_proxy=data.treatment_proxy,
                outcome_proxy=data.outcome_proxy,
                outcome=data.outcome,
                backdoor=data.backdoor,
                delta_w=torch.ones_like(data.delta_w),
            )
        if self.mode == "naive":
            keep = torch.squeeze(data.delta_w) > 0.5
            naive = data.subset(torch.where(keep)[0].tolist())
            return PVTrainDataSetMARTorch(
                treatment=naive.treatment,
                treatment_proxy=naive.treatment_proxy,
                outcome_proxy=naive.outcome_proxy,
                outcome=naive.outcome,
                backdoor=naive.backdoor,
                delta_w=torch.ones_like(naive.delta_w),
            )
        return data

    def fit(
        self,
        train_data: PVTrainDataSetMARTorch,
        dev_data: Optional[PVTrainDataSetMARTorch] = None,
        verbose: bool = False,
    ) -> None:
        train_data = self._prepare_data(train_data)
        if self.enable_cuda and torch.cuda.is_available():
            train_data = train_data.to_gpu()

        h = MLPModel(input_dim=2, layer_widths=[64, 64], activation=nn.LeakyReLU).double()
        f = MLPModel(input_dim=3, layer_widths=[64, 64], activation=nn.LeakyReLU).double()
        if self.enable_cuda and torch.cuda.is_available():
            h = h.cuda()
            f = f.cuda()
        h.initialize()
        f.initialize()

        h_opt = OptimizerFactory(OAdam, lr=2e-4, betas=(0.5, 0.9))(h)
        f_opt = OptimizerFactory(OAdam, lr=8e-4, betas=(0.5, 0.9))(f)
        objective = PCIOptimalMomentObjective(lambda_1=0.25)
        learner = SGDLearningPCI(
            game_objective=objective,
            h=h,
            f=f,
            h_optimizer=h_opt,
            f_optimizer=f_opt,
            n_folds=self.n_folds,
            max_num_epochs=self.max_num_epochs,
            batch_size=self.batch_size,
        )
        learner.fit(train_data=train_data, dev_data=dev_data, verbose=verbose)

        self.h = h
        self.f = f
        self.learner = learner
        self.train_data_ = train_data

    def beta_hat(self, a_value: float) -> float:
        if self.learner is None or self.train_data_ is None:
            raise AttributeError("fit must be called before beta_hat")
        return self.learner.estimate_beta(self.train_data_, a_value)

    def predict_ate(self, a1: float, a0: float) -> float:
        return float(self.beta_hat(a1) - self.beta_hat(a0))

