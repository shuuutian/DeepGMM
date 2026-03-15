import numpy as np
import torch
import torch.nn as nn

from data.data_class_mar import PVTrainDataSetMAR, PVTrainDataSetMARTorch, create_k_folds
from game_objectives.pci_moment_objective import PCIOptimalMomentObjective
from learning.learning_pci import SGDLearningPCI
from models.mlp_model import MLPModel
from optimizers.oadam import OAdam
from optimizers.optimizer_factory import OptimizerFactory


def _make_learner():
    h = MLPModel(input_dim=2, layer_widths=[8], activation=nn.LeakyReLU).double()
    f = MLPModel(input_dim=3, layer_widths=[8], activation=nn.LeakyReLU).double()
    h.initialize()
    f.initialize()
    return SGDLearningPCI(
        game_objective=PCIOptimalMomentObjective(),
        h=h,
        f=f,
        h_optimizer=OptimizerFactory(OAdam, lr=1e-3, betas=(0.5, 0.9))(h),
        f_optimizer=OptimizerFactory(OAdam, lr=2e-3, betas=(0.5, 0.9))(f),
        n_folds=5,
        max_num_epochs=5,
        batch_size=64,
    )


def test_imputation_model_shapes():
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=(n, 4))
    y = (x[:, :1] @ np.array([[2.0]]) + 0.1 * rng.normal(size=(n, 1))).astype(np.float64)
    learner = _make_learner()
    model = learner._fit_linear_model(torch.as_tensor(x).double(), torch.as_tensor(y).double())
    pred = model.predict(torch.as_tensor(x).double())
    mse = float(((pred - torch.as_tensor(y).double()) ** 2).mean())
    assert pred.shape == (n, 1)
    print("train MSE:", mse)


def test_imputed_residual_unbiasedness():
    rng = np.random.default_rng(1)
    n = 8000
    lplus = rng.normal(size=(n, 4))
    true_r = (
        0.7 * lplus[:, 0:1]
        - 0.3 * lplus[:, 1:2]
        + 0.4 * lplus[:, 2:3]
        + 0.1 * rng.normal(size=(n, 1))
    ).astype(np.float64)
    obs_prob = 1.0 / (1.0 + np.exp(-(0.8 * lplus[:, 0:1] - 0.6 * lplus[:, 2:3])))
    delta = (rng.uniform(size=(n, 1)) < obs_prob).astype(np.float64)

    data = PVTrainDataSetMARTorch.from_numpy(
        PVTrainDataSetMAR(
            treatment=lplus[:, :1],
            treatment_proxy=lplus[:, 1:3],
            outcome_proxy=np.zeros((n, 1), dtype=np.float64),
            outcome=lplus[:, 3:4],
            backdoor=None,
            delta_w=delta,
        )
    )
    learner = _make_learner()
    learner.fold_indices = create_k_folds(data, n_folds=5, seed=3)
    r_t = torch.as_tensor(true_r.squeeze()).double()
    m_hat, _ = learner._cross_fit_models(data, residual=r_t, residual_sq=r_t.pow(2))
    r_tilde = torch.squeeze(data.delta_w) * r_t + (1.0 - torch.squeeze(data.delta_w)) * m_hat

    e_tilde = float(r_tilde.mean())
    e_true = float(r_t.mean())
    assert abs(e_tilde - e_true) < 0.05
    print("E[r_tilde]=", e_tilde, "E[r]=", e_true, "bias=", e_tilde - e_true)


def test_cross_fit_no_data_leakage():
    rng = np.random.default_rng(2)
    n = 500
    data = PVTrainDataSetMARTorch.from_numpy(
        PVTrainDataSetMAR(
            treatment=rng.normal(size=(n, 1)),
            treatment_proxy=rng.normal(size=(n, 2)),
            outcome_proxy=rng.normal(size=(n, 1)),
            outcome=rng.normal(size=(n, 1)),
            backdoor=None,
            delta_w=np.ones((n, 1), dtype=np.float64),
        )
    )
    learner = _make_learner()
    learner.fold_indices = create_k_folds(data, n_folds=5, seed=10)
    residual = torch.randn(n, dtype=torch.double)
    learner._cross_fit_models(data, residual=residual, residual_sq=residual.pow(2))
    for k, val_idx in enumerate(learner.fold_indices):
        train_idx = set(learner.get_fold_training_indices()[k].tolist())
        overlap = train_idx.intersection(set(val_idx.tolist()))
        assert len(overlap) == 0
        print("fold", k, "train_size", len(train_idx), "val_size", len(val_idx))

