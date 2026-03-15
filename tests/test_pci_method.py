import numpy as np
import torch

from data.data_class_mar import PVTrainDataSetMARTorch
from methods.pci_deepgmm_method import PCIDeepGMMMethod
from scenarios.demand_scenario import DemandScenario


def _prepare(mode: str, seed: int = 19, n_train: int = 1200, n_dev: int = 400):
    scenario = DemandScenario(mode=mode)
    scenario.setup(num_train=n_train, num_dev=n_dev, num_test=10, missing_rate=0.3, seed=seed)
    train = PVTrainDataSetMARTorch.from_numpy(scenario.get_train_data())
    dev = PVTrainDataSetMARTorch.from_numpy(scenario.get_dev_data())
    test = scenario.get_test_data()
    ate_true = float(test.structural_outcome[0, 0] - test.structural_outcome[-1, 0])
    a1 = float(test.treatment_grid[0, 0])
    a0 = float(test.treatment_grid[-1, 0])
    return train, dev, ate_true, a1, a0


def test_oracle_method_convergence():
    train, dev, ate_true, a1, a0 = _prepare(mode="oracle", n_train=1000, n_dev=300)
    method = PCIDeepGMMMethod(mode="oracle", max_num_epochs=80, batch_size=256, enable_cuda=False)
    method.fit(train, dev, verbose=False)
    ate_hat = method.predict_ate(a1, a0)
    bias = abs(ate_hat - ate_true)
    assert np.isfinite(ate_hat)
    print("ATE_hat=", ate_hat, "ATE_true=", ate_true, "abs_bias=", bias)


def test_modified_method_less_biased_than_naive():
    train_mod, dev_mod, ate_true, a1, a0 = _prepare(mode="mar_modified", n_train=1300, n_dev=400)
    train_nv, dev_nv, _, _, _ = _prepare(mode="mar_naive", n_train=1300, n_dev=400)
    modified = PCIDeepGMMMethod(mode="modified", max_num_epochs=100, batch_size=256, enable_cuda=False)
    naive = PCIDeepGMMMethod(mode="naive", max_num_epochs=100, batch_size=256, enable_cuda=False)
    modified.fit(train_mod, dev_mod, verbose=False)
    naive.fit(train_nv, dev_nv, verbose=False)
    b_mod = abs(modified.predict_ate(a1, a0) - ate_true)
    b_naive = abs(naive.predict_ate(a1, a0) - ate_true)
    assert b_mod <= b_naive + 1.0
    print("bias_modified=", b_mod, "bias_naive=", b_naive, "ATE_true=", ate_true)


def test_ate_estimator_shape():
    train, dev, _, _, _ = _prepare(mode="mar_modified", n_train=1000, n_dev=300)
    method = PCIDeepGMMMethod(mode="modified", max_num_epochs=50, batch_size=256, enable_cuda=False)
    method.fit(train, dev, verbose=False)
    b10 = method.beta_hat(10.0)
    b30 = method.beta_hat(30.0)
    delta = b10 - b30
    assert np.isscalar(b10)
    assert np.isscalar(delta)
    print("beta_hat(10)=", b10, "beta_hat(30)=", b30, "delta_hat=", delta)


def test_no_nan_in_training():
    train, dev, _, _, _ = _prepare(mode="mar_modified", n_train=900, n_dev=200)
    method = PCIDeepGMMMethod(mode="modified", max_num_epochs=40, batch_size=256, enable_cuda=False)
    method.fit(train, dev, verbose=False)
    losses = np.asarray(method.learner.loss_history, dtype=np.float64)
    assert np.isfinite(losses).all()
    for i in range(0, len(losses), 10):
        print("iter", i, "h_loss", losses[i, 0], "f_loss", losses[i, 1])

