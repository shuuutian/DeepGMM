import torch
import torch.nn as nn

from game_objectives.pci_moment_objective import PCIOptimalMomentObjective
from models.mlp_model import MLPModel


def _build_models():
    h = MLPModel(input_dim=2, layer_widths=[16], activation=nn.LeakyReLU).double()
    f = MLPModel(input_dim=3, layer_widths=[16], activation=nn.LeakyReLU).double()
    h.initialize()
    f.initialize()
    return h, f


def _toy_batch(n=128):
    treatment = torch.randn(n, 1, dtype=torch.double)
    treatment_proxy = torch.randn(n, 2, dtype=torch.double)
    outcome_proxy = torch.randn(n, 1, dtype=torch.double)
    outcome = torch.randn(n, 1, dtype=torch.double)
    return treatment, treatment_proxy, outcome_proxy, outcome


def test_oracle_game_objective_shape():
    h, f = _build_models()
    objective = PCIOptimalMomentObjective(lambda_1=0.0)
    treatment, treatment_proxy, outcome_proxy, outcome = _toy_batch()
    delta_w = torch.ones_like(outcome)
    g_obj, f_obj = objective.calc_objective(
        h, f, treatment, treatment_proxy, outcome_proxy, outcome, delta_w
    )
    assert g_obj.ndim == 0
    assert f_obj.ndim == 0
    assert torch.allclose(g_obj + f_obj, torch.zeros_like(g_obj), atol=1e-10)
    print("g_obj=", float(g_obj), "f_obj=", float(f_obj))


def test_imputed_vs_oracle_objective_consistency():
    h, f = _build_models()
    objective = PCIOptimalMomentObjective(lambda_1=0.25)
    treatment, treatment_proxy, outcome_proxy, outcome = _toy_batch()
    delta_w = torch.ones_like(outcome)
    m = torch.randn_like(torch.squeeze(outcome))
    v = torch.rand_like(torch.squeeze(outcome))
    g1, f1 = objective.calc_objective(h, f, treatment, treatment_proxy, outcome_proxy, outcome, delta_w)
    g2, f2 = objective.calc_objective(
        h, f, treatment, treatment_proxy, outcome_proxy, outcome, delta_w, m_theta=m, v_theta_bar=v
    )
    assert torch.allclose(g1, g2, atol=1e-10)
    assert torch.allclose(f1, f2, atol=1e-10)
    print("difference:", float((g1 - g2).abs()))


def test_imputed_objective_with_missing():
    h, f = _build_models()
    objective = PCIOptimalMomentObjective(lambda_1=0.25)
    treatment, treatment_proxy, outcome_proxy, outcome = _toy_batch()
    n = outcome.shape[0]
    delta_w = torch.ones_like(outcome)
    delta_w[: n // 2] = 0.0
    m_theta = torch.randn(n, dtype=torch.double)
    v_theta_bar = torch.rand(n, dtype=torch.double) + 0.1
    g_imp, f_imp = objective.calc_objective(
        h,
        f,
        treatment,
        treatment_proxy,
        outcome_proxy,
        outcome,
        delta_w,
        m_theta=m_theta,
        v_theta_bar=v_theta_bar,
    )

    obs = torch.squeeze(delta_w) > 0.5
    g_naive, f_naive = objective.calc_objective(
        h,
        f,
        treatment[obs],
        treatment_proxy[obs],
        outcome_proxy[obs],
        outcome[obs],
        torch.ones_like(outcome[obs]),
    )
    assert torch.isfinite(g_imp)
    assert torch.isfinite(f_imp)
    assert (g_imp - g_naive).abs() > 1e-8 or (f_imp - f_naive).abs() > 1e-8
    print("imputed:", float(g_imp), "naive:", float(g_naive), "diff:", float((g_imp - g_naive).abs()))

