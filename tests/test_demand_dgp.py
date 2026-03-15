import numpy as np

from scenarios.demand_scenario import DemandScenario


def test_generate_oracle_data():
    scenario = DemandScenario(mode="oracle")
    data = scenario.generate_data(num_data=1000, missing_rate=0.3, seed=7)

    assert data.treatment.shape == (1000, 1)
    assert data.treatment_proxy.shape == (1000, 2)
    assert data.outcome_proxy.shape == (1000, 1)
    assert np.allclose(data.delta_w, 1.0)
    print(
        "oracle stats:",
        data.treatment.mean(),
        data.treatment.std(),
        data.treatment_proxy.mean(),
        data.outcome_proxy.mean(),
        data.outcome.mean(),
    )


def test_generate_mar_data():
    scenario = DemandScenario(mode="mar_modified")
    data = scenario.generate_data(num_data=2000, missing_rate=0.3, seed=11)

    observed_rate = float(data.delta_w.mean())
    assert 0.65 <= observed_rate <= 0.75
    assert np.allclose(data.outcome_proxy[data.delta_w < 0.5], 0.0)
    print("observed_rate:", observed_rate, "missing_rate:", 1.0 - observed_rate)


def test_mar_mechanism_uses_l_plus():
    scenario = DemandScenario(mode="mar_modified")
    data = scenario.generate_data(num_data=4000, missing_rate=0.3, seed=13)
    y = data.outcome.squeeze()
    delta = data.delta_w.squeeze()
    med = np.median(y)
    p_hi = float(delta[y > med].mean())
    p_lo = float(delta[y <= med].mean())
    assert abs(p_hi - p_lo) > 0.02
    print("Pr(delta=1|Y>median)=", p_hi, "Pr(delta=1|Y<=median)=", p_lo)


def test_structural_ate():
    scenario = DemandScenario(mode="oracle")
    scenario.setup(num_train=100, num_dev=50, num_test=10, seed=17)
    test = scenario.get_test_data()
    assert test.treatment_grid.shape == (10, 1)
    assert np.isfinite(test.structural_outcome).all()
    assert float(test.structural_outcome[0, 0]) > float(test.structural_outcome[-1, 0])
    print("ATE grid:", test.structural_outcome.squeeze().tolist())

