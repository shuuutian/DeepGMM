from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from data.data_class_mar import PVTrainDataSetMARTorch
from methods.pci_deepgmm_method import PCIDeepGMMMethod
from scenarios.demand_scenario import DemandScenario


def _one_rep(seed: int, n_train: int, n_dev: int, missing_rate: float, use_cuda: bool) -> List[Dict[str, float]]:
    out = []
    modes = [("oracle", "oracle"), ("modified", "mar_modified"), ("naive", "mar_naive")]
    for method_mode, scenario_mode in modes:
        scenario = DemandScenario(mode=scenario_mode)
        scenario.setup(num_train=n_train, num_dev=n_dev, num_test=10, missing_rate=missing_rate, seed=seed)
        train_np = scenario.get_train_data()
        dev_np = scenario.get_dev_data()
        test = scenario.get_test_data()
        ate_true = float(test.structural_outcome[0, 0] - test.structural_outcome[-1, 0])

        method = PCIDeepGMMMethod(
            mode=method_mode,
            n_folds=5,
            missing_rate=missing_rate,
            enable_cuda=use_cuda,
            max_num_epochs=250,
            batch_size=256,
        )
        method.fit(PVTrainDataSetMARTorch.from_numpy(train_np), PVTrainDataSetMARTorch.from_numpy(dev_np), verbose=False)
        ate_hat = method.predict_ate(a1=float(test.treatment_grid[0, 0]), a0=float(test.treatment_grid[-1, 0]))
        out.append(
            {
                "rep": seed,
                "method": method_mode,
                "ate_hat": float(ate_hat),
                "ate_true": ate_true,
                "bias": float(ate_hat - ate_true),
            }
        )
    return out


def _write_csv(path: str, rows: List[Dict[str, float]], header: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare oracle/modified/naive PCI DeepGMM.")
    parser.add_argument("--n-rep", type=int, default=50)
    parser.add_argument("--missing-rate", type=float, default=0.3)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-dev", type=int, default=800)
    parser.add_argument("--dump-root", type=str, default="dumps")
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.dump_root, f"compare_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    use_cuda = (not args.no_cuda)
    reps = list(range(args.n_rep))
    results_nested = Parallel(n_jobs=args.num_cpus)(
        delayed(_one_rep)(seed, args.n_train, args.n_dev, args.missing_rate, use_cuda) for seed in reps
    )
    results = [item for sub in results_nested for item in sub]

    _write_csv(
        os.path.join(out_dir, "results.csv"),
        results,
        header=["rep", "method", "ate_hat", "ate_true", "bias"],
    )

    summary = []
    for method in sorted(set(row["method"] for row in results)):
        biases = np.asarray([row["bias"] for row in results if row["method"] == method], dtype=np.float64)
        summary.append(
            {
                "method": method,
                "mean_bias": float(np.mean(biases)),
                "std_bias": float(np.std(biases)),
                "rmse": float(np.sqrt(np.mean(biases**2))),
            }
        )
    _write_csv(os.path.join(out_dir, "summary.csv"), summary, header=["method", "mean_bias", "std_bias", "rmse"])

    plt.figure(figsize=(8, 5))
    for method in sorted(set(row["method"] for row in results)):
        biases = np.asarray([row["bias"] for row in results if row["method"] == method], dtype=np.float64)
        plt.hist(biases, bins=20, density=True, alpha=0.35, label=method)
    plt.xlabel("ATE bias")
    plt.ylabel("Density")
    plt.title("Bias distribution across methods")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bias_distribution.png"), dpi=150)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()

