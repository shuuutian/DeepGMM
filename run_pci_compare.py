from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed

from data.data_class_mar import PVTrainDataSetMARTorch
from methods.original_deepgmm_baseline import OriginalDeepGMMBaseline
from methods.pci_deepgmm_method import PCIDeepGMMMethod
from scenarios.demand_scenario import DemandScenario

_DEFAULT_CONFIG_FILE = Path(__file__).parent / "configs" / "experiment_params.toml"


def _load_config(set_name: str, config_file: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_file) if config_file else _DEFAULT_CONFIG_FILE
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    sets = raw.get("sets", {})
    if set_name not in sets:
        raise KeyError(f"Config set '{set_name}' not found in {path}. "
                       f"Available: {list(sets.keys())}")
    params = dict(raw.get("defaults", {}))
    params.update(sets[set_name])
    return params


def _one_rep(
    seed: int,
    n_train: int,
    n_dev: int,
    missing_rate: float,
    max_num_epochs: int,
    batch_size: int,
    n_folds: int,
    use_cuda: bool,
    burn_in: int = 1000,
    lr_grid: Optional[List[float]] = None,
    selection_epochs: int = 2000,
) -> Dict[str, List]:
    results = []
    diagnostics = []  # per-epoch rows

    # oracle and naive use the original DeepGMM objective + model selection;
    # modified uses our PCI-adapted objective with MAR imputation.
    MODES = [
        ("oracle",   "oracle",       "original"),
        ("modified", "mar_modified", "pci"),
        ("naive",    "mar_naive",    "original"),
    ]

    for method_mode, scenario_mode, impl in MODES:
        scenario = DemandScenario(mode=scenario_mode)
        scenario.setup(num_train=n_train, num_dev=n_dev, num_test=10,
                       missing_rate=missing_rate, seed=seed)
        train_data = PVTrainDataSetMARTorch.from_numpy(scenario.get_train_data())
        dev_data   = PVTrainDataSetMARTorch.from_numpy(scenario.get_dev_data())
        test       = scenario.get_test_data()
        ate_true   = float(test.structural_outcome[0, 0] - test.structural_outcome[-1, 0])
        a1 = float(test.treatment_grid[0, 0])
        a0 = float(test.treatment_grid[-1, 0])

        if impl == "original":
            # oracle / naive: use original OptimalMomentObjective + model selection
            method = OriginalDeepGMMBaseline(
                mode=method_mode,
                max_num_epochs=max_num_epochs,
                batch_size=batch_size,
                burn_in=burn_in,
                lr_grid=lr_grid,
                selection_epochs=selection_epochs,
                enable_cuda=use_cuda,
            )
            method.fit(train_data, dev_data, verbose=False)
            ate_hat = method.predict_ate(a1, a0)
            beta_a1 = method.beta_hat(a1)
            beta_a0 = method.beta_hat(a0)
            loss_history = method.loss_history
            dev_moment_history = method.dev_moment_history
            best_ckpt_epoch = method.best_checkpoint_epoch
        else:
            # modified: use our PCI method with MAR-imputed residuals
            method = PCIDeepGMMMethod(
                mode="modified",
                n_folds=n_folds,
                missing_rate=missing_rate,
                enable_cuda=use_cuda,
                max_num_epochs=max_num_epochs,
                batch_size=batch_size,
            )
            method.fit(train_data, dev_data, verbose=False)
            ate_hat = method.predict_ate(a1, a0)
            beta_a1 = method.beta_hat(a1)
            beta_a0 = method.beta_hat(a0)
            loss_history = method.learner.loss_history
            dev_moment_history = method.learner.dev_moment_history
            best_ckpt_epoch = method.learner.best_checkpoint_epoch

        results.append({
            "rep": seed,
            "method": method_mode,
            "ate_hat": float(ate_hat),
            "ate_true": ate_true,
            "bias": float(ate_hat - ate_true),
            "beta_a1": float(beta_a1),
            "beta_a0": float(beta_a0),
            "best_checkpoint_epoch": best_ckpt_epoch,
        })

        # per-epoch loss curve
        dev_moment_dict = dict(dev_moment_history)
        for epoch, (h_loss, f_loss) in enumerate(loss_history):
            dev_moment = dev_moment_dict.get(epoch, float("nan"))
            diagnostics.append({
                "rep": seed,
                "method": method_mode,
                "epoch": epoch,
                "h_loss": h_loss,
                "f_loss": f_loss,
                "dev_moment": dev_moment,
            })

    return {"results": results, "diagnostics": diagnostics}


def _write_csv(path: str, rows: List[Dict], header: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _resolve(cli_val, config: Dict[str, Any], key: str, fallback):
    """Return cli_val if explicitly set (not None), else config value, else fallback."""
    if cli_val is not None:
        return cli_val
    return config.get(key, fallback)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare oracle/modified/naive PCI DeepGMM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── config source ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--config", type=str, default=None, metavar="SET_NAME",
        help="Load parameters from configs/experiment_params.toml. "
             "One of: smoke, standard, compare, ablation_missing, lr_sensitivity.",
    )
    parser.add_argument(
        "--config-file", type=str, default=None, metavar="PATH",
        help="Path to a custom TOML config file (used together with --config).",
    )

    # ── runtime args (all default None so CLI can override config) ────────────
    parser.add_argument("--n-rep",             type=int,   default=None)
    parser.add_argument("--missing-rate",      type=float, default=None)
    parser.add_argument("--n-train",           type=int,   default=None)
    parser.add_argument("--n-dev",             type=int,   default=None)
    parser.add_argument("--n-folds",           type=int,   default=None)
    parser.add_argument("--max-epochs",        type=int,   default=None)
    parser.add_argument("--batch-size",        type=int,   default=None)
    parser.add_argument("--burn-in",           type=int,   default=None)
    parser.add_argument("--selection-epochs",  type=int,   default=None)
    parser.add_argument("--no-lr-grid",        action="store_true",
                        help="Disable LR grid search for oracle/naive (use single fixed LR).")
    parser.add_argument("--dump-root",         type=str,   default=None)
    parser.add_argument("--num-cpus",          type=int,   default=None)
    parser.add_argument("--no-cuda",           action="store_true")

    args = parser.parse_args()

    # ── resolve parameters: config base → CLI overrides ───────────────────────
    config: Dict[str, Any] = {}
    if args.config:
        config = _load_config(args.config, args.config_file)
        print(f"Loaded config set '{args.config}'"
              + (f" from {args.config_file}" if args.config_file else ""))

    n_rep            = _resolve(args.n_rep,           config, "n_rep",            50)
    missing_rate     = _resolve(args.missing_rate,    config, "missing_rate",     0.3)
    n_train          = _resolve(args.n_train,         config, "n_train",          2000)
    n_dev            = _resolve(args.n_dev,           config, "n_dev",            800)
    n_folds          = _resolve(args.n_folds,         config, "n_folds",          5)
    max_num_epochs   = _resolve(args.max_epochs,      config, "max_num_epochs",   30000)
    batch_size       = _resolve(args.batch_size,      config, "batch_size",       1024)
    burn_in          = _resolve(args.burn_in,         config, "burn_in",          1000)
    selection_epochs = _resolve(args.selection_epochs,config, "selection_epochs", 2000)
    dump_root        = _resolve(args.dump_root,       config, "dump_root",        "dumps")
    num_cpus         = _resolve(args.num_cpus,        config, "num_cpus",         10)
    use_cuda         = not args.no_cuda
    lr_grid: Optional[List[float]] = (
        None if args.no_lr_grid else config.get("lr_grid", [5e-4, 2e-4, 1e-4])
    )

    # ── print resolved parameters ─────────────────────────────────────────────
    print(
        f"n_rep={n_rep}  n_train={n_train}  n_dev={n_dev}  "
        f"missing_rate={missing_rate}  max_epochs={max_num_epochs}  "
        f"batch={batch_size}  burn_in={burn_in}  folds={n_folds}  "
        f"selection_epochs={selection_epochs}  lr_grid={lr_grid}  cpus={num_cpus}"
    )

    # ── run ───────────────────────────────────────────────────────────────────
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.config if args.config else "custom"
    out_dir = os.path.join(dump_root, f"compare_{label}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    rep_outputs = Parallel(n_jobs=num_cpus)(
        delayed(_one_rep)(
            seed, n_train, n_dev, missing_rate,
            max_num_epochs, batch_size, n_folds, use_cuda,
            burn_in=burn_in,
            lr_grid=lr_grid,
            selection_epochs=selection_epochs,
        )
        for seed in range(n_rep)
    )
    results = [row for rep in rep_outputs for row in rep["results"]]
    diagnostics = [row for rep in rep_outputs for row in rep["diagnostics"]]

    _write_csv(
        os.path.join(out_dir, "results.csv"),
        results,
        header=["rep", "method", "ate_hat", "ate_true", "bias",
                "beta_a1", "beta_a0", "best_checkpoint_epoch"],
    )
    _write_csv(
        os.path.join(out_dir, "diagnostics.csv"),
        diagnostics,
        header=["rep", "method", "epoch", "h_loss", "f_loss", "dev_moment"],
    )

    summary = []
    for method in sorted({row["method"] for row in results}):
        biases = np.asarray([row["bias"] for row in results if row["method"] == method])
        summary.append({
            "method":    method,
            "mean_bias": float(np.mean(biases)),
            "std_bias":  float(np.std(biases)),
            "rmse":      float(np.sqrt(np.mean(biases ** 2))),
        })
    _write_csv(
        os.path.join(out_dir, "summary.csv"),
        summary,
        header=["method", "mean_bias", "std_bias", "rmse"],
    )

    _color_map = {
        "modified": "#B221E2",
        "naive":    "#DD8452",
        "oracle":   "#55A868",
    }
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9, 5.6))
    all_bias_vals = np.concatenate([
        np.asarray([row["bias"] for row in results if row["method"] == m])
        for m in sorted({row["method"] for row in results})
    ])
    _pad = 0.1 * max(1e-8, float(all_bias_vals.max()) - float(all_bias_vals.min()))
    x_grid = np.linspace(float(all_bias_vals.min()) - _pad,
                         float(all_bias_vals.max()) + _pad, 500)
    for method in sorted({row["method"] for row in results}):
        biases = np.asarray([row["bias"] for row in results if row["method"] == method])
        color = _color_map.get(method)
        ax.hist(biases, bins=18, density=True, alpha=0.22,
                edgecolor=color, linewidth=1.6, color=color, label=method)
        if biases.size >= 2 and np.std(biases) > 0:
            ax.plot(x_grid, gaussian_kde(biases)(x_grid), color=color, linewidth=2.0)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title(f"Bias = ATE_hat - ATE_true  (missing_rate={missing_rate})")
    ax.set_xlabel("Bias")
    ax.set_ylabel("Density")
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bias_distribution.png"), dpi=160)
    plt.close(fig)
    plt.style.use("default")

    # ── loss-curve and dev-moment plots (median ± IQR across reps) ───────────
    methods_seen = sorted({row["method"] for row in diagnostics})
    fig, axes = plt.subplots(len(methods_seen), 2,
                             figsize=(12, 4 * len(methods_seen)), squeeze=False)
    for mi, method in enumerate(methods_seen):
        rows = [r for r in diagnostics if r["method"] == method]
        epochs = sorted({r["epoch"] for r in rows})

        # h_loss per epoch
        h_losses = np.array([[r["h_loss"] for r in rows if r["epoch"] == e] for e in epochs])
        ax = axes[mi][0]
        ax.plot(epochs, np.median(h_losses, axis=1), label="h_loss median")
        q25, q75 = np.percentile(h_losses, 25, axis=1), np.percentile(h_losses, 75, axis=1)
        ax.fill_between(epochs, q25, q75, alpha=0.2)
        ax.set_title(f"{method} — h_loss")
        ax.set_xlabel("epoch"); ax.set_ylabel("h_loss")

        # dev_moment (only non-nan rows)
        dm_rows = [r for r in rows if not np.isnan(r["dev_moment"])]
        if dm_rows:
            dm_epochs = sorted({r["epoch"] for r in dm_rows})
            dm_vals = np.array([[r["dev_moment"] for r in dm_rows if r["epoch"] == e]
                                for e in dm_epochs])
            ax2 = axes[mi][1]
            ax2.plot(dm_epochs, np.median(dm_vals, axis=1), label="dev_moment median", color="orange")
            q25d = np.percentile(dm_vals, 25, axis=1)
            q75d = np.percentile(dm_vals, 75, axis=1)
            ax2.fill_between(dm_epochs, q25d, q75d, alpha=0.2, color="orange")
            ax2.set_title(f"{method} — dev MSE")
            ax2.set_xlabel("epoch"); ax2.set_ylabel("MSE")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
