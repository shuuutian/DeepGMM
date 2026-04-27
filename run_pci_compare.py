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
import torch
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed

from data.data_class_mar import PVTrainDataSetMARTorch
from methods.original_deepgmm_baseline import OriginalDeepGMMBaseline
from methods.pci_deepgmm_method import PCIDeepGMMMethod
from methods.toy_model_modified_deepgmm_method import ToyModelModifiedDeepGMMMethod
from scenarios.demand_scenario import DemandScenario
from scenarios.toy_scenarios import AGMMZoo


# ── Deterministic-threshold MAR mechanism for toy DGP (W = x) ─────────────────
# Mirrors run_pci_compare_toy.py:_make_delta_w. L+ = (A, Z, Y); standardise then
# binary-search a threshold so (score <= τ).mean() == 1 - missing_rate.
def _toy_make_delta_w(
    x: np.ndarray, z: np.ndarray, y: np.ndarray,
    missing_rate: float, mar_alpha_value: float = 1.6,
) -> np.ndarray:
    l_plus = np.concatenate([x, z, y], axis=1)
    l_tilde = (l_plus - l_plus.mean(0, keepdims=True)) / (l_plus.std(0, keepdims=True) + 1e-8)
    alpha = np.full(l_tilde.shape[1], mar_alpha_value, dtype=np.float64)
    score = l_tilde @ alpha
    target_obs = 1.0 - missing_rate
    lo, hi = float(score.min()) - 1.0, float(score.max()) + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (score <= mid).mean() > target_obs:
            hi = mid
        else:
            lo = mid
    threshold = 0.5 * (lo + hi)
    return (score <= threshold).astype(np.float64).reshape(-1, 1)

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

    # Eight canonical runs per rep — same four roles on each of two DGPs.
    # Tuple shape: (label, dgp, scenario_arg, impl, internal_mode)
    #
    # Demand DGP: stochastic logistic-MAR on outcome_proxy, applied inside
    # DemandScenario.generate_data via mar_modified / mar_naive scenario modes.
    #   B      → OriginalDeepGMMBaseline(naive)  on scenario.mar_naive
    #   O_orig → OriginalDeepGMMBaseline(oracle) on scenario.oracle (full data)
    #   O_mar  → PCIDeepGMMMethod(oracle)        on scenario.oracle (full data)
    #   M      → PCIDeepGMMMethod(modified)      on scenario.mar_modified
    #
    # Toy `sin` DGP: AGMMZoo(g_function="sin"); W := x (treatment), so
    # `outcome_proxy = treatment` in the PCI mapping. Deterministic-threshold
    # MAR on L+ = (A, Z, Y) computed once per rep and shared across all four
    # toy modes (paired comparison).
    #   B / O_orig         → ToyModelSelectionMethod (subset / full)
    #   O_mar / M          → PCIDeepGMMMethod (oracle / modified) with W = x
    # Dispatched through ToyModelModifiedDeepGMMMethod.
    MODES = [
        ("baseline",         "demand",  "mar_naive",    "original", "naive"),
        ("oracle_baseline",  "demand",  "oracle",       "original", "oracle"),
        ("oracle_modified",  "demand",  "oracle",       "pci",      "oracle"),
        ("modified",         "demand",  "mar_modified", "pci",      "modified"),
        ("baseline",         "toy_sin", "sin",          "toy",      "naive"),
        ("oracle_baseline",  "toy_sin", "sin",          "toy",      "oracle"),
        ("oracle_modified",  "toy_sin", "sin",          "toy",      "oracle_mar"),
        ("modified",         "toy_sin", "sin",          "toy",      "modified"),
    ]

    # ── pre-generate toy DGP data once per rep (shared across 4 toy modes) ───
    np.random.seed(seed); torch.manual_seed(seed)
    toy_scenario = AGMMZoo(g_function="sin")
    x_tr, z_tr, y_tr, _g_tr, _ = toy_scenario.generate_data(n_train)
    np.random.seed(seed + 1); torch.manual_seed(seed + 1)
    x_dv, z_dv, y_dv, _g_dv, _ = toy_scenario.generate_data(n_dev)
    dw_tr = _toy_make_delta_w(x_tr, z_tr, y_tr, missing_rate)
    dw_dv = _toy_make_delta_w(x_dv, z_dv, y_dv, missing_rate)
    toy_a1, toy_a0 = 1.0, -1.0
    toy_ate_true = float(
        toy_scenario._true_g_function_np(np.array([[toy_a1]])).flat[0]
        - toy_scenario._true_g_function_np(np.array([[toy_a0]])).flat[0]
    )

    def _t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float64)
    toy_x_tr_t, toy_z_tr_t, toy_y_tr_t, toy_dw_tr_t = _t(x_tr), _t(z_tr), _t(y_tr), _t(dw_tr)
    toy_x_dv_t, toy_z_dv_t, toy_y_dv_t, toy_dw_dv_t = _t(x_dv), _t(z_dv), _t(y_dv), _t(dw_dv)

    for label, dgp, scenario_arg, impl, internal_mode in MODES:
        # ── load / select data per DGP ────────────────────────────────────────
        if dgp == "demand":
            scenario = DemandScenario(mode=scenario_arg)
            scenario.setup(num_train=n_train, num_dev=n_dev, num_test=10,
                           missing_rate=missing_rate, seed=seed)
            train_data = PVTrainDataSetMARTorch.from_numpy(scenario.get_train_data())
            dev_data   = PVTrainDataSetMARTorch.from_numpy(scenario.get_dev_data())
            test       = scenario.get_test_data()
            ate_true   = float(test.structural_outcome[0, 0] - test.structural_outcome[-1, 0])
            a1 = float(test.treatment_grid[0, 0])
            a0 = float(test.treatment_grid[-1, 0])
        elif dgp == "toy_sin":
            ate_true = toy_ate_true
            a1, a0 = toy_a1, toy_a0
        else:
            raise ValueError(f"Unknown dgp: {dgp}")

        # ── dispatch method ───────────────────────────────────────────────────
        loss_history: List = []
        dev_moment_history: List = []

        if impl == "original":
            method = OriginalDeepGMMBaseline(
                mode=internal_mode,
                max_num_epochs=max_num_epochs,
                batch_size=batch_size,
                burn_in=burn_in,
                lr_grid=lr_grid,
                selection_epochs=selection_epochs,
                enable_cuda=use_cuda,
            )
            method.fit(train_data, dev_data, verbose=False)
            ate_hat = float(method.predict_ate(a1, a0))
            beta_a1 = float(method.beta_hat(a1))
            beta_a0 = float(method.beta_hat(a0))
            loss_history = method.loss_history
            dev_moment_history = method.dev_moment_history
            best_ckpt_epoch = method.best_checkpoint_epoch
        elif impl == "pci":
            method = PCIDeepGMMMethod(
                mode=internal_mode,
                n_folds=n_folds,
                missing_rate=missing_rate,
                enable_cuda=use_cuda,
                max_num_epochs=max_num_epochs,
                batch_size=batch_size,
            )
            method.fit(train_data, dev_data, verbose=False)
            ate_hat = float(method.predict_ate(a1, a0))
            beta_a1 = float(method.beta_hat(a1))
            beta_a0 = float(method.beta_hat(a0))
            loss_history = method.learner.loss_history
            dev_moment_history = method.learner.dev_moment_history
            best_ckpt_epoch = method.learner.best_checkpoint_epoch
        elif impl == "toy":
            method = ToyModelModifiedDeepGMMMethod(
                mode=internal_mode,
                n_folds=n_folds,
                missing_rate=missing_rate,
                enable_cuda=use_cuda,
                max_num_epochs=max_num_epochs,
                batch_size=batch_size,
            )
            method.fit(toy_x_tr_t, toy_z_tr_t, toy_y_tr_t, toy_dw_tr_t,
                       toy_x_dv_t, toy_z_dv_t, toy_y_dv_t, toy_dw_dv_t,
                       verbose=False)
            ate_hat = float(method.predict_ate(a1, a0))
            beta_a1 = float(method.beta_hat(a1))
            beta_a0 = float(method.beta_hat(a0))
            loss_history = method.loss_history
            dev_moment_history = method.dev_moment_history
            best_ckpt_epoch = method.best_checkpoint_epoch
        else:
            raise ValueError(f"Unknown impl: {impl}")

        results.append({
            "rep": seed,
            "method": label,
            "dgp": dgp,
            "ate_hat":  ate_hat,
            "ate_true": ate_true,
            "bias":     ate_hat - ate_true,
            "beta_a1":  beta_a1,
            "beta_a0":  beta_a0,
            "best_checkpoint_epoch": best_ckpt_epoch,
        })

        # per-epoch loss curve (loss_history is empty for toy oracle/naive,
        # populated for everything else — including toy oracle_mar / modified
        # via the underlying PCIDeepGMMMethod)
        if loss_history:
            dev_moment_dict = dict(dev_moment_history)
            for epoch, (h_loss, f_loss) in enumerate(loss_history):
                dev_moment = dev_moment_dict.get(epoch, float("nan"))
                diagnostics.append({
                    "rep": seed,
                    "method": label,
                    "dgp": dgp,
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
        description="Compare four canonical roles (B / O_orig / O_mar / M) on the demand DGP.",
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
        header=["rep", "method", "dgp", "ate_hat", "ate_true", "bias",
                "beta_a1", "beta_a0", "best_checkpoint_epoch"],
    )
    _write_csv(
        os.path.join(out_dir, "diagnostics.csv"),
        diagnostics,
        header=["rep", "method", "dgp", "epoch", "h_loss", "f_loss", "dev_moment"],
    )

    summary = []
    method_dgp_pairs = sorted({(row["method"], row["dgp"]) for row in results})
    for method, dgp in method_dgp_pairs:
        biases = np.asarray([row["bias"] for row in results
                             if row["method"] == method and row["dgp"] == dgp])
        summary.append({
            "method":    method,
            "dgp":       dgp,
            "mean_bias": float(np.mean(biases)),
            "std_bias":  float(np.std(biases)),
            "rmse":      float(np.sqrt(np.mean(biases ** 2))),
        })
    _write_csv(
        os.path.join(out_dir, "summary.csv"),
        summary,
        header=["method", "dgp", "mean_bias", "std_bias", "rmse"],
    )

    _color_map = {
        "baseline":        "#DD8452",
        "oracle_baseline": "#55A868",
        "oracle_modified": "#4C72B0",
        "modified":        "#B221E2",
    }
    # Bias distribution: one panel per DGP (scales differ markedly between
    # demand and toy_sin so a single overlay would compress the toy panel).
    dgps_seen = sorted({row["dgp"] for row in results})
    if dgps_seen:
        plt.style.use("ggplot")
        fig, axes = plt.subplots(len(dgps_seen), 1,
                                 figsize=(9, 4.5 * len(dgps_seen)), squeeze=False)
        for di, dgp in enumerate(dgps_seen):
            ax = axes[di][0]
            dgp_rows = [row for row in results if row["dgp"] == dgp]
            methods_in_dgp = sorted({row["method"] for row in dgp_rows})
            all_bias_vals = np.concatenate([
                np.asarray([row["bias"] for row in dgp_rows if row["method"] == m])
                for m in methods_in_dgp
            ])
            _pad = 0.1 * max(1e-8, float(all_bias_vals.max()) - float(all_bias_vals.min()))
            x_grid = np.linspace(float(all_bias_vals.min()) - _pad,
                                 float(all_bias_vals.max()) + _pad, 500)
            for method in methods_in_dgp:
                biases = np.asarray([row["bias"] for row in dgp_rows
                                     if row["method"] == method])
                color = _color_map.get(method)
                ax.hist(biases, bins=18, density=True, alpha=0.22,
                        edgecolor=color, linewidth=1.6, color=color, label=method)
                if biases.size >= 2 and np.std(biases) > 0:
                    ax.plot(x_grid, gaussian_kde(biases)(x_grid),
                            color=color, linewidth=2.0)
            ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
            ax.set_title(f"{dgp} — Bias = ATE_hat - ATE_true  "
                         f"(missing_rate={missing_rate})")
            ax.set_xlabel("Bias")
            ax.set_ylabel("Density")
            ax.legend(title="Method")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "bias_distribution.png"), dpi=160)
        plt.close(fig)
        plt.style.use("default")

    # ── loss-curve and dev-moment plots (median ± IQR across reps) ───────────
    method_dgp_pairs_seen = sorted({(r["method"], r["dgp"]) for r in diagnostics})
    if method_dgp_pairs_seen:
        fig, axes = plt.subplots(len(method_dgp_pairs_seen), 2,
                                 figsize=(12, 4 * len(method_dgp_pairs_seen)),
                                 squeeze=False)
        for mi, (method, dgp) in enumerate(method_dgp_pairs_seen):
            rows = [r for r in diagnostics if r["method"] == method and r["dgp"] == dgp]
            epochs = sorted({r["epoch"] for r in rows})

            h_losses = np.array([[r["h_loss"] for r in rows if r["epoch"] == e] for e in epochs])
            ax = axes[mi][0]
            ax.plot(epochs, np.median(h_losses, axis=1), label="h_loss median")
            q25, q75 = np.percentile(h_losses, 25, axis=1), np.percentile(h_losses, 75, axis=1)
            ax.fill_between(epochs, q25, q75, alpha=0.2)
            ax.set_title(f"{dgp} / {method} — h_loss")
            ax.set_xlabel("epoch"); ax.set_ylabel("h_loss")

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
                ax2.set_title(f"{dgp} / {method} — dev MSE")
                ax2.set_xlabel("epoch"); ax2.set_ylabel("MSE")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
        plt.close(fig)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
