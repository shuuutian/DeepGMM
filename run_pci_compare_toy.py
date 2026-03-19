"""
Monte Carlo comparison of oracle / naive / modified on the original toy DGP.

DGP
---
AGMMZoo (configurable g_function: 'sin', 'linear', 'abs', '2dpoly', 'sigmoid', 'step').
  x  (n, 1)  treatment
  z  (n, 2)  instruments (treatment proxy)
  y  (n, 1)  outcome   y = g(x) + confounding + noise
  g  (n, 1)  true structural values g(x)
  w  (n, 1)  outcome proxy  =  x   (same variable in AGMMZoo)

MAR mechanism  (updated spec — DGP_DEMAND.md §3)
---------------------------------------------------
L⁺ = (A, Z, Y) = (x, z, y).  Standardise column-wise.
score_i = Σ_j  mar_alpha_value · L̃⁺_ij    (uniform coefficient, default 1.6)
Binary-search a threshold τ so that (score ≤ τ).mean() ≈ 1 − missing_rate.
δ_W,i = 1  if score_i ≤ τ  else  0          ← deterministic hard threshold

Methods
-------
oracle   : ToyModelSelectionMethod, full data (δ_W forced 1).
naive    : ToyModelSelectionMethod, complete-case rows (δ_W = 1) only.
modified : PCIDeepGMMMethod  with  W = x,  MAR δ_W.

ATE
---
True ATE  = g_true(a1) − g_true(a0)   (evaluated from the scenario's ground-truth)
Estimated = method.predict_ate(a1, a0)
a1, a0    configurable; defaults  1.0  and  −1.0.

Outputs (in dumps/compare_toy_<label>_<timestamp>/)
----------------------------------------------------
  results.csv       per-rep ATE estimates and bias
  summary.csv       mean bias / std / RMSE per method
  diagnostics.csv   per-epoch loss curves (modified only)
  bias_distribution.png
  training_curves.png
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from joblib import Parallel, delayed

import torch

from methods.toy_model_modified_deepgmm_method import ToyModelModifiedDeepGMMMethod
from scenarios.toy_scenarios import AGMMZoo


# ── MAR mechanism (new deterministic-threshold spec) ─────────────────────────

def _make_delta_w(
    x: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    missing_rate: float,
    mar_alpha_value: float = 1.6,
) -> np.ndarray:
    """
    Deterministic hard-threshold MAR on L⁺ = (A, Z, Y).

    Uniform coefficient mar_alpha_value for all L⁺ columns.
    Binary-search over the threshold (not over a logit bias) to hit the
    target observed fraction 1 − missing_rate exactly on this dataset.
    Returns δ_W  shape (n, 1), dtype float64.
    """
    l_plus = np.concatenate([x, z, y], axis=1)          # (n, 4): A(1)+Z(2)+Y(1)
    l_tilde = (l_plus - l_plus.mean(0, keepdims=True)) / (l_plus.std(0, keepdims=True) + 1e-8)
    alpha = np.full(l_tilde.shape[1], mar_alpha_value, dtype=np.float64)
    score = l_tilde @ alpha                              # (n,)

    target_obs = 1.0 - missing_rate
    lo, hi = float(score.min()) - 1.0, float(score.max()) + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (score <= mid).mean() > target_obs:
            hi = mid
        else:
            lo = mid
    threshold = 0.5 * (lo + hi)
    delta = (score <= threshold).astype(np.float64).reshape(-1, 1)
    return delta


# ── single rep ────────────────────────────────────────────────────────────────

def _one_rep(
    seed: int,
    n_train: int,
    n_dev: int,
    missing_rate: float,
    max_num_epochs: int,
    batch_size: int,
    n_folds: int,
    use_cuda: bool,
    g_function: str,
    a1: float,
    a0: float,
    mar_alpha_value: float,
) -> Dict[str, List]:
    results = []
    diagnostics = []

    # ── generate data ─────────────────────────────────────────────────────────
    scenario = AGMMZoo(g_function=g_function)

    np.random.seed(seed)
    torch.manual_seed(seed)
    x_tr, z_tr, y_tr, g_tr, _ = scenario.generate_data(n_train)

    np.random.seed(seed + 1)
    torch.manual_seed(seed + 1)
    x_dv, z_dv, y_dv, g_dv, _ = scenario.generate_data(n_dev)

    # true ATE from closed-form structural function
    ate_true = float(
        scenario._true_g_function_np(np.array([[a1]])).flat[0]
        - scenario._true_g_function_np(np.array([[a0]])).flat[0]
    )

    # MAR indicators (deterministic threshold)
    dw_tr = _make_delta_w(x_tr, z_tr, y_tr, missing_rate, mar_alpha_value)
    dw_dv = _make_delta_w(x_dv, z_dv, y_dv, missing_rate, mar_alpha_value)

    # convert to torch double tensors
    def _t(a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(a, dtype=torch.float64)

    x_tr_t  = _t(x_tr);  z_tr_t = _t(z_tr);  y_tr_t = _t(y_tr);  dw_tr_t = _t(dw_tr)
    x_dv_t  = _t(x_dv);  z_dv_t = _t(z_dv);  y_dv_t = _t(y_dv);  dw_dv_t = _t(dw_dv)

    # ── run three modes ───────────────────────────────────────────────────────
    for mode in ("oracle", "naive", "modified"):
        method = ToyModelModifiedDeepGMMMethod(
            mode=mode,
            n_folds=n_folds,
            missing_rate=missing_rate,
            enable_cuda=use_cuda,
            max_num_epochs=max_num_epochs,
            batch_size=batch_size,
        )
        method.fit(
            x_tr_t, z_tr_t, y_tr_t, dw_tr_t,
            x_dv_t, z_dv_t, y_dv_t, dw_dv_t,
            verbose=False,
        )

        ate_hat   = method.predict_ate(a1, a0)
        beta_a1   = method.beta_hat(a1)
        beta_a0   = method.beta_hat(a0)
        obs_frac  = float(dw_tr.mean())

        results.append({
            "rep":                   seed,
            "method":                mode,
            "ate_hat":               float(ate_hat),
            "ate_true":              ate_true,
            "bias":                  float(ate_hat - ate_true),
            "beta_a1":               float(beta_a1),
            "beta_a0":               float(beta_a0),
            "obs_frac":              obs_frac,
            "best_checkpoint_epoch": method.best_checkpoint_epoch,
        })

        # per-epoch diagnostics (non-empty only for modified)
        dev_moment_dict = dict(method.dev_moment_history)
        for epoch, (h_loss, f_loss) in enumerate(method.loss_history):
            dev_moment = dev_moment_dict.get(epoch, float("nan"))
            diagnostics.append({
                "rep":        seed,
                "method":     mode,
                "epoch":      epoch,
                "h_loss":     h_loss,
                "f_loss":     f_loss,
                "dev_moment": dev_moment,
            })

    return {"results": results, "diagnostics": diagnostics}


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _write_csv(path: str, rows: List[Dict], header: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare oracle/naive/modified on the original toy DeepGMM DGP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-rep",           type=int,   default=50)
    parser.add_argument("--missing-rate",    type=float, default=0.3)
    parser.add_argument("--n-train",         type=int,   default=2000)
    parser.add_argument("--n-dev",           type=int,   default=800)
    parser.add_argument("--n-folds",         type=int,   default=5)
    parser.add_argument("--max-epochs",      type=int,   default=300)
    parser.add_argument("--batch-size",      type=int,   default=256)
    parser.add_argument("--dump-root",       type=str,   default="dumps")
    parser.add_argument("--num-cpus",        type=int,   default=1)
    parser.add_argument("--no-cuda",         action="store_true")
    parser.add_argument("--g-function",      type=str,   default="sin",
                        choices=["sin", "linear", "abs", "2dpoly", "sigmoid", "step", "3dpoly"])
    parser.add_argument("--a1",              type=float, default=1.0,
                        help="Upper treatment value for ATE = g(a1) - g(a0).")
    parser.add_argument("--a0",              type=float, default=-1.0,
                        help="Lower treatment value for ATE = g(a1) - g(a0).")
    parser.add_argument("--mar-alpha",       type=float, default=1.6,
                        help="Uniform MAR coefficient for all L+ columns.")
    args = parser.parse_args()

    use_cuda = not args.no_cuda

    print(
        f"g_function={args.g_function}  n_rep={args.n_rep}  "
        f"n_train={args.n_train}  n_dev={args.n_dev}  "
        f"missing_rate={args.missing_rate}  max_epochs={args.max_epochs}  "
        f"batch={args.batch_size}  folds={args.n_folds}  "
        f"a1={args.a1}  a0={args.a0}  cpus={args.num_cpus}"
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.dump_root, f"compare_toy_{args.g_function}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    rep_outputs = Parallel(n_jobs=args.num_cpus)(
        delayed(_one_rep)(
            seed, args.n_train, args.n_dev, args.missing_rate,
            args.max_epochs, args.batch_size, args.n_folds, use_cuda,
            args.g_function, args.a1, args.a0, args.mar_alpha,
        )
        for seed in range(args.n_rep)
    )

    results    = [row for rep in rep_outputs for row in rep["results"]]
    diagnostics = [row for rep in rep_outputs for row in rep["diagnostics"]]

    _write_csv(
        os.path.join(out_dir, "results.csv"),
        results,
        header=["rep", "method", "ate_hat", "ate_true", "bias",
                "beta_a1", "beta_a0", "obs_frac", "best_checkpoint_epoch"],
    )
    _write_csv(
        os.path.join(out_dir, "diagnostics.csv"),
        diagnostics,
        header=["rep", "method", "epoch", "h_loss", "f_loss", "dev_moment"],
    )

    # ── summary ───────────────────────────────────────────────────────────────
    summary = []
    for method in ("oracle", "naive", "modified"):
        biases = np.asarray([r["bias"] for r in results if r["method"] == method])
        if biases.size == 0:
            continue
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
    for row in summary:
        print(f"  {row['method']:10s}  mean_bias={row['mean_bias']:+.4f}  "
              f"std={row['std_bias']:.4f}  rmse={row['rmse']:.4f}")

    # ── bias distribution plot ────────────────────────────────────────────────
    _color_map = {"modified": "#B221E2", "naive": "#DD8452", "oracle": "#55A868"}
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9, 5.6))

    all_biases = np.concatenate([
        np.asarray([r["bias"] for r in results if r["method"] == m])
        for m in ("oracle", "naive", "modified")
    ])
    _pad = 0.1 * max(1e-8, float(all_biases.max()) - float(all_biases.min()))
    x_grid = np.linspace(float(all_biases.min()) - _pad,
                         float(all_biases.max()) + _pad, 500)

    for method in ("oracle", "naive", "modified"):
        biases = np.asarray([r["bias"] for r in results if r["method"] == method])
        color = _color_map[method]
        ax.hist(biases, bins=18, density=True, alpha=0.22,
                edgecolor=color, linewidth=1.6, color=color, label=method)
        if biases.size >= 2 and np.std(biases) > 0:
            ax.plot(x_grid, gaussian_kde(biases)(x_grid), color=color, linewidth=2.0)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_title(
        f"Bias = ATE_hat − ATE_true  "
        f"(g={args.g_function}, missing_rate={args.missing_rate}, "
        f"a1={args.a1}, a0={args.a0})"
    )
    ax.set_xlabel("Bias")
    ax.set_ylabel("Density")
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bias_distribution.png"), dpi=160)
    plt.close(fig)
    plt.style.use("default")

    # ── loss / dev-moment curves (modified only, has loss_history) ────────────
    mod_diag = [r for r in diagnostics if r["method"] == "modified" and not np.isnan(r["h_loss"])]
    if mod_diag:
        epochs = sorted({r["epoch"] for r in mod_diag})
        h_mat = np.array([[r["h_loss"]     for r in mod_diag if r["epoch"] == e] for e in epochs])
        dm_rows = [r for r in mod_diag if not np.isnan(r["dev_moment"])]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        ax = axes[0]
        ax.plot(epochs, np.median(h_mat, axis=1), label="h_loss median")
        ax.fill_between(epochs,
                        np.percentile(h_mat, 25, axis=1),
                        np.percentile(h_mat, 75, axis=1), alpha=0.2)
        ax.set_title("modified — h_loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("h_loss")

        if dm_rows:
            dm_epochs = sorted({r["epoch"] for r in dm_rows})
            dm_mat = np.array([[r["dev_moment"] for r in dm_rows if r["epoch"] == e]
                               for e in dm_epochs])
            ax2 = axes[1]
            ax2.plot(dm_epochs, np.median(dm_mat, axis=1), color="orange", label="dev MSE median")
            ax2.fill_between(dm_epochs,
                             np.percentile(dm_mat, 25, axis=1),
                             np.percentile(dm_mat, 75, axis=1), alpha=0.2, color="orange")
            ax2.set_title("modified — dev MSE")
            ax2.set_xlabel("epoch")
            ax2.set_ylabel("MSE")

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
        plt.close(fig)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
