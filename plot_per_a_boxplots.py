"""Re-plot per-a bias box-plots for one or more dump directories.

Reads each dump's `predictions.csv` (full per-grid-point β̂ values) when
available. For older dumps that predate the predictions.csv writer, falls
back to `results.csv` and plots only the two endpoints (a1, a0) that were
saved per rep — clearly labelled in the figure title as a partial view.

Usage
-----
    uv run python plot_per_a_boxplots.py dumps/<dump1> [<dump2> ...]
    uv run python plot_per_a_boxplots.py --out-dir figures/ dumps/*

Each dump produces one PNG written next to (or into) the dump.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

# Re-use the helper + canonical colors/order from run_pci_compare.py so the
# look matches the in-run plot exactly.
from run_pci_compare import (  # noqa: E402
    METHOD_ORDER,
    METHOD_COLORS,
    _plot_per_a_boxplots,
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _predictions_from_results(results_rows: List[Dict[str, str]]
                              ) -> List[Dict[str, Any]]:
    """Fallback: build pseudo-predictions from a results.csv that only has
    `beta_a1` and `beta_a0` (endpoints). Two rows per (rep, method, dgp).

    The structural curve β(a) is *not* in results.csv. We back out
    β(a1) and β(a0) from `beta_a1` and `bias` only when the relationship is
    consistent — but `bias = ate_hat − ate_true = (β̂(a1) − β̂(a0)) −
    (β(a1) − β(a0))` mixes both endpoints, so we cannot recover
    β(a1) and β(a0) individually. Instead we emit only `beta_hat` rows and
    let the plotting function ignore β(a)-relative bias by computing it
    against an unknown true → here we use 0 as a placeholder for β(a),
    making the y-axis "β̂(a)" rather than "β̂(a) − β(a)".

    To keep the plot meaningful in fallback mode, we recompute the structural
    curve directly from the demand DGP (closed-form Monte Carlo) for the
    two saved endpoints. This requires DemandScenario; for toy_sin we use
    AGMMZoo._true_g_function_np directly.
    """
    import numpy as np

    # Lazy imports — only needed in fallback mode.
    from scenarios.demand_scenario import DemandScenario
    from scenarios.toy_scenarios import AGMMZoo

    # Pull (a1, a0) per DGP from the rows; assume they're constant across reps
    # (true for the current pipeline).
    dgps = sorted({r["dgp"] for r in results_rows})

    # Cache: dgp → (a1, a0, beta_a1_true, beta_a0_true)
    truth: Dict[str, Dict[str, float]] = {}
    for dgp in dgps:
        # Find any rep's first row to read endpoint values used in saving.
        # The endpoints are fixed per DGP / scenario, so any rep is fine.
        sample = next(r for r in results_rows if r["dgp"] == dgp)
        # We can't read a1/a0 directly from the row — they aren't saved.
        # But we know them by convention:
        if dgp == "demand":
            scen = DemandScenario(mode="oracle")
            scen.setup(num_train=10, num_dev=0, num_test=10,
                       missing_rate=0.0, seed=0)
            test = scen.get_test_data()
            a1 = float(test.treatment_grid[0, 0])
            a0 = float(test.treatment_grid[-1, 0])
            beta_a1_true = float(test.structural_outcome[0, 0])
            beta_a0_true = float(test.structural_outcome[-1, 0])
        elif dgp == "toy_sin":
            scen = AGMMZoo(g_function="sin")
            a1, a0 = 1.0, -1.0
            beta_a1_true = float(
                scen._true_g_function_np(np.array([[a1]])).flat[0]
            )
            beta_a0_true = float(
                scen._true_g_function_np(np.array([[a0]])).flat[0]
            )
        else:
            # Skip unknown DGPs gracefully.
            continue
        truth[dgp] = {
            "a1": a1, "a0": a0,
            "beta_a1_true": beta_a1_true, "beta_a0_true": beta_a0_true,
        }

    out: List[Dict[str, Any]] = []
    for r in results_rows:
        dgp = r["dgp"]
        if dgp not in truth:
            continue
        t = truth[dgp]
        beta_a1 = float(r["beta_a1"])
        beta_a0 = float(r["beta_a0"])
        # Two pseudo-rows per (rep, method, dgp): one per endpoint.
        out.append({
            "rep":       int(r["rep"]),
            "method":    r["method"],
            "dgp":       dgp,
            "a_idx":     0,
            "a":         t["a1"],
            "beta_hat":  beta_a1,
            "beta_true": t["beta_a1_true"],
            "bias_at_a": beta_a1 - t["beta_a1_true"],
        })
        out.append({
            "rep":       int(r["rep"]),
            "method":    r["method"],
            "dgp":       dgp,
            "a_idx":     1,
            "a":         t["a0"],
            "beta_hat":  beta_a0,
            "beta_true": t["beta_a0_true"],
            "bias_at_a": beta_a0 - t["beta_a0_true"],
        })
    return out


def _load_predictions(dump_dir: Path) -> tuple[List[Dict[str, Any]], bool]:
    """Return (predictions, is_full_grid). Full-grid means predictions.csv
    was present; otherwise fallback was used (endpoints only)."""
    pred_path = dump_dir / "predictions.csv"
    if pred_path.exists():
        rows = _read_csv(pred_path)
        # Coerce numeric fields.
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "rep":       int(r["rep"]),
                "method":    r["method"],
                "dgp":       r["dgp"],
                "a_idx":     int(r["a_idx"]),
                "a":         float(r["a"]),
                "beta_hat":  float(r["beta_hat"]),
                "beta_true": float(r["beta_true"]),
                "bias_at_a": float(r["bias_at_a"]),
            })
        return out, True

    res_path = dump_dir / "results.csv"
    if not res_path.exists():
        raise FileNotFoundError(
            f"Neither predictions.csv nor results.csv in {dump_dir}"
        )
    res_rows = _read_csv(res_path)
    return _predictions_from_results(res_rows), False


def _label_for_dump(dump_dir: Path) -> str:
    """A short label drawn from the dump folder name, e.g. 'task0_10000ep'."""
    name = dump_dir.name
    # Try to extract a tagged suffix like __task0_10000ep
    if "__" in name:
        tail = name.rsplit("__", 1)[1]
        return tail
    return name


def _missing_rate_from_dump(dump_dir: Path) -> Optional[float]:
    """Best-effort: read missing_rate from config.json if present."""
    cfg = dump_dir / "config.json"
    if not cfg.exists():
        return None
    try:
        import json
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("missing_rate"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot per-a bias box-plots for one or more dump dirs.",
    )
    parser.add_argument(
        "dumps", nargs="+", type=str,
        help="Dump directories (each must contain predictions.csv "
             "or results.csv).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory; default writes the PNG inside each dump.",
    )
    parser.add_argument(
        "--out-suffix", type=str, default="per_a_boxplots",
        help="Filename stem (PNG written as <stem>.png or "
             "<stem>__<dump_label>.png if --out-dir is set).",
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir) if args.out_dir else None
    if out_root is not None:
        out_root.mkdir(parents=True, exist_ok=True)

    for dump_str in args.dumps:
        dump_dir = Path(dump_str)
        if not dump_dir.is_dir():
            print(f"[skip] not a directory: {dump_dir}")
            continue
        try:
            predictions, is_full_grid = _load_predictions(dump_dir)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue

        if not predictions:
            print(f"[skip] no usable rows in {dump_dir}")
            continue

        label = _label_for_dump(dump_dir)
        missing_rate = _missing_rate_from_dump(dump_dir)
        suffix = label + (
            "  [FULL 10-pt grid]" if is_full_grid
            else "  [PARTIAL: endpoints only]"
        )

        if out_root is not None:
            out_path = out_root / f"{args.out_suffix}__{label}.png"
        else:
            stem = args.out_suffix
            if not is_full_grid:
                stem = f"{args.out_suffix}_endpoints_only"
            out_path = dump_dir / f"{stem}.png"

        _plot_per_a_boxplots(
            predictions,
            out_path=str(out_path),
            missing_rate=missing_rate,
            title_suffix=suffix,
        )
        print(f"[ok ] {dump_dir.name} → {out_path}  "
              f"({'full' if is_full_grid else 'endpoints-only'})")


if __name__ == "__main__":
    main()
