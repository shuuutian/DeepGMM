"""
Lightweight config loader for PCI DeepGMM experiments.

Usage
-----
    from configs.load_config import load_params

    params = load_params("standard")       # Set B
    params = load_params("smoke")          # Set A
    params = load_params("compare")        # Set C
    params = load_params("ablation_missing")
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Dict

_CONFIG_PATH = Path(__file__).parent / "experiment_params.toml"


def _load_raw() -> Dict[str, Any]:
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def load_params(set_name: str) -> Dict[str, Any]:
    """
    Return a merged dict: defaults overridden by the named set.

    Parameters
    ----------
    set_name : str
        One of 'smoke', 'standard', 'compare', 'ablation_missing',
        'lr_sensitivity'.

    Returns
    -------
    dict
        Flat dict of all parameters for the requested experiment set.
    """
    raw = _load_raw()
    available = list(raw.get("sets", {}).keys())
    if set_name not in raw.get("sets", {}):
        raise KeyError(f"Unknown set '{set_name}'. Available: {available}")

    params = dict(raw.get("defaults", {}))
    params.update(raw["sets"][set_name])
    params["_set_name"] = set_name
    return params


def load_all_sets() -> Dict[str, Dict[str, Any]]:
    """Return all named sets, each merged with defaults."""
    raw = _load_raw()
    return {name: load_params(name) for name in raw.get("sets", {})}


def print_params(set_name: str) -> None:
    """Pretty-print a parameter set."""
    params = load_params(set_name)
    width = max(len(k) for k in params)
    print(f"\n── {set_name} ──")
    for k, v in params.items():
        print(f"  {k:<{width}} = {v}")
    print()


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "standard"
    print_params(name)
