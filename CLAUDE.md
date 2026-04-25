# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands must be run from the repo root via `uv run` (no manual venv activation).

**Run all PCI tests:**
```bash
uv run pytest tests/ -v
```

**Run a single test file or function:**
```bash
uv run pytest tests/test_demand_dgp.py -v
uv run pytest tests/test_pci_method.py::test_oracle_method_convergence -v
```

**Run the full comparison experiment (local, 1 CPU):**
```bash
uv run python run_pci_compare.py --n-rep 5 --missing-rate 0.3 --n-train 2000 --no-cuda
```

**Submit to Spartan HPC:**
```bash
sbatch slurm_job.sh   # runs 100 reps, 8 CPUs, on /home/wzzho2/DeepGMM
```

**Install dependencies:**
```bash
uv pip install -e .   # uses pyproject.toml (numpy, scipy, torch, matplotlib, scikit-learn, tensorflow)
```

There is no linter or formatter configured.

## Architecture

This repo contains two coexisting codebases:

### 1. Original DeepGMM (Bennett et al. 2019)
The original IV framework for general nonparametric IV regression. Entry points are `run_mnist_experiments_*.py` and `run_zoo_experiments_ours.py`. Uses `scenarios/toy_scenarios.py` and `scenarios/mnist_scenarios.py` for DGPs. The data structure is a plain `Dataset` object (fields: `x, z, y, g, w`) defined in `scenarios/abstract_scenario.py`.

The reference original DeepGMM commit is `c2a62ed` (the merge from `CausalML/DeepGMM`, immediately before the MAR fork at `15453ca`). Walk-throughs and theory anchoring reference this hash.

### 2. Modified DeepGMM for Proximal Causal Inference (PCI) — the active research extension
Implements the min-max estimator for the PCI bridge function $h_\theta(W,A)$ under Missing-at-Random (MAR) outcome proxy $W$. All new code lives in:

| Module | Role |
|--------|------|
| `data/data_class_mar.py` | `PVTrainDataSetMAR` (numpy) and `PVTrainDataSetMARTorch` (torch) named tuples; `create_k_folds` / `get_train_val_split` for cross-fitting |
| `scenarios/demand_scenario.py` | Demand DGP: $A$ = price, $Z$ = (cost1, cost2), $W$ = views, $Y$ = sales. Three modes: `oracle` / `mar_modified` / `mar_naive` |
| `game_objectives/pci_moment_objective.py` | `PCIOptimalMomentObjective`: computes $U(\theta,\tau)$ with imputed residuals $\tilde{r}_i(\theta)$ and $\tilde{s}_i(\bar\theta)$; returns `(h_loss, f_loss)` both to be minimised |
| `learning/learning_pci.py` | `SGDLearningPCI`: alternating SGD loop, cross-fit linear imputation models ($\hat{m}_\theta$, $\hat{v}_{\bar\theta}$), EMA update for $\bar\theta$, `estimate_beta` ATE estimator |
| `methods/pci_deepgmm_method.py` | `PCIDeepGMMMethod`: top-level `fit` / `beta_hat` / `predict_ate` API; wires together scenario, objective, learner |
| `run_pci_compare.py` | Monte Carlo comparison script; outputs `dumps/compare_<timestamp>/{results.csv, summary.csv, bias_distribution.png}` |

### Game objective sign conventions
`calc_objective` always returns `(h_loss, f_loss)` where **both are minimised**. The generator $h_\theta$ minimises the moment $\hat{U}$; the critic $f_\tau$ minimises $-\hat{U} + \text{f\_reg}$ (i.e., maximises $\hat{U}$). This is the same convention as `OptimalMomentObjective` in the original codebase.

### Cross-fitting and $\bar\theta$ (EMA)
`SGDLearningPCI` fits two ridge-regression imputation models per fold at each epoch:
- $\hat{m}_\theta$: predicts $r_i(\theta) = Y_i - h_\theta(W_i, A_i)$ for observed units, features = $L^+ = (A, Z, Y)$
- $\hat{v}_{\bar\theta}$: predicts $r_i(\bar\theta)^2$, using the EMA-frozen model $\bar\theta$

EMA update (correct direction): `theta_bar ← (1 - ema_alpha) * theta_bar + ema_alpha * theta` with `ema_alpha=0.05`.

### Model inputs
- $h_\theta$: `input_dim=2` — concatenation of $(W, A)$, both scalar in the demand DGP
- $f_\tau$: `input_dim=3` — concatenation of $L = (A, Z)$ where $Z$ is 2-dimensional

### ATE estimation
`estimate_beta(data, a_value)` implements the cross-fit estimator:
- Observed units ($\delta_W=1$): use $h_\theta(W_i, a)$ directly
- Missing units ($\delta_W=0$): use a cross-fit linear $\hat{q}_{a,\theta}^{(-k)}(L^+_i)$ trained on observed units in the complementary fold

### Reference files (outside this repo)
| Purpose | Path |
|---------|------|
| Original demand DGP | `/Users/apple/DeepFeatureProxyVariable/src/data/ate/demand_pv.py` |
| MAR data class reference | `/Users/apple/DeepFeatureProxyVariable/src/data/ate/data_class_mar.py` |
| Comparison script reference | `/Users/apple/DeepFeatureProxyVariable/scripts/compare_dfpv_variants.py` |
| Working paper (LaTeX) | `/Users/apple/2602_WenPaper/PCI_paper_draft_overleaf/main.tex` |
| Spartan remote root | `/home/wzzho2/DeepGMM` |

## Validation-protocol artefacts

Work on this repo follows the systematic validation protocol at the Notion page **0425 - Systematic Validation Plan** (https://www.notion.so/34d64b068dfc80259c89d01feee5fc3c). Read it before starting any new validation iteration.

Conventions defined there and used throughout the repo:

- **Four canonical run roles** on identical DGP and seeds:
  - `B` — original DeepGMM, naive on the observed-only subsample (lower bound).
  - `O_orig` — original DeepGMM on full data, no missingness (upper bound).
  - `O_mar` — MAR-DeepGMM on full data, no missingness (upper bound; sanity check).
  - `M` — MAR-DeepGMM on the partial-data setting (the run under test, replaced each iteration).
- **Git tagging.** Each run tags the code before executing: `git tag run/<role>/<YYYYMMDD-NN>`. No tag, no run.
- **Output folder.** `dumps/<YYYYMMDD-HHMMSS>__<git-tag>__<role>/` containing `results.csv`, `config.json`, `env.lock`, `README.md`.
- **Research diary.** `RESEARCH_DIARY.md` at repo root is the chronological per-run record. Consulted before designing any new sub-experiment to avoid re-running prior trials. (File created on the first run of Phase 1, not yet present.)

Walk-throughs and the §4.2 theory-anchoring map live as separate pages in the same Notion Research database.
