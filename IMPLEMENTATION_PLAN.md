# Implementation Plan: Modified DeepGMM for Proximal Causal Inference under MAR

## Overview

This document outlines the full implementation plan for extending the original DeepGMM codebase
to support **Modified DeepGMM for Proximal Causal Inference (PCI)** with missing outcome proxy W
under Missing-at-Random (MAR). Three experiment conditions are compared:

| Condition      | Description                                           |
|----------------|-------------------------------------------------------|
| **Oracle**     | Full W observed (no missingness), original DeepGMM    |
| **Modified**   | W partially missing, MAR-imputed residuals (our method)|
| **Naive**      | W partially missing, complete-case only (biased)      |

---

## 0. Reference Files

| Purpose              | Path                                                              |
|----------------------|-------------------------------------------------------------------|
| DGP source           | `/Users/apple/DeepFeatureProxyVariable/src/data/ate/demand_pv.py`|
| MAR data class       | `/Users/apple/DeepFeatureProxyVariable/src/data/ate/data_class_mar.py` |
| MAR data generator   | `/Users/apple/DeepFeatureProxyVariable/src/data/ate/__init__.py` (`generate_train_data_ate_mar`) |
| Compare script ref   | `/Users/apple/DeepFeatureProxyVariable/scripts/compare_dfpv_variants.py` |
| Working paper        | `/Users/apple/2602_WenPaper/PCI_paper_draft_overleaf/main.tex`  |
| Spartan remote root  | `/home/wzzho2/DeepGMM`                                           |

---

## 1. New File Structure

Files to create inside `/Users/apple/DeepGMM/` (mirroring existing structure):

```
DeepGMM/
├── scenarios/
│   ├── demand_scenario.py          [NEW] PCI demand DGP (oracle / MAR / naive variants)
│   └── abstract_scenario.py       [EXISTING, unchanged]
│
├── data/
│   ├── __init__.py                 [NEW] empty
│   └── data_class_mar.py           [NEW] PVTrainDataSetMAR, fold utilities (ported from DFPV project)
│
├── game_objectives/
│   └── pci_moment_objective.py     [NEW] MAR-adapted game objective U_θ̄,n(θ,τ) with imputed residuals
│
├── learning/
│   └── learning_pci.py             [NEW] SGD training loop with cross-fit imputation step
│
├── methods/
│   └── pci_deepgmm_method.py       [NEW] top-level method combining all components
│
├── tests/
│   ├── __init__.py                 [NEW]
│   ├── test_demand_dgp.py          [NEW] tests for DGP correctness
│   ├── test_pci_objective.py       [NEW] tests for MAR-adapted game objective
│   ├── test_imputation.py          [NEW] tests for cross-fit imputation (m_θ, v_θ̄)
│   └── test_pci_method.py          [NEW] end-to-end smoke tests + ATE bias diagnostics
│
└── run_pci_compare.py              [NEW] comparison script: oracle / modified / naive
```

---

## 2. Step 1 — DGP: `scenarios/demand_scenario.py`

**Source**: Port from `/Users/apple/DeepFeatureProxyVariable/src/data/ate/demand_pv.py`

### Variable mapping (paper → code)

| Paper symbol | Demand DGP variable | Code name        |
|--------------|---------------------|------------------|
| $A$          | price               | `treatment`      |
| $Z$          | [cost1, cost2]      | `treatment_proxy`|
| $W$          | views               | `outcome_proxy`  |
| $Y$          | sales outcome       | `outcome`        |
| $U$          | demand (latent)     | hidden in DGP    |
| $\delta_W$   | missingness mask    | `delta_w`        |

### DGP equations

The demand DGP follows:

```
demand ~ Uniform(0, 10)                       # latent confounder U
cost1 = 2*sin(demand*2π/10) + ε1,  ε1~N(0,1) # Z[:,0]
cost2 = 2*cos(demand*2π/10) + ε2,  ε2~N(0,1) # Z[:,1]
price = 35 + (cost1+3)*ψ(demand) + cost2 + ε3 # A
views = 7*ψ(demand) + 45 + ε4                 # W
Y = clip(exp((views-price)/10), 0, 5)*price - 5*ψ(demand) + ε5  # outcome
```

where `ψ(t) = 2*((t-5)^4/600 + exp(-4*(t-5)^2) + t/10 - 2)`.

### Three dataset variants from this DGP

```python
class DemandScenario(AbstractScenario):
    """mode in {'oracle', 'mar_modified', 'mar_naive'}"""
    def generate_data(self, num_data, missing_rate=0.3, seed=42): ...
    def get_train_data(self): ...   # returns (treatment, treatment_proxy, outcome_proxy, outcome, delta_w)
    def get_dev_data(self):   ...
    def get_test_data(self):  ...   # returns test grid + structural ATE
```

**MAR mechanism** (ported from `generate_train_data_ate_mar`):
- $L^+ = (A, Z, Y)$ (no backdoor X in demand DGP)
- Logistic score: `score = L+_standardised @ alpha` where `alpha = [1.6, ...]`
- Intercept calibrated by binary search to hit target `missing_rate`
- `delta_w = 1` if observed, `0` if missing; `outcome_proxy[delta_w=0] = 0.0`

---

## 3. Step 2 — Data Class: `data/data_class_mar.py`

Direct port of `PVTrainDataSetMAR` and `PVTrainDataSetMARTorch` from the DFPV project.

Key classes:

```python
class PVTrainDataSetMAR(NamedTuple):
    treatment: np.ndarray        # A, shape (n, d_A)
    treatment_proxy: np.ndarray  # Z, shape (n, d_Z)
    outcome_proxy: np.ndarray    # W (zeros where missing), shape (n, d_W)
    outcome: np.ndarray          # Y, shape (n, 1)
    backdoor: Optional[...]      # X (None for demand)
    delta_w: np.ndarray          # shape (n, 1), float32

class PVTrainDataSetMARTorch(NamedTuple):
    # same fields as above but torch.Tensor
    def from_numpy(cls, data): ...
    def to_gpu(self): ...
    def subset(self, idx): ...

def create_k_folds(data, n_folds, seed) -> List[torch.Tensor]: ...
def get_train_val_split(data, fold_indices, val_fold): ...
```

---

## 4. Step 3 — Tests (write BEFORE implementation)

All tests in `tests/`. Run with `pytest tests/ -v`.

### 4.1 `test_demand_dgp.py`

```
Test: test_generate_oracle_data
  - generate n=1000, mode='oracle'
  - check shapes: treatment (1000,1), treatment_proxy (1000,2), outcome_proxy (1000,1)
  - check delta_w all ones
  - DIAGNOSTIC: print mean/std of each variable

Test: test_generate_mar_data
  - generate n=2000, missing_rate=0.3
  - check delta_w has ~30% ones (tolerance ±5%)
  - check outcome_proxy is 0.0 wherever delta_w=0
  - DIAGNOSTIC: print actual missing rate

Test: test_mar_mechanism_uses_L_plus
  - compare missing rates across high-Y vs low-Y subgroups
  - verify they differ (MAR depends on Y, not MCAR)
  - DIAGNOSTIC: print Pr(delta_w=1 | Y>median) vs Pr(delta_w=1 | Y<=median)

Test: test_structural_ate
  - generate test grid (10 price points)
  - check structural values are finite and decreasing (demand curve)
  - DIAGNOSTIC: print ATE grid
```

### 4.2 `test_pci_objective.py`

```
Test: test_oracle_game_objective_shape
  - construct toy h_theta (MLP), f_tau (MLP), small batch
  - call calc_objective(h, f, treatment, treatment_proxy, outcome_proxy, outcome, delta_w=all_ones)
  - check g_obj, f_obj are scalar tensors
  - check g_obj + f_obj ≈ 0 (zero-sum property when no f_reg)
  - DIAGNOSTIC: print g_obj, f_obj values

Test: test_imputed_vs_oracle_objective_consistency
  - with delta_w=all_ones, MAR-imputed objective should equal oracle objective exactly
  - DIAGNOSTIC: print difference

Test: test_imputed_objective_with_missing
  - set delta_w to 50% missing, provide m_theta estimates
  - verify objective is computable (no NaN/Inf)
  - verify result differs from complete-case (naive) objective
  - DIAGNOSTIC: print both values and their difference
```

### 4.3 `test_imputation.py`

```
Test: test_imputation_model_shapes
  - train a simple imputation model m_theta on complete cases
  - check output shape matches (n, 1)
  - DIAGNOSTIC: print train MSE of imputation model

Test: test_imputed_residual_unbiasedness
  - generate data where true m_theta is known analytically (linear case)
  - compare E[r_tilde] to E[r] on a large sample
  - assert |E[r_tilde] - E[r]| < 0.05
  - DIAGNOSTIC: print E[r_tilde], E[r], bias

Test: test_cross_fit_no_data_leakage
  - verify that for fold k, imputation model was trained only on I_{-k}
  - check by recording which indices were used in each fold's training
  - DIAGNOSTIC: print fold sizes
```

### 4.4 `test_pci_method.py`

```
Test: test_oracle_method_convergence
  - fit oracle DeepGMM on n=3000 demand data, 500 iterations (quick smoke test)
  - check that ATE estimate is finite
  - DIAGNOSTIC: print estimated ATE vs true ATE, absolute bias

Test: test_modified_method_less_biased_than_naive
  - fit both modified and naive methods, n=3000, missing_rate=0.3
  - |bias_modified| should be < |bias_naive|  (not guaranteed in one run, but typically holds)
  - DIAGNOSTIC: print bias_modified, bias_naive, ATE_true

Test: test_ate_estimator_shape
  - after fit, beta_hat(a=20.0) should be a scalar
  - delta_hat should be scalar
  - DIAGNOSTIC: print beta_hat(10), beta_hat(30), delta_hat

Test: test_no_nan_in_training
  - run 100 iterations, check that loss values are never NaN
  - DIAGNOSTIC: print loss curve every 10 iterations
```

---

## 5. Step 4 — Game Objective: `game_objectives/pci_moment_objective.py`

**Paper reference**: Section 4.2, equations for $U_{\bar\theta,n}(\theta,\tau)$ with
$\widetilde{r}_i(\theta)$ and $\widetilde{s}_i(\bar\theta)$.

```python
class PCIOptimalMomentObjective:
    """
    MAR-adapted DeepGMM game objective.

    U(θ, τ) = (1/n) Σ f_τ(L_i) * r̃_i(θ)
            - (1/4n) Σ f_τ(L_i)² * s̃_i(θ̄)

    where:
        L_i       = (A_i, Z_i)               [always observed]
        r̃_i(θ)   = δ_i * r_i(θ) + (1-δ_i) * m̂_θ(L+_i)   [imputed residual]
        s̃_i(θ̄)   = δ_i * r_i(θ̄)² + (1-δ_i) * v̂_θ̄(L+_i) [imputed sq. residual]
        r_i(θ)    = Y_i - h_θ(W_i, A_i)      [only computable when δ_i=1]
        m̂_θ(L+)  = E[r(θ) | L+, δ=1]        [fitted imputation model]
        v̂_θ̄(L+)  = E[r(θ̄)² | L+, δ=1]       [fitted variance imputation]
    """
    def calc_objective(self, h, f, treatment, treatment_proxy,
                       outcome_proxy, outcome, delta_w,
                       m_theta, v_theta_bar) -> Tuple[Tensor, Tensor]:
        # Returns (h_loss, f_loss) where h_loss is minimised and f_loss is minimised
        # (f_loss = -moment + f_reg, same sign convention as original codebase)
        ...
```

**Key implementation notes**:
- `h` takes `(outcome_proxy, treatment)` as input (PCI bridge function)
- `f` takes `(treatment, treatment_proxy)` = `L` as input (test function)
- When `delta_w[i]=0`, `h(W_i, A_i)` must NOT be called; use `m_theta` output instead
- Frozen `theta_bar` handled by passing pre-computed `v_theta_bar` tensor (EMA updated externally)

---

## 6. Step 5 — Learning Loop: `learning/learning_pci.py`

Extends `SGDLearningDevF` with:

1. **Cross-fit imputation training** (before main training loop):
   - Split data into K folds
   - For each fold k: train `m_theta_model` and `v_theta_bar_model` on `I_{-k}` complete cases
   - Store fold assignments for use during forward pass

2. **EMA update for theta_bar** (inside training loop):
   ```python
   theta_bar_state = ema_alpha * theta_bar_state + (1 - ema_alpha) * h.state_dict()
   ```
   But gradient does NOT flow through `theta_bar` (detach).

3. **Modified `update_params_iter`**:
   ```python
   def update_params_iter(self, iteration, treatment, treatment_proxy,
                          outcome_proxy, outcome, delta_w):
       # compute imputed residuals using current fold's m_theta, v_theta_bar
       # call pci_objective.calc_objective(...)
       # update h and f with one SGD step each
   ```

4. **ATE estimation after training** (Section 4 of paper, eq. for β̂):
   ```python
   def estimate_ate(self, h_hat, data, a_values, fold_indices, q_model):
       # For each i in fold k:
       #   if delta_w[i]=1: use h_hat(W_i, a, X_i) directly
       #   else:            use q̂_{a,θ̂}^{(-k)}(L+_i)
       # Average over all i
   ```

---

## 7. Step 6 — Top-Level Method: `methods/pci_deepgmm_method.py`

```python
class PCIDeepGMMMethod:
    """
    Mirrors ToyModelSelectionMethod interface.
    Supports three modes: 'oracle', 'modified', 'naive'
    """
    def __init__(self, mode='modified', n_folds=5, missing_rate=0.3,
                 enable_cuda=False): ...

    def fit(self, train_data: PVTrainDataSetMARTorch,
            dev_data: PVTrainDataSetMARTorch,
            verbose=False): ...

    def predict_ate(self, a1: float, a0: float) -> float:
        """Returns ATE estimate: β̂(a1) - β̂(a0)"""
        ...
```

**Mode behaviour**:
- `oracle`: uses all W (delta_w all ones), standard DeepGMM objective (no imputation)
- `modified`: MAR-imputed residuals, cross-fit imputation, full ATE estimator
- `naive`: sets delta_w to observed subset only, discards missing rows entirely (complete-case)

---

## 8. Step 7 — Comparison Script: `run_pci_compare.py`

Mirrors `/Users/apple/DeepFeatureProxyVariable/scripts/compare_dfpv_variants.py`.

```
usage: python run_pci_compare.py [--n-rep 50] [--missing-rate 0.3]
                                  [--n-train 2000] [--dump-root dumps/]
```

### Workflow

```
1. For each repetition (seed):
   a. Generate demand data (oracle / MAR variants)
   b. Fit Oracle DeepGMM     → ATE_oracle_hat
   c. Fit Modified DeepGMM  → ATE_modified_hat
   d. Fit Naive DeepGMM     → ATE_naive_hat
   e. Compute bias = ATE_hat - ATE_true for each

2. Aggregate across repetitions:
   - Report: mean bias, std bias, RMSE for each method

3. Save outputs:
   dumps/
   └── compare_{timestamp}/
       ├── results.csv              (rep, method, ate_hat, bias)
       ├── summary.csv              (method, mean_bias, std_bias, rmse)
       └── bias_distribution.png   (KDE plot, same style as DFPV project)
```

### Spartan HPC compatibility

- No `matplotlib` display (use `Agg` backend): `matplotlib.use('Agg')`
- Argument `--num-cpus` for parallel repetitions via `joblib`
- Argument `--no-cuda` fallback for CPU-only nodes
- All paths relative to script location (no hardcoded `/Users/apple/`)
- Add `slurm_job.sh` to submit to Spartan:

```bash
#!/bin/bash
#SBATCH --job-name=pci_deepgmm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/%j.out

module load python/3.10
source ~/venv/bin/activate
cd /home/wzzho2/DeepGMM
python run_pci_compare.py --n-rep 100 --missing-rate 0.3 --num-cpus 8
```

---

## 9. Implementation Order & Dependencies

```
[1] data/data_class_mar.py          (no dependencies, pure data structures)
      ↓
[2] scenarios/demand_scenario.py    (depends on data_class_mar)
      ↓
[3] tests/test_demand_dgp.py        (write + run before proceeding)
      ↓
[4] game_objectives/pci_moment_objective.py  (depends on nothing, pure torch)
      ↓
[5] tests/test_pci_objective.py     (write + run)
      ↓
[6] learning/learning_pci.py        (depends on pci_moment_objective)
      ↓
[7] tests/test_imputation.py        (write + run)
      ↓
[8] methods/pci_deepgmm_method.py   (depends on all above)
      ↓
[9] tests/test_pci_method.py        (write + run, end-to-end)
      ↓
[10] run_pci_compare.py             (depends on method + scenario)
      ↓
[11] slurm_job.sh                   (Spartan submission script)
```

---

## 10. Key Design Decisions

### Why cross-fitting is mandatory
Without cross-fitting, `m_theta` is trained on the same data used in the game objective,
causing overfitting of the imputation model and potentially biasing the bridge function estimate.
Cross-fitting ensures that for sample $i$ in fold $k$, the imputation model was trained only
on $I_{-k}$, breaking this feedback loop.

### How theta_bar is handled
In the original DeepGMM code, `epsilon_dev_tilde` (the frozen residual for variance estimation)
is computed from the average of past checkpoints. In the PCI version, we use EMA:
```python
theta_bar ← (1-alpha)*theta_bar + alpha*theta   (alpha=0.05 default)
```
`v_theta_bar` is re-computed from `theta_bar` at each evaluation step,
but gradient is stopped (`detach()`) when computing `s̃_i`.

### Naive baseline implementation
The naive (complete-case) method is implemented by:
1. Filtering the dataset to only rows where `delta_w=1`
2. Running standard DeepGMM on this filtered dataset
3. Averaging `h_hat(W_i, a, X_i)` only over the filtered test rows

This converges to `E[h(W,a,X) | delta_w=1]` ≠ `E[h(W,a,X)]` under MAR, giving biased ATE.

---

## 11. Success Criteria

| Check | Criterion |
|-------|-----------|
| DGP test passes | All 4 DGP tests green, missing rate within ±5% of target |
| Objective test passes | No NaN in objective, imputed=oracle when delta_w=all_ones |
| Imputation test passes | E[r_tilde] - E[r] < 0.05 on linear toy case |
| Method smoke test | ATE estimate finite, no NaN in training |
| Comparison result | |bias_modified| < |bias_naive| on average over ≥20 repetitions |
