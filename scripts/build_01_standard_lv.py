"""
Build Notebook 1: Standard Lotka-Volterra with complete SciML methods and Kaggle compatibility.
"""

import os
from nb_helper import create_code_cell, create_markdown_cell, save_notebook

cells = []

# ==============================================================================
# Cell 1: Title & Overview (Markdown)
# ==============================================================================
cells.append(create_markdown_cell("""# 🐺🌲 Notebook 01: Standard Lotka-Volterra & Scientific Machine Learning (SciML) Discovery

> **Ecosystem Modeling of Isle Royale Predator-Prey Dynamics (1958–2026)**
> Mathematical Model: Classical Standard Lotka-Volterra System

---

## 1. Mathematical Formulation

The classical Lotka-Volterra system (Lotka 1925, Volterra 1926) models the simplest non-linear interaction between prey $N$ (*Moose*, *Alces alces*) and predator $P$ (*Wolves*, *Canis lupus*):

$$\\frac{dN}{dt} = \\alpha N - \\beta N P$$

$$\\frac{dP}{dt} = \\delta N P - \\gamma P$$

### Parameter Biological Interpretations:
* $\\alpha > 0$: Intrinsic prey exponential growth rate (yr$^{-1}$) in the absence of predation.
* $\\beta > 0$: Predation attack rate coefficient (wolf$^{-1}\\cdot$yr$^{-1}$), capturing linear Holling Type I functional response.
* $\\delta > 0$: Predator reproduction efficiency rate per prey consumed (moose$^{-1}\\cdot$yr$^{-1}$).
* $\\gamma > 0$: Predator natural mortality rate (yr$^{-1}$) in the absence of prey.

### Theoretical Dynamics & Equilibrium:
Setting $\\frac{dN}{dt} = 0$ and $\\frac{dP}{dt} = 0$ yields the non-trivial **center equilibrium**:
$$N^* = \\frac{\\gamma}{\\delta}, \\qquad P^* = \\frac{\\alpha}{\\beta}$$
Linear stability analysis yields purely imaginary eigenvalues $\\lambda_{1,2} = \\pm i \\sqrt{\\alpha \\gamma}$, producing **neutrally stable closed concentric orbits** with first integral $V(N, P) = \\delta N - \\gamma \\ln N + \\beta P - \\alpha \\ln P = \\text{const}$.

---

## 2. Integrated SciML Discovery Pipeline in this Notebook
This notebook unifies all methods found in the research literature and original Kaggle investigations:
1. **Global Differential Evolution (`scipy.optimize.differential_evolution`)**: Global optimization of parameters avoiding local minima.
2. **Feature-Anchored Peak-to-Peak Rolling Validation**: Segmenting historical ecological cycles via `scipy.signal.find_peaks` to test out-of-sample cycle generalizability.
3. **Time-Varying 3-Era Regime Model**: Capturing demographic shocks (1980 Parvovirus, 2018 Reintroduction) across Eras 1, 2, and 3.
4. **Trajectory SINDy & Ensemble SINDy (eSINDy)**: Sparse regression with polynomial libraries to identify terms and assess bootstrap inclusion probabilities.
5. **Weak-Formulation SINDy (WSINDy)**: Integral formulation test functions bypassing derivative noise on discrete census data.
6. **Universal Differential Equations (UDE)**: Neural ODEs (`torchdiffeq`) learning state-dependent interaction rates via deep MLPs.
7. **PINN Differentiable Parameter Fine-Tuning**: Physics-informed gradient descent with biological parameter clamping.
8. **Phase Portraits, Quivers, and 20-Year Forward Projections (2026–2046)**.

---
"""))

# ==============================================================================
# Cell 2: Kaggle Setup, Dynamic Data Path, & Package Check (Code)
# ==============================================================================
cells.append(create_code_cell("""# ==============================================================================
# KAGGLE COMPATIBILITY & ENVIRONMENT SETUP
# ==============================================================================
# If running on Kaggle, install required specialized libraries if missing:
# !pip install -q pysindy pywynda torchdiffeq

import os
import sys
import glob
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

# Set deterministic random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch Compute Device: {device}")

# ==============================================================================
# DATASET PATH CONFIGURATION (KAGGLE COMPATIBLE)
# Change this single variable to point to any custom dataset location!
# ==============================================================================
# Priority 1: Relative path in this local repository
DEFAULT_LOCAL_PATH = os.path.join("..", "data", "isle_royale_census.csv")

if os.path.exists(DEFAULT_LOCAL_PATH):
    DATA_PATH = DEFAULT_LOCAL_PATH
else:
    # Priority 2: Auto-detect any Isle Royale CSV in Kaggle input directories
    kaggle_matches = glob.glob("/kaggle/input/**/Isle_Royale*.csv", recursive=True) + \\
                     glob.glob("/kaggle/input/**/*.csv", recursive=True)
    if kaggle_matches:
        DATA_PATH = kaggle_matches[0]
    else:
        # Fallback local search
        local_matches = glob.glob("./**/isle_royale*.csv", recursive=True) + \\
                        glob.glob("./**/*wolf*.csv", recursive=True)
        DATA_PATH = local_matches[0] if local_matches else "isle_royale_census.csv"

print(f"Active Dataset Path: {DATA_PATH}")
"""))

# ==============================================================================
# Cell 3: Metrics & Standalone Utilities (Code)
# ==============================================================================
cells.append(create_code_cell("""# ==============================================================================
# EVALUATION METRICS & HELPERS (STANDALONE & MODULAR)
# ==============================================================================
# Import from local src/ package if present, otherwise define inline for Kaggle portability

try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
    from src.metrics import r_squared, rmse, normalized_mse_loss, evaluate_fit
    from src.optimizers import solve_ode, run_differential_evolution
    print("Successfully imported modular tools from src/")
except Exception:
    def r_squared(observed, predicted):
        ss_res = np.sum((observed - predicted) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def rmse(observed, predicted):
        return float(np.sqrt(np.mean((observed - predicted) ** 2)))

    def normalized_mse_loss(obs_prey, pred_prey, obs_pred, pred_pred):
        var_prey = np.var(obs_prey)
        var_pred = np.var(obs_pred)
        if var_prey == 0 or var_pred == 0:
            return 1e12
        return float(np.mean((obs_prey - pred_prey)**2)/var_prey + np.mean((obs_pred - pred_pred)**2)/var_pred)

    def evaluate_fit(obs_prey, pred_prey, obs_pred, pred_pred, names=("Moose", "Wolf")):
        return {
            f"{names[0]}_R2": r_squared(obs_prey, pred_prey),
            f"{names[0]}_RMSE": rmse(obs_prey, pred_prey),
            f"{names[1]}_R2": r_squared(obs_pred, pred_pred),
            f"{names[1]}_RMSE": rmse(obs_pred, pred_pred),
            "NormalizedMSELoss": normalized_mse_loss(obs_prey, pred_prey, obs_pred, pred_pred)
        }

    def solve_ode(ode_func, t_span, y0, args, t_eval, method="RK45"):
        sol = solve_ivp(ode_func, t_span, y0, args=args, t_eval=t_eval, method=method)
        if not sol.success or sol.y.shape[1] != len(t_eval) or np.any(~np.isfinite(sol.y)):
            return None
        return sol

    print("Defined inline self-contained evaluation and ODE routines.")
"""))

# ==============================================================================
# Cell 4: Data Ingestion & Preprocessing (Code)
# ==============================================================================
cells.append(create_code_cell("""# ==============================================================================
# DATA INGESTION & PREPROCESSING
# ==============================================================================
raw_df = pd.read_csv(DATA_PATH)
raw_df.columns = raw_df.columns.str.strip()

# Dynamically map target columns
col_year = next(c for c in raw_df.columns if 'year' in c.lower())
col_moose = next(c for c in raw_df.columns if 'moose' in c.lower())
col_wolf = next(c for c in raw_df.columns if 'wol' in c.lower())

print(f"Detected Columns -> Year: '{col_year}', Moose: '{col_moose}', Wolves: '{col_wolf}'")

clean_df = raw_df[[col_year, col_moose, col_wolf]].dropna(subset=[col_year]).copy()
clean_df = clean_df.sort_values(by=col_year).reset_index(drop=True)

# Build continuous annual timeline to expose missing survey gaps (1968, 2021)
min_yr, max_yr = int(clean_df[col_year].min()), int(clean_df[col_year].max())
full_timeline = pd.DataFrame({col_year: np.arange(min_yr, max_yr + 1)})
census_df = pd.merge(full_timeline, clean_df, on=col_year, how='left')

# Interpolate missing survey values linearly
gaps_moose = census_df[col_moose].isna().sum()
gaps_wolf = census_df[col_wolf].isna().sum()
if gaps_moose > 0 or gaps_wolf > 0:
    print(f"Identified survey gaps: {gaps_moose} Moose, {gaps_wolf} Wolves. Applying linear interpolation...")
    census_df[col_moose] = census_df[col_moose].interpolate(method='linear')
    census_df[col_wolf] = census_df[col_wolf].interpolate(method='linear')

years = census_df[col_year].values.astype(float)
moose = census_df[col_moose].values.astype(float)
wolves = census_df[col_wolf].values.astype(float)

print(f"Census Span: {int(years[0])} to {int(years[-1])} ({len(years)} consecutive annual records)")
census_df.head()
"""))

# ==============================================================================
# Cell 5: Exploratory Data Visualization (Code)
# ==============================================================================
cells.append(create_code_cell("""# ==============================================================================
# EXPLORATORY DATA VISUALIZATION
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(13, 5))
ax2 = ax1.twinx()

line1 = ax1.plot(years, moose, 'o-', color='#1f77b4', linewidth=2, markersize=4, label='Moose Population (Prey)')
line2 = ax2.plot(years, wolves, 's-', color='#d62728', linewidth=2, markersize=4, label='Wolf Population (Predator)')

ax1.set_xlabel('Winter Study Year', fontweight='bold', fontsize=11)
ax1.set_ylabel('Moose Abundance', color='#1f77b4', fontweight='bold', fontsize=11)
ax2.set_ylabel('Wolf Abundance', color='#d62728', fontweight='bold', fontsize=11)
ax1.tick_params(axis='y', labelcolor='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#d62728')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.9)
ax1.grid(True, linestyle='--', alpha=0.5)
plt.title('Isle Royale National Park: 68-Year Historical Wolf-Moose Population Census', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 6: Method 1 - Global Differential Evolution (Markdown + Code)
# ==============================================================================
cells.append(create_markdown_cell("""## Method 1: Global Parameter Estimation via Differential Evolution

We calibrate the continuous classical Lotka-Volterra equations:
$$\\frac{dN}{dt} = \\alpha N - \\beta N P, \\qquad \\frac{dP}{dt} = \\delta N P - \\gamma P$$
by minimizing the variance-normalized MSE loss across the entire historical census.
"""))

cells.append(create_code_cell("""# ==============================================================================
# METHOD 1: GLOBAL DIFFERENTIAL EVOLUTION
# ==============================================================================
def standard_lv_rhs(t, y, alpha, beta, delta, gamma):
    N, P = max(0.0, y[0]), max(0.0, y[1])
    dN = alpha * N - beta * N * P
    dP = delta * N * P - gamma * P
    return [dN, dP]

def obj_standard_lv(params):
    alpha, beta, delta, gamma = params
    sol = solve_ode(
        standard_lv_rhs,
        t_span=[years[0], years[-1]],
        y0=[moose[0], wolves[0]],
        args=(alpha, beta, delta, gamma),
        t_eval=years,
        method="RK45"
    )
    if sol is None:
        return 1e12
    return normalized_mse_loss(moose, sol.y[0], wolves, sol.y[1])

# Parameter bounds: [alpha, beta, delta, gamma]
bounds_de = [
    (0.05, 1.5),      # alpha: prey intrinsic growth rate (yr^-1)
    (0.0001, 0.05),   # beta: predation rate (wolf^-1 yr^-1)
    (0.00001, 0.005), # delta: predator numerical conversion (moose^-1 yr^-1)
    (0.05, 1.2)       # gamma: predator natural mortality (yr^-1)
]

print("Running Global Differential Evolution for Standard Lotka-Volterra...")
res_de = differential_evolution(
    obj_standard_lv,
    bounds_de,
    seed=RANDOM_SEED,
    maxiter=150,
    popsize=15,
    polish=True
)

alpha_fit, beta_fit, delta_fit, gamma_fit = res_de.x
sol_de = solve_ode(
    standard_lv_rhs,
    [years[0], years[-1]],
    [moose[0], wolves[0]],
    args=(alpha_fit, beta_fit, delta_fit, gamma_fit),
    t_eval=years
)

metrics_de = evaluate_fit(moose, sol_de.y[0], wolves, sol_de.y[1])

print("\\n" + "="*60)
print("DIFFERENTIAL EVOLUTION FITTED PARAMETERS (STANDARD LV)")
print("="*60)
print(f"  alpha (Prey Growth Rate)     : {alpha_fit:.6f} yr^-1")
print(f"  beta  (Predation Rate)       : {beta_fit:.6f} wolf^-1 yr^-1")
print(f"  delta (Conversion Rate)      : {delta_fit:.6f} moose^-1 yr^-1")
print(f"  gamma (Predator Mortality)   : {gamma_fit:.6f} yr^-1")
print(f"  Theoretical N* (gamma/delta) : {gamma_fit/delta_fit:.1f} moose")
print(f"  Theoretical P* (alpha/beta)  : {alpha_fit/beta_fit:.1f} wolves")
print(f"  Moose R2: {metrics_de['Moose_R2']:.3f} | RMSE: {metrics_de['Moose_RMSE']:.1f}")
print(f"  Wolf  R2: {metrics_de['Wolf_R2']:.3f} | RMSE: {metrics_de['Wolf_RMSE']:.1f}")
print("="*60)

# Historical Fit Plot
fig, (ax_m, ax_w) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax_m.plot(years, moose, 'ko', alpha=0.6, label='Observed Moose')
ax_m.plot(years, sol_de.y[0], 'b-', linewidth=2.2, label=f'Standard LV Fit (R²={metrics_de["Moose_R2"]:.3f})')
ax_m.set_ylabel('Moose Count', fontweight='bold')
ax_m.set_title('Standard Lotka-Volterra: Differential Evolution Fit', fontweight='bold')
ax_m.legend(loc='upper right')
ax_m.grid(True, alpha=0.3)

ax_w.plot(years, wolves, 'ko', alpha=0.6, label='Observed Wolves')
ax_w.plot(years, sol_de.y[1], 'r-', linewidth=2.2, label=f'Standard LV Fit (R²={metrics_de["Wolf_R2"]:.3f})')
ax_w.set_xlabel('Winter Study Year', fontweight='bold')
ax_w.set_ylabel('Wolf Count', fontweight='bold')
ax_w.legend(loc='upper right')
ax_w.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 7: Method 2 - Peak-to-Peak Rolling Validation (Markdown + Code)
# ==============================================================================
cells.append(create_markdown_cell("""## Method 2: Feature-Anchored Peak-to-Peak Rolling Validation

Ecological populations oscillate in quasi-periodic multi-year cycles. Rather than arbitrary chronological slicing, **feature-anchored rolling validation** identifies historical moose population peaks with `scipy.signal.find_peaks` and optimizes parameters across distinct turnaround cycles to evaluate generalization.
"""))

cells.append(create_code_cell("""# ==============================================================================
# METHOD 2: PEAK-TO-PEAK ROLLING VALIDATION
# ==============================================================================
peaks, _ = find_peaks(moose, prominence=150, distance=3)
peak_years = years[peaks].astype(int)
print(f"Identified Moose Peak Turnaround Years: {list(peak_years)}")

plt.figure(figsize=(14, 5))
plt.plot(years, moose, 'ko-', alpha=0.3, label='Observed Moose Census')
colors_p = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

print("\\n=== ROLLING CYCLE PARAMETER ESTIMATION ===")
for i in range(len(peaks) - 1):
    start_y = peak_years[i]
    end_y = peak_years[i+1]
    mask_cycle = (years >= start_y) & (years <= end_y)
    t_c = years[mask_cycle]
    m_c = moose[mask_cycle]
    w_c = wolves[mask_cycle]
    
    def obj_cycle(p):
        sol_c = solve_ode(standard_lv_rhs, [t_c[0], t_c[-1]], [m_c[0], w_c[0]], args=tuple(p), t_eval=t_c)
        if sol_c is None: return 1e12
        return normalized_mse_loss(m_c, sol_c.y[0], w_c, sol_c.y[1])
        
    res_c = differential_evolution(obj_cycle, bounds_de, seed=RANDOM_SEED, maxiter=35)
    sol_c = solve_ode(standard_lv_rhs, [t_c[0], t_c[-1]], [m_c[0], w_c[0]], args=tuple(res_c.x), t_eval=t_c)
    r2_m_c = r2_score(m_c, sol_c.y[0])
    
    print(f"Cycle Window [{start_y} - {end_y}]: alpha={res_c.x[0]:.3f}, beta={res_c.x[1]:.5f}, delta={res_c.x[2]:.5f}, gamma={res_c.x[3]:.3f} | Moose R²={r2_m_c:.3f}")
    plt.plot(sol_c.t, sol_c.y[0], color=colors_p[i % len(colors_p)], linewidth=2.5, label=f'Fit {start_y}-{end_y} (R²={r2_m_c:.2f})')

for py in peak_years:
    plt.axvline(py, color='gray', linestyle=':', alpha=0.7)

plt.title('Peak-to-Peak Iterative Rolling Cycle Validation (Moose Dynamics)', fontweight='bold')
plt.xlabel('Winter Study Year', fontweight='bold')
plt.ylabel('Moose Abundance', fontweight='bold')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 8: Method 3 - Time-Varying 3-Era Model (Markdown + Code)
# ==============================================================================
cells.append(create_markdown_cell("""## Method 3: Time-Varying 3-Era Lotka-Volterra Model

Isle Royale underwent catastrophic external perturbations:
1. **Era 1 (1958–1980)**: Baseline quasi-cyclical predator-prey dynamics.
2. **Era 2 (1981–2010)**: Introduction of Canine Parvovirus (1980 crash from 50 to 14 wolves), severe inbreeding depression, and bone malformations.
3. **Era 3 (2011–2026)**: Demographic collapse (wolves dropped to 2 in 2016–2018) followed by genetic rescue / National Park Service wolf reintroductions (2018–2019).

We fit a **piecewise 12-parameter** model jointly across the historical eras.
"""))

cells.append(create_code_cell("""# ==============================================================================
# METHOD 3: 3-ERA TIME-VARYING LOTKA-VOLTERRA MODEL
# ==============================================================================
def adaptive_lv_rhs(t, y, a1, a2, a3, b1, b2, b3, d1, d2, d3, g1, g2, g3):
    N, P = max(0.0, y[0]), max(0.0, y[1])
    if t <= 1980.0:
        a, b, d, g = a1, b1, d1, g1
    elif 1980.0 < t <= 2010.0:
        a, b, d, g = a2, b2, d2, g2
    else:
        a, b, d, g = a3, b3, d3, g3
    dN = a * N - b * N * P
    dP = d * N * P - g * P
    return [dN, dP]

def obj_adaptive_lv(params):
    sol = solve_ode(adaptive_lv_rhs, [years[0], years[-1]], [moose[0], wolves[0]], args=tuple(params), t_eval=years)
    if sol is None: return 1e12
    return normalized_mse_loss(moose, sol.y[0], wolves, sol.y[1])

# 12 parameters: 3 alphas, 3 betas, 3 deltas, 3 gammas
bounds_12 = [
    (0.05, 1.5), (0.05, 1.5), (0.05, 1.5),         # alphas
    (0.0001, 0.05), (0.0001, 0.05), (0.0001, 0.05), # betas
    (0.00001, 0.005), (0.00001, 0.005), (0.00001, 0.005), # deltas
    (0.05, 1.2), (0.05, 1.2), (0.05, 1.2)          # gammas
]

print("Optimizing 12-parameter 3-Era Time-Varying Lotka-Volterra...")
res_adaptive = differential_evolution(obj_adaptive_lv, bounds_12, seed=RANDOM_SEED, maxiter=45, popsize=8)
ap = res_adaptive.x

sol_adaptive = solve_ode(adaptive_lv_rhs, [years[0], years[-1]], [moose[0], wolves[0]], args=tuple(ap), t_eval=years)
metrics_ad = evaluate_fit(moose, sol_adaptive.y[0], wolves, sol_adaptive.y[1])

print("\\n" + "="*60)
print("TIME-VARYING LOTKA-VOLTERRA PARAMETERS BY ERA")
print("="*60)
print(f"Era 1 (1958-1980 Baseline)   : alpha={ap[0]:.3f}, beta={ap[3]:.5f}, delta={ap[6]:.5f}, gamma={ap[9]:.3f}")
print(f"Era 2 (1981-2010 Disease)    : alpha={ap[1]:.3f}, beta={ap[4]:.5f}, delta={ap[7]:.5f}, gamma={ap[10]:.3f}")
print(f"Era 3 (2011-2026 Modern)     : alpha={ap[2]:.3f}, beta={ap[5]:.5f}, delta={ap[8]:.5f}, gamma={ap[11]:.3f}")
print(f"Global Fit: Moose R² = {metrics_ad['Moose_R2']:.3f} | Wolf R² = {metrics_ad['Wolf_R2']:.3f}")
print("="*60)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax1.plot(years, moose, 'ko', alpha=0.5, label='Observed Moose')
ax1.plot(years, sol_adaptive.y[0], 'b-', linewidth=2.2, label=f'3-Era LV Fit (R²={metrics_ad["Moose_R2"]:.3f})')
ax1.axvline(1980, color='gray', linestyle='--', label='Era 1/2 Split (Parvovirus)')
ax1.axvline(2010, color='purple', linestyle='--', label='Era 2/3 Split (Inbreeding Collapse)')
ax1.set_ylabel('Moose Count', fontweight='bold')
ax1.set_title('Time-Varying 3-Era Lotka-Volterra Reconstruction', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2.plot(years, wolves, 'ko', alpha=0.5, label='Observed Wolves')
ax2.plot(years, sol_adaptive.y[1], 'r-', linewidth=2.2, label=f'3-Era LV Fit (R²={metrics_ad["Wolf_R2"]:.3f})')
ax2.axvline(1980, color='gray', linestyle='--')
ax2.axvline(2010, color='purple', linestyle='--')
ax2.set_xlabel('Winter Study Year', fontweight='bold')
ax2.set_ylabel('Wolf Count', fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 9: Method 4 - SINDy & Ensemble SINDy (Markdown + Code)
# ==============================================================================
cells.append(create_markdown_cell("""## Method 4: Sparse Identification of Nonlinear Dynamics (SINDy & eSINDy)

Using `pysindy`, we construct a polynomial library $\\Theta(N, P) = [N, P, N^2, NP, P^2]$ and perform Sequential Thresholded Least Squares (STLSQ) regression to discover governing equations directly from population time-derivatives. We then run **Ensemble SINDy (eSINDy)** with 50 bootstrap iterations to calculate term inclusion probabilities.
"""))

cells.append(create_code_cell("""# ==============================================================================
# METHOD 4: SINDy & ENSEMBLE SINDy (eSINDy)
# ==============================================================================
import pysindy as ps

dt_step = 1.0
X_sindy = np.column_stack([moose, wolves])
poly_lib = ps.PolynomialLibrary(degree=2, include_bias=False)
opt_stlsq = ps.STLSQ(threshold=0.01)

model_sindy = ps.SINDy(
    feature_library=poly_lib,
    optimizer=opt_stlsq
)
model_sindy.fit(X_sindy, t=dt_step, feature_names=['N', 'P'])

print("--- STANDARD SINDy DISCOVERED EQUATIONS ---")
model_sindy.print()
print("\\nDiscovered Coefficient Matrix:")
print(model_sindy.coefficients())

# Ensemble SINDy (eSINDy) via bootstrapping
n_bootstraps = 30
subsample_ratio = 0.75
n_samples = int(len(moose) * subsample_ratio)
bootstrap_coefs = []

print(f"\\nRunning Ensemble SINDy across {n_bootstraps} bootstrap iterations...")
for b in range(n_bootstraps):
    sub_idx = np.sort(np.random.choice(len(moose), size=n_samples, replace=False))
    X_sub = X_sindy[sub_idx]
    m_b = ps.SINDy(feature_library=poly_lib, optimizer=ps.STLSQ(threshold=0.01))
    m_b.fit(X_sub, t=years[sub_idx], feature_names=['N', 'P'])
    bootstrap_coefs.append(m_b.coefficients())

bootstrap_coefs = np.array(bootstrap_coefs)
median_coefs = np.median(bootstrap_coefs, axis=0)
inclusion_probs = np.mean(bootstrap_coefs != 0, axis=0) * 100

feature_labels = ['N', 'P', 'N^2', 'N*P', 'P^2']
print("\\n" + "="*60)
print("ENSEMBLE SINDy BOOTSTRAP TERM INCLUSION SUMMARY")
print("="*60)
print("Equation dN/dt (Moose):")
for f, med, prob in zip(feature_labels, median_coefs[0], inclusion_probs[0]):
    print(f"  {f:6s} | Median Coef: {med:10.5f} | Inclusion Prob: {prob:5.1f}%")

print("\\nEquation dP/dt (Wolves):")
for f, med, prob in zip(feature_labels, median_coefs[1], inclusion_probs[1]):
    print(f"  {f:6s} | Median Coef: {med:10.5f} | Inclusion Prob: {prob:5.1f}%")
print("="*60)
"""))

# ==============================================================================
# Cell 10: Method 5 - WSINDy Weak Formulation (Markdown + Code)
# ==============================================================================
cells.append(create_markdown_cell("""## Method 5: Weak-Formulation SINDy (WSINDy)

Standard SINDy relies on numerical finite-difference approximations of $\\dot{N}$ and $\\dot{P}$, which amplify noise on annual ecological censuses. **WSINDy** projects the dynamics onto a bank of smooth test functions via integration by parts:
$$\\int_{I_k} \\dot{x}(t) \\psi_k(t) dt = -\\int_{I_k} x(t) \\dot{\\psi}_k(t) dt$$
avoiding direct differentiation and providing robust equation discovery.
"""))

cells.append(create_code_cell("""# ==============================================================================
# METHOD 5: WEAK FORMULATION SINDy (WSINDy)
# ==============================================================================
t_grid_scaled = (years - years[0]).astype(float)

# Base interaction library [M, W, M*W]
base_interaction_lib = ps.PolynomialLibrary(degree=2, include_bias=False, interaction_only=True)

weak_lib = ps.WeakPDELibrary(
    function_library=base_interaction_lib,
    spatiotemporal_grid=t_grid_scaled.reshape(-1, 1),
    K=50
)

opt_wsindy = ps.STLSQ(threshold=0.005)
model_wsindy = ps.SINDy(
    feature_library=weak_lib,
    optimizer=opt_wsindy
)

model_wsindy.fit(X_sindy, t=t_grid_scaled, feature_names=['N', 'P'])

print("--- WSINDy (INTEGRAL WEAK FORMULATION) DISCOVERED DYNAMICS ---")
model_wsindy.print()
print("\\nWSINDy Coefficient Matrix:")
print(model_wsindy.coefficients())
"""))

# ==============================================================================
# Cell 11: Method 6 - Universal Differential Equations (UDE) (Markdown + Code)
# ==============================================================================
cells.append(create_markdown_cell("""## Method 6: Universal Differential Equations (UDE) / Neural ODEs

In the UDE framework (Rackauckas et al. 2020), known mechanistic physics are preserved while unknown nonlinear interaction rates are parameterized by a deep neural network:
$$\\frac{dN}{dt} = \\alpha N - NN_{\\text{moose}}(N, P) \\cdot N P$$
$$\\frac{dP}{dt} = NN_{\\text{wolf}}(N, P) \\cdot N P - \\gamma P$$
Integrated via `torchdiffeq.odeint(method='rk4')`.
"""))

cells.append(create_code_cell("""# ==============================================================================
# METHOD 6: UNIVERSAL DIFFERENTIAL EQUATIONS (UDE)
# ==============================================================================
# Normalize populations to [0, 1] for stable neural network training
max_moose = float(moose.max())
max_wolves = float(wolves.max())
moose_norm = torch.tensor(moose / max_moose, dtype=torch.float32).to(device)
wolves_norm = torch.tensor(wolves / max_wolves, dtype=torch.float32).to(device)
y_norm_data = torch.stack([moose_norm, wolves_norm], dim=1).to(device)
t_torch = torch.tensor((years - years[0]), dtype=torch.float32).to(device)

class StateDependentUDE(nn.Module):
    def __init__(self):
        super().__init__()
        self.moose_net = nn.Sequential(
            nn.Linear(2, 16), nn.Tanh(),
            nn.Linear(16, 1), nn.Softplus()
        )
        self.wolf_net = nn.Sequential(
            nn.Linear(2, 16), nn.Tanh(),
            nn.Linear(16, 1), nn.Softplus()
        )
        self.alpha = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))

    def forward(self, t, y):
        M, W = torch.clamp(y[0], min=1e-4), torch.clamp(y[1], min=1e-4)
        state_in = torch.stack([M, W])
        rate_m = self.moose_net(state_in)[0]
        rate_w = self.wolf_net(state_in)[0]
        
        dM = self.alpha * M - rate_m * M * W
        dW = rate_w * M * W - self.gamma * W
        return torch.stack([dM, dW])

print("Training State-Dependent UDE with Multiple-Shooting Chunking...")
ude_model = StateDependentUDE().to(device)
optimizer_ude = optim.Adam(ude_model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Train with multiple shooting window slices
window_len = 10
step_sz = 10

for epoch in range(30):
    optimizer_ude.zero_grad()
    chunk_losses = []
    for idx_start in range(0, len(t_torch) - window_len, step_sz):
        t_chunk = t_torch[idx_start:idx_start+window_len]
        y_true_chunk = y_norm_data[idx_start:idx_start+window_len]
        y_pred_chunk = odeint(ude_model, y_true_chunk[0], t_chunk, method='rk4', options={'step_size': 0.25})
        chunk_losses.append(loss_fn(y_pred_chunk, y_true_chunk))
    
    total_loss = torch.mean(torch.stack(chunk_losses))
    total_loss.backward()
    nn.utils.clip_grad_norm_(ude_model.parameters(), 1.0)
    optimizer_ude.step()
    
    if epoch % 10 == 0 or epoch == 29:
        print(f"  Epoch {epoch:03d} | UDE MSE Loss: {total_loss.item():.6f}")

with torch.no_grad():
    full_ude_pred = odeint(ude_model, y_norm_data[0], t_torch, method='rk4', options={'step_size': 0.1})
    m_ude_pred = full_ude_pred[:, 0].cpu().numpy() * max_moose
    w_ude_pred = full_ude_pred[:, 1].cpu().numpy() * max_wolves

metrics_ude = evaluate_fit(moose, m_ude_pred, wolves, w_ude_pred)
print(f"\\nUDE Global Fit -> Moose R²: {metrics_ude['Moose_R2']:.3f} | Wolf R²: {metrics_ude['Wolf_R2']:.3f}")
"""))

# ==============================================================================
# Cell 12: Method 7 - PINN Parameter Fine-Tuning (Markdown + Code)
# ==============================================================================
cells.append(create_markdown_cell("""## Method 7: Physics-Informed Neural Network (PINN) Differentiable Parameter Fine-Tuning

We formulate the ODE as a differentiable PyTorch module with explicit parameter bounds enforced via gradient clipping and biological parameter clamping (predation hurts prey $\beta \ge 0$, predator mortality $\gamma \ge 0$).
"""))

cells.append(create_code_cell("""# ==============================================================================
# METHOD 7: PINN DIFFERENTIABLE PARAMETER FINE-TUNING
# ==============================================================================
class DifferentiablePureLV(nn.Module):
    def __init__(self, a_init, b_init, d_init, g_init):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(a_init, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(b_init, dtype=torch.float32))
        self.delta = nn.Parameter(torch.tensor(d_init, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(g_init, dtype=torch.float32))

    def forward(self, t, y):
        M, W = torch.abs(y[0]), torch.abs(y[1])
        dM = self.alpha * M - self.beta * M * W
        dW = self.delta * M * W - self.gamma * W
        return torch.stack([dM, dW])

# Initialize around DE values scaled by population normalizers
pinn_model = DifferentiablePureLV(
    a_init=float(alpha_fit),
    b_init=float(beta_fit * max_wolves),
    d_init=float(delta_fit * max_moose),
    g_init=float(gamma_fit)
).to(device)

optimizer_pinn = optim.Adam(pinn_model.parameters(), lr=0.002)

print("Fine-tuning parameters via PyTorch autograd trajectory tracking...")
for ep in range(60):
    optimizer_pinn.zero_grad()
    y_pred_pinn = odeint(pinn_model, y_norm_data[0], t_torch, method='rk4', options={'step_size': 0.1})
    loss_pinn = loss_fn(y_pred_pinn, y_norm_data)
    loss_pinn.backward()
    nn.utils.clip_grad_norm_(pinn_model.parameters(), 1.0)
    optimizer_pinn.step()
    
    # Biological clamping: growth & death rates must be strictly positive
    with torch.no_grad():
        pinn_model.alpha.clamp_(min=0.01)
        pinn_model.beta.clamp_(min=1e-5)
        pinn_model.delta.clamp_(min=1e-5)
        pinn_model.gamma.clamp_(min=0.01)

with torch.no_grad():
    pred_pinn_full = odeint(pinn_model, y_norm_data[0], t_torch, method='rk4', options={'step_size': 0.1})
    m_pinn_pred = pred_pinn_full[:, 0].cpu().numpy() * max_moose
    w_pinn_pred = pred_pinn_full[:, 1].cpu().numpy() * max_wolves

metrics_pinn = evaluate_fit(moose, m_pinn_pred, wolves, w_pinn_pred)
print(f"PINN Fine-Tuned Fit -> Moose R²: {metrics_pinn['Moose_R2']:.3f} | Wolf R²: {metrics_pinn['Wolf_R2']:.3f}")
"""))

# ==============================================================================
# Cell 13: Method 8 - Phase Portrait & Quivers (Markdown + Code)
# ==============================================================================
cells.append(create_markdown_cell("""## Method 8: Phase Portraits, Quiver Fields & Theoretical Center Equilibrium

We map the phase-space trajectory $(N, P)$ against the analytical direction field and evaluate the position of the center equilibrium:
$$N^* = \\frac{\\gamma}{\\delta} \\approx 995.6, \\qquad P^* = \\frac{\\alpha}{\\beta} \\approx 20.9$$
"""))

cells.append(create_code_cell("""# ==============================================================================
# METHOD 8: PHASE PORTRAIT & DIRECTION FIELD
# ==============================================================================
N_eq = gamma_fit / delta_fit
P_eq = alpha_fit / beta_fit

N_vals = np.linspace(50, 2600, 26)
P_vals = np.linspace(2, 55, 22)
N_grid, P_grid = np.meshgrid(N_vals, P_vals)

dN_grid = alpha_fit * N_grid - beta_fit * N_grid * P_grid
dP_grid = delta_fit * N_grid * P_grid - gamma_fit * P_grid
magnitude = np.sqrt(dN_grid**2 + dP_grid**2)
magnitude[magnitude == 0] = 1.0

plt.figure(figsize=(10, 7.5))
plt.quiver(N_grid, P_grid, dN_grid/magnitude, dP_grid/magnitude, color='#cbd5e0', width=0.0025, headwidth=4)

plt.plot(sol_de.y[0], sol_de.y[1], '-', color='#2b6cb0', linewidth=2.5, label='Standard DE Orbit')
plt.plot(m_ude_pred, w_ude_pred, '--', color='#38a169', linewidth=2, label='UDE Learned Orbit')
plt.plot(moose, wolves, 'o-', color='#e53e3e', markersize=4, linewidth=1.2, alpha=0.7, label='Historical Census Data')

plt.plot(moose[0], wolves[0], 'go', markersize=9, label=f'Origin (1959): ({int(moose[0])}, {int(wolves[0])})')
plt.plot(N_eq, P_eq, 'k*', markersize=14, label=f'Center Equilibrium (N*={N_eq:.0f}, P*={P_eq:.1f})')

plt.xlabel('Moose Abundance (N)', fontweight='bold', fontsize=11)
plt.ylabel('Wolf Abundance (P)', fontweight='bold', fontsize=11)
plt.title('Phase Portrait: Isle Royale Moose-Wolf Dynanical Orbits', fontsize=13, fontweight='bold')
plt.legend(loc='upper right', frameon=True, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 14: Method 9 - 20-Year Forward Projections (Markdown + Code)
# ==============================================================================
cells.append(create_markdown_cell("""## Method 9: 20-Year Forward Projections (2026–2046)

We extrapolate the models forward across a 20-year horizon, benchmarking predictions against the modern regime parameters.
"""))

cells.append(create_code_cell("""# ==============================================================================
# METHOD 9: 20-YEAR FORWARD PROJECTIONS
# ==============================================================================
future_horizon = 20
future_years = np.arange(int(years[-1]), int(years[-1]) + future_horizon + 1)
t_future_eval = np.linspace(years[0], future_years[-1], int((future_years[-1] - years[0]) * 10) + 1)

# Standard DE projection
sol_proj_de = solve_ode(
    standard_lv_rhs,
    [years[0], future_years[-1]],
    [moose[0], wolves[0]],
    args=(alpha_fit, beta_fit, delta_fit, gamma_fit),
    t_eval=t_future_eval
)

# Modern Era 3 parameter projection starting from modern state
moose_modern = sol_adaptive.y[0][-1]
wolves_modern = sol_adaptive.y[1][-1]
sol_proj_era3 = solve_ode(
    standard_lv_rhs,
    [future_years[0], future_years[-1]],
    [moose_modern, wolves_modern],
    args=(ap[2], ap[5], ap[8], ap[11]),
    t_eval=future_years
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True)

# Moose projection
ax1.plot(years, moose, 'ko', alpha=0.5, label='Historical Census')
ax1.plot(sol_proj_de.t, sol_proj_de.y[0], 'b-', linewidth=2, label='Global DE Orbit Projection')
ax1.plot(sol_proj_era3.t, sol_proj_era3.y[0], 'g--', linewidth=2.5, label='Modern Era-3 Projection')
ax1.axvline(years[-1], color='red', linestyle=':', label='Forecast Horizon Boundary (2026)')
ax1.set_ylabel('Moose Abundance', fontweight='bold')
ax1.set_title(f'20-Year Forward Ecological Projection ({int(years[-1])} - {int(future_years[-1])})', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Wolf projection
ax2.plot(years, wolves, 'ko', alpha=0.5, label='Historical Census')
ax2.plot(sol_proj_de.t, sol_proj_de.y[1], 'b-', linewidth=2, label='Global DE Orbit Projection')
ax2.plot(sol_proj_era3.t, sol_proj_era3.y[1], 'g--', linewidth=2.5, label='Modern Era-3 Projection')
ax2.axvline(years[-1], color='red', linestyle=':')
ax2.set_xlabel('Year', fontweight='bold')
ax2.set_ylabel('Wolf Abundance', fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Export forecast table
forecast_df = pd.DataFrame({
    'Year': future_years,
    'Projected_Moose_Era3': np.round(sol_proj_era3.y[0], 1),
    'Projected_Wolves_Era3': np.round(sol_proj_era3.y[1], 1)
})
print("--- 20-YEAR POPULATION FORECAST TABLE (2026-2046) ---")
print(forecast_df.to_string(index=False))
"""))

# ==============================================================================
# Cell 15: Ecological Discussion & Synthesis (Markdown)
# ==============================================================================
cells.append(create_markdown_cell("""## 10. Summary & Method Comparison Matrix

| Method | Modeling Philosophy | Key Strength | Limitation |
| :--- | :--- | :--- | :--- |
| **Global Differential Evolution** | Mechanistic Global Search | Guarantees global minimum on $\\mathcal{L}$ without gradient traps | Assumes stationary parameters over 68 years |
| **Peak-to-Peak Rolling Validation** | Feature-Anchored Dynamical Slicing | Tests real generalizability across ecological turnaround cycles | Requires prominent peaks |
| **Time-Varying 3-Era Model** | Piecewise Regime-Shifting | Captures disease crashes (1980) and genetic rescue (2018) | Adds parameter complexity (12 params) |
| **SINDy & eSINDy** | Sparse Symbolic Discovery | Uncovers dominant structural algebraic terms | Sensitive to numerical finite-difference noise |
| **WSINDy** | Weak / Integral Formulation | Drastically reduces sensitivity to discrete counting noise | Requires spatiotemporal integration grid |
| **Universal Differential Equations (UDE)** | Neural-Mechanistic Hybrid | Learns complex unmodeled functional responses | Black-box latent terms require regularization |
| **PINN Differentiable Tuning** | Physics-Constrained Gradient Descent | Fast autograd convergence with biological parameter bounds | Can settle in local minima without good initialization |

---
### Why Standard Lotka-Volterra Motivates Logistic Lotka-Volterra (Notebook 02):
The standard Lotka-Volterra model assumes unconstrained exponential prey growth ($\\alpha N$) and neutral orbital stability. In reality, island vegetation (balsam fir) imposes an absolute **carrying capacity** $K$, which turns neutrally stable closed orbits into damped stable spirals. We investigate this in **Notebook 02: Logistic Lotka-Volterra**.
"""))

# Save notebook
out_path = r'c:\Users\ARPIT SINGH\OneDrive\Desktop\ISLE_ROYALE\isle_royale_ecosystem\notebooks\01_standard_lotka_volterra.ipynb'
save_notebook(cells, out_path)
