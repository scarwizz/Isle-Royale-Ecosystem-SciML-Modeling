# 🐺🌲 Isle Royale Predator-Prey Ecosystem: Scientific Machine Learning (SciML) Repository

Production-grade, highly reproducible repository modeling the multi-decadal predator-prey dynamics of the **Isle Royale wolf-moose ecosystem (1958–2026)** across three distinct mathematical formulations and an advanced suite of Scientific Machine Learning (SciML) and system identification architectures.

---

## 📂 Target Directory Structure

```text
isle_royale_ecosystem/
├── data/
│   └── isle_royale_census.csv                   # Annual winter survey census records (1958–2026)
├── src/
│   ├── __init__.py                              # Module re-exports
│   ├── metrics.py                               # R-squared, RMSE, normalized MSE loss, evaluate_fit
│   └── optimizers.py                            # ODE definitions, solve_ode wrappers, DE optimizers
├── notebooks/
│   ├── 01_standard_lotka_volterra.ipynb         # Classical LV, Peak Rolling, 3-Era, SINDy, WSINDy, UDE, PINNs
│   ├── 02_logistic_lotka_volterra.ipynb         # Logistic LV, Stability, Constrained SINDy, PyWynda, K-Sweep
│   └── 03_rosenzweig_macarthur_corrected.ipynb  # Corrected RM (e<=1.0), h-Sweep, 18-Param Era, OSF, Implicit SINDy, UDEs
└── README.md                                    # Comprehensive documentation & replication guide
```

---

## 🔬 Mathematical Formulations & SciML Architecture Matrix

| Notebook | Formulation | Functional Response | Phase Space Dynamics | SciML & System Identification Methods |
| :--- | :--- | :--- | :--- | :--- |
| **01: Standard LV** | $\begin{aligned}\dot{N}&=\alpha N - \beta N P\\\dot{P}&=\delta N P - \gamma P\end{aligned}$ | Linear (Type I) | Neutral concentric center orbits | • Global Differential Evolution<br>• Peak-to-Peak Rolling Validation (`find_peaks`)<br>• 3-Era Time-Varying Model (12 params)<br>• SINDy & Ensemble SINDy (eSINDy)<br>• Weak Formulation SINDy (WSINDy)<br>• State-Dependent UDE (`torchdiffeq`)<br>• PINN Differentiable Parameter Fine-Tuning |
| **02: Logistic LV** | $\begin{aligned}\dot{N}&=r N (1 - N/K) - a N P\\\dot{P}&=b N P - d P\end{aligned}$ | Linear (Type I) with Carrying Capacity $K$ | Asymptotically stable focus (damped spiral) | • Global Differential Evolution ($K \in [500, 5000]$)<br>• Nullcline Analysis & Coexistence Equilibrium<br>• Jacobian Linear Stability (Damped Oscillation Period)<br>• Constrained SINDy ($C\xi \le d$ biological signs)<br>• Constrained Ensemble SINDy (E-SINDy)<br>• WyNDA Adaptive Observer (`pywynda.WyNDAObserver`)<br>• 3-Era Logistic Model (15 params)<br>• Parametric $K$-Sensitivity Analysis |
| **03: Rosenzweig-MacArthur (Corrected)** | $\begin{aligned}\dot{N}&=r N (1 - N/K) - \frac{a N}{1+ahN} P\\\dot{P}&=e \frac{a N}{1+ahN} P - m P\end{aligned}$ | Holling Type II (Handling Time $h$) | Nonlinear Limit Cycles / Paradox of Enrichment | • **Mass Conservation Fix** ($e \le 1.0$, fixing unphysical $e=5.37$)<br>• 1D Handling Time Sweep ($h \in [10^{-3}, 1.0]$)<br>• 80/20 Train/Validation Split (1959–2012 / 2013–2026)<br>• Peak-to-Peak Cycle Validation<br>• 18-Parameter 3-Era Regime Model<br>• Phase Space Geometric Orbit Distance<br>• Observation Spectral Filtering (OSF)<br>• WyNDA Online Observer (`pywynda`)<br>• Batch Implicit SINDy (Holling Type II)<br>• UDE with Climate $c(t)$ & Genetic $g(t)$ MLPs<br>• Management Pulse $I(t)$ & 20-Year Counterfactuals |

---

## 🛠️ Critical Error Correction: Mass Conservation ($e \le 1.0$)

* **The Problem**: In raw, unconstrained Kaggle runs, multi-parameter regression yielded an impossible wolf assimilation efficiency of **$e = 5.37$**. This implied wolves convert 100 kg of consumed moose into **537 kg of wolf mass**—a severe violation of the First and Second Laws of Thermodynamics.
* **The Mathematical Correction**: We enforce strict physical inequality bounds:
  $$0 < e \le 1.0, \qquad h > 0$$
  both inside optimization bounding boxes and via hard exterior barrier penalties.
* **The Result**: Calibrated biological efficiency converges to $e \approx 0.034\text{--}0.08$ ($3\%\text{--}8\%$ trophic conversion efficiency), precisely matching mammalian physiological literature.

---

## 🌐 Kaggle Compatibility: Zero-Friction Execution

All three notebooks are **100% Kaggle-ready** and self-contained:
1. **Configurable Dataset Path**: Each notebook defines a single top-level variable:
   ```python
   DATA_PATH = "../data/isle_royale_census.csv"  # Local default
   # If running on Kaggle, the notebook automatically detects any dataset in /kaggle/input/
   ```
   Users can run the notebooks on Kaggle without editing code by simply adding the dataset to the session.
2. **Graceful Standalone Execution**: All helper metrics and ODE routines contain standalone inline definitions if imported without the `src/` directory.
3. **Hardware Agnostic**: Automatically detects and leverages GPU acceleration via `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`, falling back seamlessly to CPU.

---

## 🚀 Environment Setup & Execution

### 1. Local Installation
```bash
# Clone or navigate to the repository
cd isle_royale_ecosystem

# Install dependencies
pip install numpy scipy pandas matplotlib scikit-learn pysindy pywynda torch torchdiffeq
```

### 2. Running the Pipelines Locally
```bash
# Launch Jupyter Notebook
jupyter notebook notebooks/
```

### 3. Running on Kaggle
1. Upload any notebook (`01_standard_lotka_volterra.ipynb`, `02_logistic_lotka_volterra.ipynb`, or `03_rosenzweig_macarthur_corrected.ipynb`) to a Kaggle Notebook session.
2. Attach the Isle Royale dataset CSV (e.g. `arpitsingh5134/isle-royale`).
3. Run the first cell to auto-install any missing specialized packages (`pysindy`, `pywynda`, `torchdiffeq`).
4. Execute All Cells.
