"""
ODE Definitions and Optimization Wrappers
==========================================

Contains the three ecological ODE models (Standard Lotka-Volterra,
Logistic Lotka-Volterra, Rosenzweig-MacArthur) and wrapper functions
around ``scipy.integrate.solve_ivp`` and
``scipy.optimize.differential_evolution``.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

from .metrics import normalized_mse_loss


# ===================================================================
#  ODE Model Definitions
# ===================================================================

def standard_lotka_volterra(t, y, alpha, beta, delta, gamma):
    """
    Standard (classical) Lotka-Volterra predator-prey equations.

    dN/dt = alpha * N  -  beta * N * P
    dP/dt = delta * N * P  -  gamma * P

    Parameters
    ----------
    t : float
        Current time (unused, required by solve_ivp).
    y : array-like
        [N, P] — prey and predator populations.
    alpha : float
        Prey intrinsic growth rate.
    beta : float
        Predation rate coefficient.
    delta : float
        Predator reproduction rate per prey consumed.
    gamma : float
        Predator natural death rate.

    Returns
    -------
    list of float
        [dN/dt, dP/dt]
    """
    N, P = max(y[0], 0.0), max(y[1], 0.0)
    dN = alpha * N - beta * N * P
    dP = delta * N * P - gamma * P
    return [dN, dP]


def logistic_lotka_volterra(t, y, r, K, a, b, d):
    """
    Logistic Lotka-Volterra model with prey carrying capacity.

    dN/dt = r * N * (1 - N/K) - a * N * P
    dP/dt = b * N * P  -  d * P

    Parameters
    ----------
    t : float
        Current time.
    y : array-like
        [N, P] — prey and predator populations.
    r : float
        Prey intrinsic growth rate.
    K : float
        Prey carrying capacity.
    a : float
        Attack (predation) rate.
    b : float
        Predator conversion efficiency × attack rate.
    d : float
        Predator death rate.

    Returns
    -------
    list of float
        [dN/dt, dP/dt]
    """
    N, P = max(y[0], 0.0), max(y[1], 0.0)
    dN = r * N * (1.0 - N / K) - a * N * P
    dP = b * N * P - d * P
    return [dN, dP]


def rosenzweig_macarthur(t, y, r, K, a, h, e, m):
    """
    Rosenzweig-MacArthur model with Holling Type II functional response.

    dN/dt = r * N * (1 - N/K) - (a*N / (1 + a*h*N)) * P
    dP/dt = e * (a*N / (1 + a*h*N)) * P  -  m * P

    Parameters
    ----------
    t : float
        Current time.
    y : array-like
        [N, P] — prey and predator populations.
    r : float
        Prey intrinsic growth rate.
    K : float
        Prey carrying capacity.
    a : float
        Attack rate.
    h : float
        Handling time per prey item.
    e : float
        Assimilation (conversion) efficiency.  Must satisfy 0 < e ≤ 1.
    m : float
        Predator mortality rate.

    Returns
    -------
    list of float
        [dN/dt, dP/dt]
    """
    N, P = max(y[0], 0.0), max(y[1], 0.0)
    functional_response = (a * N) / (1.0 + a * h * N)
    dN = r * N * (1.0 - N / K) - functional_response * P
    dP = e * functional_response * P - m * P
    return [dN, dP]


# ===================================================================
#  ODE Solver Wrapper
# ===================================================================

def solve_ode(ode_func, t_span, y0, args, t_eval, method="RK45",
              max_step=np.inf):
    """
    Solve an ODE initial-value problem with ``scipy.integrate.solve_ivp``.

    Always checks ``sol.success`` and ensures the returned time array
    matches ``t_eval`` exactly.

    Parameters
    ----------
    ode_func : callable
        Right-hand side of the ODE  dy/dt = f(t, y, *args).
    t_span : tuple of float
        (t_start, t_end).
    y0 : array-like
        Initial conditions [N0, P0].
    args : tuple
        Extra arguments forwarded to *ode_func*.
    t_eval : np.ndarray
        Times at which to store the solution.
    method : str, optional
        Integration method (default ``"RK45"``).
    max_step : float, optional
        Maximum step size (default ``np.inf``).

    Returns
    -------
    sol : OdeResult or None
        The solution object if successful; ``None`` otherwise.
    """
    sol = solve_ivp(
        ode_func,
        t_span,
        y0,
        args=args,
        t_eval=t_eval,
        method=method,
        max_step=max_step,
        dense_output=False,
    )
    if not sol.success:
        return None
    if sol.y.shape[1] != len(t_eval):
        return None
    # Check for NaN / Inf
    if np.any(~np.isfinite(sol.y)):
        return None
    return sol


# ===================================================================
#  Differential Evolution Wrapper
# ===================================================================

def run_differential_evolution(
    objective,
    bounds,
    seed=42,
    maxiter=200,
    tol=1e-10,
    popsize=25,
    mutation=(0.5, 1.5),
    recombination=0.9,
    polish=True,
    disp=False,
    **kwargs,
):
    """
    Run ``scipy.optimize.differential_evolution`` with sensible defaults.

    Parameters
    ----------
    objective : callable
        Scalar objective function to minimise.
    bounds : list of tuple
        [(lo, hi), ...] for each parameter.
    seed : int
        Random seed for reproducibility.
    maxiter : int
        Maximum number of generations.
    tol : float
        Convergence tolerance.
    popsize : int
        Population multiplier (per parameter).
    mutation : tuple
        Differential weight range.
    recombination : float
        Cross-over probability.
    polish : bool
        Whether to polish the result with L-BFGS-B.
    disp : bool
        Whether to print convergence messages.
    **kwargs
        Extra keyword arguments forwarded to ``differential_evolution``.

    Returns
    -------
    result : OptimizeResult
        Optimisation result.
    """
    return differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=maxiter,
        tol=tol,
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        polish=polish,
        disp=disp,
        **kwargs,
    )


# ===================================================================
#  Pre-built Objective Factories
# ===================================================================

def make_objective_standard_lv(years, moose, wolves):
    """
    Return an objective function for the Standard Lotka-Volterra model.

    Parameters
    ----------
    years, moose, wolves : np.ndarray
        Observed time series.

    Returns
    -------
    callable
        Objective function  f(params) → float.
    """
    def objective(params):
        alpha, beta, delta, gamma = params
        sol = solve_ode(
            standard_lotka_volterra,
            [years[0], years[-1]],
            [moose[0], wolves[0]],
            args=(alpha, beta, delta, gamma),
            t_eval=years,
        )
        if sol is None:
            return 1e12
        return normalized_mse_loss(moose, sol.y[0], wolves, sol.y[1])
    return objective


def make_objective_logistic_lv(years, moose, wolves):
    """
    Return an objective function for the Logistic Lotka-Volterra model.

    Parameters
    ----------
    years, moose, wolves : np.ndarray
        Observed time series.

    Returns
    -------
    callable
        Objective function  f(params) → float.
    """
    def objective(params):
        r, K, a, b, d = params
        sol = solve_ode(
            logistic_lotka_volterra,
            [years[0], years[-1]],
            [moose[0], wolves[0]],
            args=(r, K, a, b, d),
            t_eval=years,
        )
        if sol is None:
            return 1e12
        return normalized_mse_loss(moose, sol.y[0], wolves, sol.y[1])
    return objective


def make_objective_rosenzweig_macarthur(years, moose, wolves, h_fixed=None):
    """
    Return an objective function for the Rosenzweig-MacArthur model.

    When *h_fixed* is given the handling time is held constant and
    the parameter vector has length 4: [r, K, a, e, m]  (h omitted).
    Otherwise the full 6-parameter vector [r, K, a, h, e, m] is used.

    **Critical**: the objective enforces  0 < e ≤ 1  (mass conservation).

    Parameters
    ----------
    years, moose, wolves : np.ndarray
        Observed time series.
    h_fixed : float or None
        If provided, handling time is fixed at this value.

    Returns
    -------
    callable
        Objective function  f(params) → float.
    """
    def objective(params):
        if h_fixed is not None:
            r, K, a, e, m = params
            h = h_fixed
        else:
            r, K, a, h, e, m = params

        # Strict physical bounds — penalise violations
        if e > 1.0 or e <= 0.0 or h <= 0.0:
            return 1e12

        sol = solve_ode(
            rosenzweig_macarthur,
            [years[0], years[-1]],
            [moose[0], wolves[0]],
            args=(r, K, a, h, e, m),
            t_eval=years,
        )
        if sol is None:
            return 1e12
        return normalized_mse_loss(moose, sol.y[0], wolves, sol.y[1])
    return objective
