"""
Isle Royale Ecosystem Modeling - Source Package
================================================

Reusable modules for modeling the predator-prey dynamics of the
Isle Royale wolf-moose ecosystem using ODE-based ecological models.

Modules
-------
metrics : R-squared, RMSE, and error evaluation functions.
optimizers : Differential Evolution wrappers and scipy.integrate ODE solvers.
"""

from .metrics import r_squared, rmse, normalized_mse_loss
from .optimizers import (
    solve_ode,
    run_differential_evolution,
    standard_lotka_volterra,
    logistic_lotka_volterra,
    rosenzweig_macarthur,
)

__all__ = [
    "r_squared",
    "rmse",
    "normalized_mse_loss",
    "solve_ode",
    "run_differential_evolution",
    "standard_lotka_volterra",
    "logistic_lotka_volterra",
    "rosenzweig_macarthur",
]
