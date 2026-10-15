"""
Evaluation Metrics for Ecological Models
=========================================

Provides R-squared, RMSE, and normalized MSE loss functions for
comparing model trajectories against observed census data.
"""

import numpy as np


def r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute the coefficient of determination (R²).

    Parameters
    ----------
    observed : np.ndarray
        Observed (actual) values.
    predicted : np.ndarray
        Model-predicted values.

    Returns
    -------
    float
        R² value.  1.0 indicates a perfect fit; values below 0
        indicate the model is worse than predicting the mean.
    """
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute the Root Mean Squared Error (RMSE).

    Parameters
    ----------
    observed : np.ndarray
        Observed (actual) values.
    predicted : np.ndarray
        Model-predicted values.

    Returns
    -------
    float
        RMSE value (same units as the data).
    """
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def normalized_mse_loss(
    observed_prey: np.ndarray,
    predicted_prey: np.ndarray,
    observed_predator: np.ndarray,
    predicted_predator: np.ndarray,
) -> float:
    """
    Compute a variance-normalized MSE loss summed over both species.

    The loss for each species is  MSE / Var(observed), making the
    prey and predator contributions comparable regardless of scale.

    Parameters
    ----------
    observed_prey : np.ndarray
        Observed prey (moose) counts.
    predicted_prey : np.ndarray
        Model-predicted prey counts.
    observed_predator : np.ndarray
        Observed predator (wolf) counts.
    predicted_predator : np.ndarray
        Model-predicted predator counts.

    Returns
    -------
    float
        Combined normalized MSE loss.
    """
    var_prey = np.var(observed_prey)
    var_pred = np.var(observed_predator)
    if var_prey == 0.0 or var_pred == 0.0:
        return 1e12

    loss_prey = np.mean((observed_prey - predicted_prey) ** 2) / var_prey
    loss_predator = np.mean((observed_predator - predicted_predator) ** 2) / var_pred
    return float(loss_prey + loss_predator)


def evaluate_fit(
    observed_prey: np.ndarray,
    predicted_prey: np.ndarray,
    observed_predator: np.ndarray,
    predicted_predator: np.ndarray,
    species_names: tuple[str, str] = ("Moose", "Wolf"),
) -> dict:
    """
    Compute and return a full evaluation report for both species.

    Parameters
    ----------
    observed_prey, predicted_prey : np.ndarray
        Observed and predicted prey values.
    observed_predator, predicted_predator : np.ndarray
        Observed and predicted predator values.
    species_names : tuple of str
        Labels for the prey and predator species (default: Moose, Wolf).

    Returns
    -------
    dict
        Dictionary with R², RMSE, and normalized MSE for each species.
    """
    prey_name, pred_name = species_names
    return {
        f"{prey_name}_R2": r_squared(observed_prey, predicted_prey),
        f"{prey_name}_RMSE": rmse(observed_prey, predicted_prey),
        f"{pred_name}_R2": r_squared(observed_predator, predicted_predator),
        f"{pred_name}_RMSE": rmse(observed_predator, predicted_predator),
        "NormalizedMSELoss": normalized_mse_loss(
            observed_prey, predicted_prey,
            observed_predator, predicted_predator,
        ),
    }
