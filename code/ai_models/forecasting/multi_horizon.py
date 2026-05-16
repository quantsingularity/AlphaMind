"""
Multi-horizon forecast evaluation utilities.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def evaluate_multi_horizon(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    horizons: List[int],
) -> Dict[str, Dict[str, float]]:
    """
    Compute IC, R2, MSE, and directional hit-rate for each horizon.

    Parameters
    ----------
    y_pred   : Predictions  ``(N, max_horizon)``.
    y_true   : Actuals      ``(N, max_horizon)``.
    horizons : List of 1-based horizon indices to evaluate.

    Returns
    -------
    dict
        ``{str(horizon): {metric: value, ...}, ...}``
    """
    from scipy.stats import spearmanr

    results: Dict[str, Dict[str, float]] = {}
    for h in horizons:
        hi = h - 1
        if hi >= y_pred.shape[1]:
            continue
        pred, true = y_pred[:, hi], y_true[:, hi]
        ic, _ = spearmanr(pred, true)
        mse = float(np.mean((pred - true) ** 2))
        var_t = float(np.var(true))
        r2 = 1.0 - mse / var_t if var_t > 0 else float("nan")
        hit = float(np.mean(np.sign(pred) == np.sign(true)))
        results[str(h)] = {"ic": float(ic), "r2": r2, "mse": mse, "hit_rate": hit}
    return results
