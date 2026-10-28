"""
RUL evaluation metrics.

RMSE/MAE treat early and late predictions the same. In prognostics they are NOT
the same: predicting a stack has 300 h left when it really has 100 h (a LATE
prediction, error > 0) means it fails in service. Predicting too little life
(EARLY) only costs an unnecessary maintenance stop. The PHM score below is the
asymmetric exponential score from the PHM'08 challenge, with time constants
rescaled to hours.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def phm_score(y_true, y_pred, a_early=130.0, a_late=100.0):
    """Sum over units of exp(|d|/a)-1, d = pred - true. Lower is better; late is punished harder."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    s = np.where(d < 0, np.exp(-d / a_early) - 1, np.exp(d / a_late) - 1)
    return float(np.sum(s))


def late_fraction(y_true, y_pred, tol=0.0):
    """Share of predictions that over-estimate the remaining life by more than tol hours."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true) > tol))


def alpha_accuracy(y_true, y_pred, alpha=0.2, min_band=24.0):
    """Share of predictions within +/- alpha * true RUL (at least +/- min_band hours)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    band = np.maximum(alpha * y_true, min_band)
    return float(np.mean(np.abs(y_pred - y_true) <= band))


def coverage(y_true, lo, hi):
    """Fraction of true values inside predicted intervals: checks uncertainty calibration."""
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= np.asarray(lo)) & (y_true <= np.asarray(hi))))


def report(y_true, y_pred, std=None):
    out = {
        "n": int(len(y_true)),
        "rmse_h": rmse(y_true, y_pred),
        "mae_h": mae(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
        "late_fraction": late_fraction(y_true, y_pred),
        "alpha20_accuracy": alpha_accuracy(y_true, y_pred),
    }
    if std is not None:
        std = np.asarray(std)
        out["coverage_90"] = coverage(y_true, np.asarray(y_pred) - 1.645 * std, np.asarray(y_pred) + 1.645 * std)
    return out
