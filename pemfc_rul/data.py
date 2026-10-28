"""
Data pipeline: cleaning -> feature engineering -> RUL labels -> scaling -> windows.

Every function here is used by the lessons (07-13) and by the CLI, so the exact
same preprocessing runs at training time and at prediction time. Keeping those
two paths identical is one of the most important habits in applied ML.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CELL_COLS = ["U1", "U2", "U3", "U4", "U5"]

# Features fed to the models by default. Engineered ones are created by engineer().
DEFAULT_FEATURES = [
    "v_rel", "v_rel_smooth", "v_slope_48h", "cell_spread", "min_cell_rel",
    "J", "TinWAT", "ToutWAT", "HrAIRFC", "dp_air_rel", "dp_h2",
    "time_h",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, fill short gaps, and suppress obvious voltage glitches, per unit."""
    df = df.sort_values(["unit", "time_h"]).reset_index(drop=True)
    num_cols = [c for c in df.columns if c not in ("unit",) and pd.api.types.is_numeric_dtype(df[c])]
    df[num_cols] = df.groupby("unit")[num_cols].transform(lambda s: s.interpolate(limit_direction="both"))
    if "Utot" in df and all(c in df for c in CELL_COLS):
        # Glitch filter using redundancy: the stack voltage must equal the sum of
        # the cell voltages. A sensor glitch on Utot breaks that; a real load change
        # doesn't. (A plain rolling-median filter would "repair" genuine load swings.)
        cell_sum = df[CELL_COLS].sum(axis=1)
        bad = (df["Utot"] - cell_sum).abs() > 0.03
        df.loc[bad, "Utot"] = cell_sum[bad]
    return df


def engineer(df: pd.DataFrame, baseline_hours: float = 24.0) -> pd.DataFrame:
    """
    Build physically meaningful features. Most are *relative to the stack's own
    beginning-of-life*, which makes models transfer across stacks that start at
    slightly different voltages (a big deal with real hardware).
    """
    df = df.copy()
    g = df.groupby("unit", sort=False)

    first = df["time_h"] - g["time_h"].transform("min") <= baseline_hours
    def baseline(col):
        return df[col].where(first).groupby(df["unit"]).transform("median")

    df["v_rel"] = df["Utot"] / baseline("Utot")
    df["v_rel_smooth"] = g["v_rel"].transform(lambda s: s.rolling(24, min_periods=1).mean())
    # Local degradation rate: change in smoothed relative voltage over 48 h (per 1000 h)
    df["v_slope_48h"] = g["v_rel_smooth"].transform(lambda s: s.diff(48).fillna(0.0)) * (1000.0 / 48.0)
    cells = df[CELL_COLS]
    df["cell_spread"] = cells.std(axis=1)
    cell_rel = pd.concat([df[c] / baseline(c) for c in CELL_COLS], axis=1)
    df["min_cell_rel"] = cell_rel.min(axis=1)
    df["dp_air"] = df["PinAIR"] - df["PoutAIR"]
    df["dp_air_rel"] = df["dp_air"] / baseline("dp_air")
    df["dp_h2"] = df["PinH2"] - df["PoutH2"]
    return df


def add_rul(df: pd.DataFrame, cap: float | None = None, eol_col: str | None = None) -> pd.DataFrame:
    """
    Label each row with Remaining Useful Life (hours).

    For run-to-failure data the last row of each unit IS end-of-life, so
    RUL = t_last - t. With `cap`, use the piecewise-linear target
    min(RUL, cap): early in life the stack is "healthy" and the exact RUL is
    not predictable from data, so we stop asking the model to guess it.
    """
    df = df.copy()
    t_end = df.groupby("unit")["time_h"].transform("max") if eol_col is None else df[eol_col]
    df["RUL"] = t_end - df["time_h"]
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Raw sensor frame -> model-ready frame (no labels). Used by train AND predict."""
    return engineer(clean(df))


def split_units(units, val_frac=0.2, test_frac=0.0, seed=0):
    """
    Split by UNIT, never by row. Rows of the same stack are highly correlated;
    a random row split leaks the future of a stack into training and gives
    wildly optimistic scores.
    """
    units = np.array(sorted(set(units)))
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_test = int(round(len(units) * test_frac))
    n_val = max(1, int(round(len(units) * val_frac)))
    test, val, train = units[:n_test], units[n_test:n_test + n_val], units[n_test + n_val:]
    return train.tolist(), val.tolist(), test.tolist()


class Standardizer:
    """z = (x - mean) / std, fit on TRAINING data only. Serialisable for checkpoints."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def fit(self, x: np.ndarray):
        self.mean = np.nanmean(x, axis=0).astype(np.float32)
        self.std = (np.nanstd(x, axis=0) + 1e-8).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])


def make_windows(df: pd.DataFrame, features, window: int, stride: int = 1,
                 target: str | None = "RUL", last_only: bool = False):
    """
    Turn per-unit time series into fixed-length windows for sequence models.

    Returns X [N, window, F] float32, y [N] (or None), and an index DataFrame
    (unit, time_h) saying which unit/time each window ENDS at.

    Units shorter than `window` are left-padded by repeating their first row,
    so you can still predict on a stack that has only just started.
    """
    xs, ys, idx = [], [], []
    for unit, g in df.groupby("unit", sort=False):
        a = g[features].to_numpy(dtype=np.float32)
        if len(a) < window:
            a = np.concatenate([np.repeat(a[:1], window - len(a), axis=0), a])
            pad = window - len(g)
        else:
            pad = 0
        ends = [len(a) - 1] if last_only else list(range(window - 1, len(a), stride))
        if not last_only and ends[-1] != len(a) - 1:
            ends.append(len(a) - 1)  # always include the most recent point
        times = g["time_h"].to_numpy()
        yv = g[target].to_numpy(dtype=np.float32) if target and target in g else None
        for e in ends:
            xs.append(a[e - window + 1:e + 1])
            row = max(e - pad, 0)
            idx.append((unit, times[row]))
            if yv is not None:
                ys.append(yv[row])
    X = np.stack(xs)
    y = np.array(ys, dtype=np.float32) if ys else None
    return X, y, pd.DataFrame(idx, columns=["unit", "time_h"])


class WindowDataset(Dataset):
    """Minimal torch Dataset over pre-built windows."""

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.from_numpy(X)
        self.y = None if y is None else torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        if self.y is None:
            return self.X[i]
        return self.X[i], self.y[i]


def scale_windows(X: np.ndarray, scaler: Standardizer) -> np.ndarray:
    n, w, f = X.shape
    return scaler.transform(X.reshape(-1, f)).reshape(n, w, f)
