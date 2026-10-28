"""
Adapters for real data.

IEEE PHM 2014 Data Challenge (FCLAB, FC1 static load / FC2 quasi-dynamic load):
    Files like FC1_Ageing.csv with headers such as "Time (h)", "U1 (V)", ...
    "Utot (V)", "J (A/cm²)", "I (A)", "TinH2 (°C)", "PinAIR (mbara)",
    "DinH2 (l/mn)", "HrAIRFC (%)". Raw sampling is a few minutes; we resample to
    1 h means, which is what the simulator produces.

    Note: FC1/FC2 were not run all the way to a 5 % voltage drop, so the RUL
    label has to be defined against a threshold you choose (e.g. 3.5 % or 4 %),
    or you use them only for fine-tuning / testing the shape of predictions.

Your own test bench:
    Map your column names onto the simulator's names (U1..U5, Utot, J, I,
    TinWAT, ToutWAT, PinAIR, PoutAIR, PinH2, PoutH2, HrAIRFC, ...) and you can
    reuse the whole pipeline. Use `load_generic(path, mapping)`.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .simulate import SENSOR_COLUMNS

# Columns the default feature engineering needs (see data.engineer / DEFAULT_FEATURES)
REQUIRED_COLUMNS = ["time_h", "U1", "U2", "U3", "U4", "U5", "Utot", "J",
                    "TinWAT", "ToutWAT", "HrAIRFC", "PinAIR", "PoutAIR", "PinH2", "PoutH2"]


def _strip_units(name: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*", "", str(name)).strip()


def load_phm2014(path: str, unit=1, resample_h: float = 1.0) -> pd.DataFrame:
    """Load an FCLAB PHM 2014 ageing CSV and resample it onto a regular hourly grid."""
    raw = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
    raw.columns = [_strip_units(c) for c in raw.columns]
    time_col = next(c for c in raw.columns if c.lower().startswith("time"))
    raw = raw.rename(columns={time_col: "time_h"})
    raw["time_h"] = pd.to_numeric(raw["time_h"], errors="coerce")
    raw = raw.dropna(subset=["time_h"])
    raw["bin"] = (np.floor(raw["time_h"] / resample_h) * resample_h)
    keep = [c for c in SENSOR_COLUMNS if c in raw.columns]
    out = raw.groupby("bin")[keep].mean().reset_index().rename(columns={"bin": "time_h"})
    out.insert(0, "unit", unit)
    return _fill_missing_columns(out)


def load_generic(path: str, mapping: dict | None = None, unit_col: str = "unit") -> pd.DataFrame:
    """Load any CSV; `mapping` renames your columns to the simulator's names."""
    df = pd.read_csv(path)
    if mapping:
        df = df.rename(columns=mapping)
    if unit_col != "unit" and unit_col in df:
        df = df.rename(columns={unit_col: "unit"})
    if "unit" not in df:
        df["unit"] = 1
    return _fill_missing_columns(df)


def _fill_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derive what can be derived; fail loudly on what can't."""
    if "Utot" not in df and all(f"U{i}" in df for i in range(1, 6)):
        df["Utot"] = df[[f"U{i}" for i in range(1, 6)]].sum(axis=1)
    missing = [c for c in REQUIRED_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"input is missing columns {missing}. Rename yours to match, "
                         f"or train a model with a feature list that doesn't need them.")
    return df
