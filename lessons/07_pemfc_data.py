"""
LESSON 07 - PEM fuel cell degradation data and what "RUL" means
================================================================

Goals
  * Understand the data: what's measured on a PEMFC stack and how it ages.
  * Define end-of-life (EOL) and Remaining Useful Life (RUL) labels.
  * Clean the data and engineer health-indicator features.
  * Learn the piecewise-linear (capped) RUL target and why it's used.

Background (read pemfc_rul/simulate.py's docstring too)
  A 5-cell stack at constant current loses voltage over time:
    * irreversible: catalyst (Pt) dissolution/agglomeration -> less ECSA,
      membrane thinning, rising contact/ionic resistance, GDL hydrophobicity loss
    * reversible: water management, Pt oxide coverage; recovers after a stop or
      characterisation, producing a sawtooth
  The usual EOL criterion is a % drop of stack voltage vs beginning of life
  (DOE uses 10 % for automotive; the PHM 2014 challenge used 3.5-5 % on a
  ~1000 h test). Here: 5 % at reference conditions.

  RUL(t) = t_EOL - t, measured in operating hours.

Data source
  pemfc_rul.simulate creates a synthetic fleet with the same column names as
  the IEEE PHM 2014 FCLAB dataset. When you have real data, use
  pemfc_rul/adapters.py to load it into the same format.
"""

# %%
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pemfc_rul.data import DEFAULT_FEATURES, add_rul, clean, engineer
from pemfc_rul.simulate import SENSOR_COLUMNS, standard_datasets

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
ds = standard_datasets("data")
train, meta = ds["train"], ds["train_meta"]
print(train.shape, "rows;", train.unit.nunique(), "stacks")
print(train.head(3).T.head(12))

# %% Look before you model -----------------------------------------------------
print("\nLifetimes (h):")
print(meta["eol_time_h"].describe().round(0))
print("\nMissing values per column (sensor dropouts):")
print(train[SENSOR_COLUMNS].isna().sum()[lambda s: s > 0])

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for u in [1, 2, 3, 4, 5]:
    g = train[train.unit == u]
    axes[0, 0].plot(g.time_h, g.Utot, lw=0.7, label=f"unit {u} ({'dyn' if meta.set_index('unit').loc[u, 'dynamic_load'] else 'static'})")
axes[0, 0].set(title="Stack voltage (raw)", xlabel="h", ylabel="V"); axes[0, 0].legend(fontsize=7)

g = train[train.unit == 1]
for c in ["U1", "U2", "U3", "U4", "U5"]:
    axes[0, 1].plot(g.time_h, g[c].rolling(24).mean(), lw=0.8, label=c)
axes[0, 1].set(title="Unit 1 cell voltages (24 h mean): cells age differently", xlabel="h"); axes[0, 1].legend(fontsize=7)

axes[1, 0].plot(g.time_h, g.J, lw=0.5)
axes[1, 0].set(title="Unit 1 current density", xlabel="h", ylabel="A/cm²")

m = meta
axes[1, 1].scatter(m.stack_mult, m.eol_time_h, c=m.dynamic_load, cmap="coolwarm")
axes[1, 1].set(title="Life vs hidden stack quality (red = dynamic load)", xlabel="stack_mult (hidden)", ylabel="EOL h")
plt.tight_layout(); plt.savefig(OUT / "07_raw_data.png", dpi=110); plt.close()
print("saved outputs/07_raw_data.png  <- open it")
# Notice: (1) the sawtooth from maintenance stops, (2) glitches (sharp downward
# spikes), (3) dynamic-load stacks are noisier AND die sooner, (4) life varies a
# lot even for similar conditions: the hidden quality factor. No model can see
# that directly; it must infer it from how fast THIS stack is degrading.

# %% Cleaning ------------------------------------------------------------------
cleaned = clean(train)
print("\nNaNs after clean():", int(cleaned[SENSOR_COLUMNS].isna().sum().sum()))
fixed = (train["Utot"].to_numpy() - cleaned["Utot"].to_numpy()) < -0.03   # raw far BELOW cleaned
print("voltage glitches repaired (Utot vs sum of cell voltages):", int(fixed.sum()))

# %% Feature engineering: health indicators -------------------------------------
# Raw voltage depends on BOTH health and operating point (current, temperature).
# Good features isolate health:
#   v_rel         Utot / beginning-of-life Utot    (comparable across stacks)
#   v_rel_smooth  24 h rolling mean                (suppress load noise & sawtooth)
#   v_slope_48h   recent degradation RATE          (fast-ageing stacks have less life left)
#   cell_spread   std across cells                 (one failing cell = trouble)
#   min_cell_rel  weakest cell vs its own start
#   dp_air_rel    cathode pressure drop vs start   (GDL/flow-field ageing, flooding)
#   + operating conditions (J, temperatures, RH) so the model can "subtract" them
#   + time_h      operating hours (legit: you always know a stack's age)
feat = engineer(cleaned)
lab = add_rul(feat)                    # uncapped RUL
lab["RUL_cap600"] = lab["RUL"].clip(upper=600)
print("\nfeature table:")
print(lab[DEFAULT_FEATURES + ["RUL"]].describe().T[["mean", "std", "min", "max"]].round(4))

corr = lab[DEFAULT_FEATURES + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()
print("\ncorrelation with RUL (linear only; a hint, not the full story):")
print(corr.round(3))

# %% The RUL target: linear vs piecewise-linear ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for u in [1, 2, 3]:
    g = lab[lab.unit == u]
    axes[0].plot(g.time_h, g.v_rel_smooth, label=f"unit {u}")
axes[0].axhline(0.95, c="r", ls="--", label="EOL threshold")
axes[0].set(title="Health indicator v_rel_smooth", xlabel="h"); axes[0].legend()
g = lab[lab.unit == 1]
axes[1].plot(g.time_h, g.RUL, label="RUL (linear)")
axes[1].plot(g.time_h, g.RUL_cap600, label="RUL capped at 600 h")
axes[1].set(title="Two ways to label unit 1", xlabel="h", ylabel="RUL h"); axes[1].legend()
plt.tight_layout(); plt.savefig(OUT / "07_rul_labels.png", dpi=110); plt.close()
print("saved outputs/07_rul_labels.png")
# Why cap? Early in life every stack looks "new"; the data contains almost no
# information that separates a stack with 1200 h left from one with 900 h left.
# Asking the model to predict that difference mostly teaches it noise. With a
# cap, the model learns "healthy (>= 600 h)" early on, then a real countdown once
# degradation is visible, and the countdown is the part maintenance planning needs.
# The cap is a HYPERPARAMETER: choose it near the horizon you care about.

# Note: v_rel_smooth crossing 0.95 doesn't line up exactly with EOL because the
# simulator defines EOL at REFERENCE conditions, not at the measured operating
# point. Same as in real life: your HI is a proxy for the true state of health.

# %%
# EXERCISES
# 1. Plot v_rel_smooth for all static vs dynamic units on two subplots. Which
#    group has more spread in lifetime?
# 2. Create a new feature: hours since the last voltage "recovery" jump
#    (detect jumps with diff of v_rel_smooth > threshold). Does it correlate with RUL?
# 3. Simulate with eol_drop=0.035 (simulate_fleet(..., eol_drop=0.035)). How do
#    lifetimes change? Which threshold matters for YOUR application?
