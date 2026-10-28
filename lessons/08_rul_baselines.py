"""
LESSON 08 - RUL baselines and how to score them
===============================================

Goals
  * Build three baselines BEFORE any deep learning:
      1. Fleet average:   RUL = mean life - age
      2. Trend extrapolation (classic model-based prognostics): fit the recent
         health-indicator trend and extrapolate it to the EOL threshold
      3. MLP on hand-crafted features at one time step (no history window)
  * Understand RUL metrics: RMSE, MAE, the asymmetric PHM score, late predictions.

Why baselines? A neural net that can't beat "fit a line and extrapolate" isn't
worth deploying. Baselines also tell you how hard the problem is.

Evaluation protocol (used for the rest of the course):
  * test.csv: 15 unseen stacks, each cut at a random "now". One prediction per
    stack; compare with test_truth.csv. That's the deployment situation.
  * test_full.csv: the same stacks run to failure, used to plot RUL trajectories.
"""

# %%
import numpy as np
import pandas as pd
import torch
from torch import nn

from pemfc_rul.data import DEFAULT_FEATURES, Standardizer, add_rul, prepare, split_units
from pemfc_rul.metrics import report
from pemfc_rul.simulate import standard_datasets

torch.manual_seed(0)
ds = standard_datasets("data")
train = add_rul(prepare(ds["train"]))
test = prepare(ds["test"])
truth = ds["test_truth"].set_index("unit")["rul_true_h"]
last = test.groupby("unit").tail(1).set_index("unit")      # the "now" row of each test stack
y_true = truth.loc[last.index].to_numpy()

def show(name, y_pred):
    r = report(y_true, y_pred)
    print(f"{name:<30} RMSE {r['rmse_h']:6.1f} h  MAE {r['mae_h']:6.1f} h  "
          f"PHM {r['phm_score']:7.2f}  late {r['late_fraction']:.0%}")
    return r

# %% Baseline 1: fleet average life ----------------------------------------------
mean_life = train.groupby("unit")["time_h"].max().mean()
show("1. mean life - age", np.clip(mean_life - last["time_h"].to_numpy(), 0, None))

# %% Baseline 2: trend extrapolation ---------------------------------------------
def extrapolate(g, threshold=0.95, lookback=200, degree=1):
    """Fit v_rel_smooth over the last `lookback` hours; solve for the threshold crossing."""
    g = g[g.time_h >= g.time_h.max() - lookback]
    t, v = g.time_h.to_numpy(), g.v_rel_smooth.to_numpy()
    if len(t) < 24:
        return np.nan
    coef = np.polyfit(t, v, degree)
    now = t[-1]
    future = np.arange(now, now + 5000, 1.0)
    vf = np.polyval(coef, future)
    hit = np.nonzero(vf <= threshold)[0]
    return float(future[hit[0]] - now) if len(hit) else 5000.0

for lb, deg in [(200, 1), (400, 1), (400, 2)]:
    pred = np.array([extrapolate(test[test.unit == u], lookback=lb, degree=deg) for u in last.index])
    pred = np.nan_to_num(pred, nan=mean_life)
    show(f"2. trend lookback={lb} deg={deg}", np.clip(pred, 0, 1500))
# The PHM score explodes because it is EXPONENTIAL in the error: one stack
# predicted 1000 h late contributes e^10 ~ 22000. That's intended: a metric for
# a safety/cost-critical decision should punish wild over-estimates hard.
# Physically intuitive but fragile: noisy slopes early in life give wild answers,
# and the threshold on the *measured* HI isn't exactly the true EOL criterion.
# Calibrating the threshold on training stacks is a good exercise (below).

# %% Baseline 3: MLP on single-time-step features ----------------------------------
CAP = 600.0
feats = DEFAULT_FEATURES
tr_u, va_u, _ = split_units(train.unit.unique(), val_frac=0.2, seed=0)
tr, va = train[train.unit.isin(tr_u)], train[train.unit.isin(va_u)]
scaler = Standardizer().fit(tr[feats].to_numpy())
def tensors(df):
    X = torch.tensor(scaler.transform(df[feats].to_numpy()))
    y = torch.tensor(df["RUL"].clip(upper=CAP).to_numpy(dtype=np.float32)) / CAP
    return X, y
Xtr, ytr = tensors(tr); Xva, yva = tensors(va)

mlp = nn.Sequential(nn.Linear(len(feats), 128), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
best, best_state = 1e9, None
for epoch in range(40):
    mlp.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 256):
        idx = perm[i:i + 256]
        loss = nn.functional.huber_loss(mlp(Xtr[idx]).squeeze(-1), ytr[idx], delta=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    mlp.eval()
    with torch.no_grad():
        vl = nn.functional.mse_loss(mlp(Xva).squeeze(-1), yva).item()
    if vl < best:
        best, best_state = vl, {k: v.clone() for k, v in mlp.state_dict().items()}
mlp.load_state_dict(best_state)
print(f"   (MLP best val RMSE on capped target: {best ** .5 * CAP:.1f} h)")
with torch.no_grad():
    p = mlp(torch.tensor(scaler.transform(last[feats].to_numpy()))).squeeze(-1).numpy() * CAP
show("3. MLP, single time step", p)

# %% Understanding the PHM score -------------------------------------------------
from pemfc_rul.metrics import phm_score
print("\nPHM score for a single prediction error of:")
for d in [-100, -50, 50, 100]:
    print(f"  {d:+4d} h -> {phm_score([300], [300 + d]):.2f}")
print("Late predictions (+) cost more: a stack failing before its predicted EOL is the expensive case.")

# Caveat about evaluating with a cap: a stack whose true RUL is 900 h but which
# the model says is ">= 600 h (at cap)" is not really wrong. Note how many test
# stacks have true RUL above the cap:
print(f"\ntest stacks with true RUL > cap: {(y_true > CAP).sum()} of {len(y_true)}")

# %%
# EXERCISES
# 1. Calibrate the trend-extrapolation threshold: for each TRAINING stack, read
#    v_rel_smooth at its EOL. Use the median as the threshold. Better?
# 2. Evaluate every method at many cut points on test_full.csv (e.g. every 50 h)
#    and plot RMSE vs true RUL bucket (0-100, 100-200, ...). Where are errors largest?
# 3. Remove time_h from the MLP features. What happens and why?
