"""
LESSON 11 - Uncertainty: how sure is the RUL prediction?
=========================================================

Goals
  * Know the two kinds of uncertainty and how to estimate each:
      aleatoric  noise inherent in the problem (two stacks that look identical
                 today can still live 200 h apart). -> predict a distribution
      epistemic  the model's own ignorance from limited data; shrinks with more
                 data. -> ensembles / MC dropout
  * Train with Gaussian negative log-likelihood (mean + variance head).
  * Use MC dropout. Write a quantile (pinball) loss.
  * CHECK calibration: does a "90 % interval" contain the truth 90 % of the time?
  * Turn a predictive distribution into a maintenance decision.

A point estimate of "250 h" is not enough to plan maintenance. "250 h, 90 %
interval 180-330 h" is; you schedule against the lower bound.
"""

# %%
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pemfc_rul.data import (DEFAULT_FEATURES, Standardizer, WindowDataset, add_rul, make_windows,
                            prepare, scale_windows, split_units)
from pemfc_rul.metrics import coverage, report
from pemfc_rul.models import RNNRegressor
from pemfc_rul.predict import Predictor
from pemfc_rul.simulate import standard_datasets
from pemfc_rul.train import TrainConfig, train

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
ds = standard_datasets("data")
CAP = 600.0
Z = {0.5: 0.674, 0.8: 1.282, 0.9: 1.645, 0.95: 1.960}   # Gaussian half-widths for central intervals

def calibration(y, mean, std, label):
    row = "  ".join(f"{int(q * 100)}%:{coverage(y, mean - z * std, mean + z * std):.0%}" for q, z in Z.items())
    print(f"{label:<28} coverage  {row}   (ideal = the nominal level)")

full = ds["test_full"]
labels = add_rul(full[["unit", "time_h"]].copy(), cap=CAP)

# %% 1. Gaussian NLL: predict mean AND variance -------------------------------------
# Loss = 0.5 * [ log(var) + (y - mean)^2 / var ]. The model is rewarded for
# small variance only where it is actually accurate.
cfg = TrainConfig(data="data/train.csv", out="models/l11_gru_gauss.pt", model="gru",
                  loss="gaussian", rul_cap=CAP, epochs=40, patience=8, dropout=0.2)
train(cfg, verbose=False)
pg = Predictor(cfg.out)
traj = pg.predict_frame(full, all_steps=True).merge(labels, on=["unit", "time_h"])
print("Gaussian head:", {k: round(v, 3) for k, v in report(traj.RUL, traj.rul_h, traj.rul_std_h).items()})
calibration(traj.RUL, traj.rul_h, traj.rul_std_h, "Gaussian head (aleatoric)")

# %% 2. MC dropout: epistemic uncertainty from the same network ---------------------
trajm = pg.predict_frame(full, all_steps=True, mc_samples=30).merge(labels, on=["unit", "time_h"])
calibration(trajm.RUL, trajm.rul_h, trajm.rul_std_h, "Gaussian + MC dropout")

# A point-loss model with MC dropout ONLY (epistemic only) is usually overconfident:
pp = Predictor("models/l09_gru.pt") if Path("models/l09_gru.pt").exists() else None
if pp is not None:
    t2 = pp.predict_frame(full, all_steps=True, mc_samples=30).merge(labels, on=["unit", "time_h"])
    calibration(t2.RUL, t2.rul_h, t2.rul_std_h, "MC dropout only (lesson 09)")
# Typical outcome: MC dropout alone is badly OVER-confident (intervals too
# narrow: it ignores the inherent stack-to-stack scatter), while the Gaussian
# head may be UNDER-confident (too wide; its variance is learned from only ~32
# stacks). Neither is "right" out of the box. You must measure calibration and
# usually correct it on validation data (exercise 2).

# %% 3. Quantile regression with the pinball loss -----------------------------------
# Predict several quantiles directly, no distribution shape assumed.
# Pinball loss for quantile q: max(q * e, (q - 1) * e), e = y - pred.
# For q = 0.05 under-predictions are cheap and over-predictions are expensive,
# so the output settles at the 5th percentile.
QS = torch.tensor([0.05, 0.5, 0.95])
def pinball(pred, y):
    e = y.unsqueeze(1) - pred                      # [B, n_quantiles]
    return torch.maximum(QS * e, (QS - 1) * e).mean()

frame = add_rul(prepare(ds["train"]), cap=CAP)
tr_u, va_u, _ = split_units(frame.unit.unique(), val_frac=0.2, seed=0)
Xtr, ytr, _ = make_windows(frame[frame.unit.isin(tr_u)], DEFAULT_FEATURES, 96, stride=4)
scaler = Standardizer().fit(Xtr.reshape(-1, Xtr.shape[-1]))
dl = DataLoader(WindowDataset(scale_windows(Xtr, scaler), ytr / CAP), batch_size=256, shuffle=True)
torch.manual_seed(0)
qmodel = RNNRegressor(n_features=len(DEFAULT_FEATURES), window=96, hidden=64, layers=2, dropout=0.1,
                      out_dim=3, cell="gru")
opt = torch.optim.AdamW(qmodel.parameters(), lr=1e-3, weight_decay=1e-4)
for epoch in range(12):
    qmodel.train()
    for xb, yb in dl:
        loss = pinball(qmodel(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
qmodel.eval()
Xf, yf, _ = make_windows(add_rul(prepare(full), cap=CAP), DEFAULT_FEATURES, 96, stride=1)
with torch.no_grad():
    q = qmodel(torch.from_numpy(scale_windows(Xf, scaler))).numpy() * CAP
q = np.sort(q, axis=1)                               # guard against quantile crossing
print(f"quantile GRU: median RMSE {report(yf, q[:, 1])['rmse_h']:.1f} h, "
      f"90% interval coverage {coverage(yf, q[:, 0], q[:, 2]):.0%}, "
      f"mean width {np.mean(q[:, 2] - q[:, 0]):.0f} h")

# %% 4. Plot predictive bands ------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, u in zip(axes, full.unit.unique()[:3]):
    g = trajm[trajm.unit == u]
    ax.fill_between(g.time_h, g.rul_p05_h, g.rul_p95_h, alpha=0.3, label="90% interval")
    ax.plot(g.time_h, g.rul_h, lw=1, label="mean")
    ax.plot(g.time_h, g.RUL, "k", label="true (capped)")
    ax.set(title=f"unit {u}", xlabel="h")
axes[0].set_ylabel("RUL h"); axes[0].legend()
plt.tight_layout(); plt.savefig(OUT / "11_uncertainty.png", dpi=110); plt.close()
print("saved outputs/11_uncertainty.png")

# %% 5. From distribution to decision ------------------------------------------------
# Rule: schedule a stack replacement when P(RUL < lead_time) exceeds a risk
# tolerance. With a Gaussian, P(RUL < L) = Phi((L - mean) / std).
from math import erf, sqrt
def p_fail_before(mean, std, lead_h):
    return 0.5 * (1 + erf((lead_h - mean) / (sqrt(2) * max(std, 1e-6))))

lead, risk = 100.0, 0.05     # need 100 h to get a spare stack; accept 5 % risk
now = pg.predict_frame(ds["test"], mc_samples=30).merge(ds["test_truth"], on="unit")
now["p_fail_before_lead"] = [p_fail_before(m, s, lead) for m, s in zip(now.rul_h, now.rul_std_h)]
now["action"] = np.where(now.p_fail_before_lead > risk, "ORDER SPARE", "ok")
print(now[["unit", "rul_h", "rul_std_h", "p_fail_before_lead", "action", "rul_true_h"]].round(2).to_string(index=False))

# %%
# EXERCISES
# 1. Deep ensemble: train the gaussian GRU with 5 seeds (TrainConfig(seed=k)).
#    Combine: mean of means; variance = mean of variances + variance of means.
#    Is it better calibrated than MC dropout? (Usually yes, at 5x the cost.)
# 2. Post-hoc calibration: find a factor s so that mean +/- 1.645*s*std covers
#    90 % on the VALIDATION stacks; apply s to test. (Never fit s on test.)
# 3. Show how coverage depends on true RUL: split by RUL bucket. Where is the
#    model over/under-confident?
