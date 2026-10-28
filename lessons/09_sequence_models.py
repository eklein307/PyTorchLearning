"""
LESSON 09 - Sequence models for RUL: windows, 1D-CNN, LSTM, GRU
================================================================

Goals
  * Turn per-stack time series into training windows [batch, window, features].
  * Understand what CNNs and RNNs (LSTM/GRU) do with a sequence.
  * Train them with a hand-written loop, then with pemfc_rul.train.
  * Compare models on unseen stacks and plot RUL trajectories.

Why look at a window instead of one time step (lesson 08's MLP)?
  The RATE and SHAPE of degradation over the recent past say a lot about the
  remaining life: a stack losing 30 uV/h has more left than one losing 80 uV/h
  at the same voltage. One time step can't show that; a window can.

The architectures (see pemfc_rul/models.py)
  1D-CNN  slides small filters along time: detects local patterns (slopes,
          recoveries after stops). Fast, parallel, fixed receptive field.
  LSTM    reads the window step by step, carrying a memory (hidden + cell
          state) with gates that decide what to keep/forget. Good at long-ish
          dependencies; sequential, so slower.
  GRU     a simpler gated RNN (2 gates instead of 3). Often equal to LSTM, cheaper.
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
from pemfc_rul.metrics import report
from pemfc_rul.models import RNNRegressor, count_parameters
from pemfc_rul.predict import Predictor
from pemfc_rul.simulate import standard_datasets
from pemfc_rul.train import TrainConfig, train

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
torch.manual_seed(0)
ds = standard_datasets("data")
CAP, WINDOW = 600.0, 96

# %% 1. Windowing --------------------------------------------------------------
frame = add_rul(prepare(ds["train"]), cap=CAP)
X, y, idx = make_windows(frame[frame.unit == 1], DEFAULT_FEATURES, window=WINDOW, stride=24)
print("unit 1 windows:", X.shape, "-> [n_windows, window_hours, n_features]")
print(idx.head(3).assign(RUL=y[:3]))
# Each window ends at time_h and is labelled with the RUL AT ITS LAST STEP.
# stride=24: one window per day. Smaller stride = more (but more correlated) samples.

# %% 2. A hand-written training run (LSTM) -------------------------------------
tr_u, va_u, _ = split_units(frame.unit.unique(), val_frac=0.2, seed=0)
Xtr, ytr, _ = make_windows(frame[frame.unit.isin(tr_u)], DEFAULT_FEATURES, WINDOW, stride=4)
Xva, yva, _ = make_windows(frame[frame.unit.isin(va_u)], DEFAULT_FEATURES, WINDOW, stride=4)
scaler = Standardizer().fit(Xtr.reshape(-1, Xtr.shape[-1]))   # fit on TRAIN windows only
Xtr, Xva = scale_windows(Xtr, scaler), scale_windows(Xva, scaler)
tr_dl = DataLoader(WindowDataset(Xtr, ytr / CAP), batch_size=256, shuffle=True)
va_dl = DataLoader(WindowDataset(Xva, yva / CAP), batch_size=1024)

device = torch.device("cpu")   # small model: CPU is fine. Try "mps"/"cuda".
model = RNNRegressor(n_features=len(DEFAULT_FEATURES), window=WINDOW, hidden=64, layers=2,
                     dropout=0.1, cell="lstm").to(device)
print(f"\nLSTM parameters: {count_parameters(model):,}")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.HuberLoss(delta=0.1)   # robust to the label noise near the cap

for epoch in range(12):
    model.train()
    for xb, yb in tr_dl:
        pred = model(xb.to(device))[:, 0]
        loss = loss_fn(pred, yb.to(device))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # RNNs: clip gradients
        opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(xb.to(device))[:, 0].cpu() for xb, _ in va_dl]).numpy() * CAP
    print(f"epoch {epoch:2d}  val RMSE {report(yva, p)['rmse_h']:6.1f} h")
# That's the whole recipe. pemfc_rul/train.py is the same loop plus early
# stopping, a LR schedule, checkpointing, and config handling.

# %% 3. Compare architectures with the project's train() ---------------------------
results = {}
for name in ["mlp", "cnn", "lstm", "gru"]:
    cfg = TrainConfig(data="data/train.csv", out=f"models/l09_{name}.pt", model=name,
                      window=WINDOW, rul_cap=CAP, epochs=40, patience=8)  # device="auto"
    print(f"\n--- {name}")
    train(cfg, verbose=False)
    p = Predictor(cfg.out)
    pred = p.predict_frame(ds["test"]).merge(ds["test_truth"], on="unit")
    r = report(pred.rul_true_h, pred.rul_h)
    results[name] = r
    print(f"{name:5s} val RMSE {p.ckpt['val_metrics']['rmse_h']:6.1f} h | TEST (15 stacks) "
          f"RMSE {r['rmse_h']:6.1f} h  PHM {r['phm_score']:6.2f}  late {r['late_fraction']:.0%}")
# Expect them to be in the same ballpark and to beat the lesson-08 baselines.
# With only 32 training stacks, differences between architectures are often
# smaller than differences between random seeds (lesson 12 measures that).

# %% 4. RUL trajectories: the plot engineers actually look at -----------------------
best = min(results, key=lambda k: results[k]["rmse_h"])
p = Predictor(f"models/l09_{best}.pt")
full = ds["test_full"]
traj = p.predict_frame(full, all_steps=True)
traj = traj.merge(add_rul(full[["unit", "time_h"]].copy()), on=["unit", "time_h"])
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, u in zip(axes, full.unit.unique()[:3]):
    g = traj[traj.unit == u]
    ax.plot(g.time_h, g.RUL, "k", label="true RUL")
    ax.plot(g.time_h, g.RUL.clip(upper=CAP), "k:", label="capped target")
    ax.plot(g.time_h, g.rul_h, "C1", lw=1, label=f"{best} prediction")
    ax.set(title=f"unit {u}", xlabel="operating hours")
axes[0].set_ylabel("RUL [h]"); axes[0].legend()
plt.tight_layout(); plt.savefig(OUT / "09_trajectories.png", dpi=110); plt.close()
print(f"\nsaved outputs/09_trajectories.png (best model: {best})")
# Read it: early on, prediction sits near the cap ("healthy"). As degradation
# becomes visible, it should track the diagonal, getting MORE accurate near EOL,
# which is exactly when accuracy matters.

# %%
# EXERCISES
# 1. Window length: train the GRU with window 24, 48, 96, 192. Plot test RMSE
#    vs window. Too short can't see trends; too long has fewer, padded samples.
# 2. Make the LSTM bidirectional (nn.LSTM(..., bidirectional=True)) in a copy of
#    RNNRegressor. Is "looking backwards" inside a past window legitimate? (Yes:
#    the whole window is already in the past. Contrast with centred smoothing.)
# 3. Remove the engineered features and use only raw sensors
#    (TrainConfig(features=["Utot","J","TinWAT","HrAIRFC","PinAIR","PoutAIR","time_h"])).
#    How much does feature engineering help a deep model with this little data?
