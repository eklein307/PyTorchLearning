"""
LESSON 06 - Generalisation: overfitting, regularisation, early stopping, schedules
==================================================================================

Goals
  * See overfitting happen and read learning curves.
  * Tools against it: more data, weight decay, dropout, early stopping, smaller models.
  * Learning-rate schedules and gradient clipping.
  * Bias/variance in plain words.

Underfitting (high bias): model too simple, train AND val error are high.
Overfitting (high variance): model memorises training noise; train error keeps
falling while validation error rises. Prognostics datasets are small (tens of
stacks), so overfitting is the usual problem; the LSTM in the final project
early-stops after ~10 epochs for exactly this reason.
"""

# %%
import copy

import torch
from torch import nn

torch.manual_seed(0)

# %% A small, noisy dataset: invites overfitting --------------------------------
def make(n):
    x = torch.rand(n, 1) * 2 - 1
    y = torch.sin(3 * x) + 0.3 * torch.randn(n, 1)
    return x, y

Xtr, ytr = make(40)          # only 40 training points
Xva, yva = make(500)

def net(hidden=256, dropout=0.0):
    return nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, 1))

def train(model, epochs=3000, lr=3e-3, weight_decay=0.0, patience=None, log_every=500, label=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad, curve = float("inf"), None, 0, []
    for ep in range(epochs):
        model.train()
        loss = nn.functional.mse_loss(model(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(Xva), yva).item()
        curve.append((loss.item(), vl))
        if vl < best:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if patience and bad >= patience:
            break
    if patience:
        model.load_state_dict(best_state)   # restore the best epoch, not the last
    tl, vl = curve[-1]
    print(f"{label:<28} epochs {len(curve):4d}  train {tl:.3f}  val(last) {vl:.3f}  val(best) {best:.3f}")
    return curve

print("noise variance (the best achievable MSE) = 0.09\n")
c_over = train(net(), label="big net, no regularisation")
train(net(), weight_decay=1e-1, label="+ weight decay 0.1")
train(net(dropout=0.3), label="+ dropout 0.3")
train(net(), patience=100, label="+ early stopping")
train(net(hidden=8), label="small net (8 hidden)")

# %% Read the learning curve --------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path("outputs").mkdir(exist_ok=True)
    tr, va = zip(*c_over)
    plt.figure(figsize=(6, 4))
    plt.semilogy(tr, label="train"); plt.semilogy(va, label="validation")
    plt.axhline(0.09, ls="--", c="gray", label="noise floor")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("Overfitting")
    plt.tight_layout(); plt.savefig("outputs/06_learning_curve.png", dpi=120)
    print("saved outputs/06_learning_curve.png")
except ImportError:
    pass
# Train loss falling below the noise floor = memorising noise.

# %% Learning-rate schedules ----------------------------------------------------
# Large steps early (explore), small steps late (settle). Common choices:
#   StepLR, CosineAnnealingLR, ReduceLROnPlateau (watch val loss), OneCycleLR.
# pemfc_rul/train.py uses OneCycleLR: warm up, then anneal, stepped every batch.
model = net(64)
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-2, total_steps=100)
lrs = []
for _ in range(100):
    opt.step(); sched.step(); lrs.append(sched.get_last_lr()[0])
print("OneCycle lr at steps 0/30/60/99:", [f"{lrs[i]:.1e}" for i in (0, 30, 60, 99)])

# %% Gradient clipping ---------------------------------------------------------
# Recurrent nets can have occasional huge gradients ("exploding gradients").
# Clipping the global gradient norm keeps one bad batch from wrecking training:
#     loss.backward()
#     nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#     optimizer.step()

# %% Summary checklist when a model doesn't generalise -----------------------------
# 1. Is validation split honest (by unit, no leakage)?        -> lesson 05
# 2. Train error high too? -> underfitting: bigger model, more epochs, better features.
# 3. Train << val? -> overfitting: more data (more stacks!), augmentation,
#    weight decay, dropout, early stopping, smaller model, simpler features.
# 4. Is the noise floor itself high? Then nothing will fix it; the target is
#    inherently uncertain. Quantify that instead (lesson 11).

# %%
# EXERCISES
# 1. Increase training points from 40 to 400. How do the five configurations compare now?
# 2. Combine weight decay + dropout + early stopping. Is it better than each alone?
# 3. Replace OneCycleLR with ReduceLROnPlateau(opt, factor=0.5, patience=50)
#    stepped on validation loss inside train().
