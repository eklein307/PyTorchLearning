"""
LESSON 04 - Neural networks: non-linearity, classification and regression
=========================================================================

Goals
  * Build multi-layer perceptrons (MLPs) with nn.Sequential and custom modules.
  * Understand why activations (ReLU/GELU) are needed.
  * Classification with cross-entropy; regression of a non-linear function.

A stack of linear layers is still just ONE linear function. Inserting a
non-linear activation between them lets the network bend: an MLP with enough
hidden units can approximate any reasonable function.

Fuel-cell flavoured tasks:
  (a) Classify an operating state as healthy / degraded / faulty (e.g. flooding)
      from two measured quantities: a classification problem.
  (b) Learn a full non-linear polarisation curve V(j): a regression problem.
"""

# %%
import math

import torch
from torch import nn

torch.manual_seed(0)

# %% (a) Classification --------------------------------------------------------
# Two features: normalised voltage deviation and pressure-drop deviation.
# Classes form curved regions a linear model can't separate.
def make_states(n):
    x = torch.randn(n, 2)
    r = x.norm(dim=1)
    angle = torch.atan2(x[:, 1], x[:, 0])
    label = torch.zeros(n, dtype=torch.long)            # 0 healthy
    label[(r > 1.0)] = 1                                # 1 degraded (ring)
    label[(r > 1.0) & (angle > 1.0)] = 2                # 2 flooding (wedge)
    return x, label

Xtr, ytr = make_states(3000)
Xte, yte = make_states(1000)

def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()

def train_classifier(model, epochs=300, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()      # takes raw scores ("logits"), applies softmax internally
    for _ in range(epochs):
        loss = loss_fn(model(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return accuracy(model(Xte), yte)

linear = nn.Linear(2, 3)
mlp = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 3),                   # 3 logits, one per class
)
print("linear test accuracy:", round(train_classifier(linear), 3))
print("MLP    test accuracy:", round(train_classifier(mlp), 3))

# Probabilities: softmax over logits
with torch.no_grad():
    probs = torch.softmax(mlp(Xte[:3]), dim=1)
print("class probabilities for 3 samples:\n", probs.round(decimals=3))

# %% (b) Regression: a non-linear polarisation curve ---------------------------
# V(j) = E - A ln(j/j0) - j R + B ln(1 - j/jlim), per cell (same model as the simulator)
j = torch.linspace(0.02, 1.5, 400).unsqueeze(1)
V = 1.0 - 0.03 * torch.log(j / 7.5e-4) - 0.15 * j + 0.05 * torch.log(1 - j / 1.6)
V_noisy = V + 0.003 * torch.randn_like(V)

class CurveNet(nn.Module):
    """A custom module: use this style when forward() needs more than a straight pipe."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))
    def forward(self, x):
        return self.net((x - 0.75) / 0.43)    # simple fixed input scaling

net = CurveNet()
opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for step in range(2000):
    loss = nn.functional.mse_loss(net(j), V_noisy)
    opt.zero_grad(); loss.backward(); opt.step()
print(f"polarisation-curve fit RMSE: {math.sqrt(loss.item()) * 1000:.2f} mV (noise level 3 mV)")

# Extrapolation warning: networks are only trustworthy where they saw data.
with torch.no_grad():
    for jj in [0.5, 1.4, 2.0]:
        print(f"  j={jj}: net {net(torch.tensor([[jj]])).item():.3f} V")
print("  (j=2.0 is beyond the limiting current; physics says no voltage at all, the net still answers)")

# Optional plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path("outputs").mkdir(exist_ok=True)
    with torch.no_grad():
        jj = torch.linspace(0.02, 2.0, 300).unsqueeze(1)
        plt.figure(figsize=(6, 4))
        plt.scatter(j, V_noisy, s=3, alpha=0.4, label="data")
        plt.plot(jj, net(jj), "r", label="MLP")
        plt.axvline(1.5, ls="--", c="gray"); plt.xlabel("j [A/cm²]"); plt.ylabel("V_cell [V]")
        plt.legend(); plt.tight_layout(); plt.savefig("outputs/04_polarisation_fit.png", dpi=120)
    print("saved outputs/04_polarisation_fit.png")
except ImportError:
    pass

# %% model.train() vs model.eval() ---------------------------------------------
# Some layers act differently in training (Dropout, BatchNorm). Always call
# model.eval() before evaluating/predicting and model.train() before training.

# %%
# EXERCISES
# 1. Replace ReLU with GELU/Tanh in the classifier. Try hidden sizes 4, 16, 256.
# 2. Plot the decision regions of the MLP on a grid (torch.meshgrid) and save the figure.
# 3. Fit the polarisation curve with only 20 noisy points. What happens with
#    hidden=256 vs hidden=8? (Preview of overfitting, lesson 06.)
