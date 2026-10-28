"""
LESSON 03 - From hand-written gradient descent to nn.Module + optimizers
=========================================================================

Goals
  * Replace manual parameters with nn.Linear and an nn.Module.
  * Use loss functions (nn.MSELoss) and optimisers (SGD, Adam).
  * Learn THE training loop: forward -> loss -> zero_grad -> backward -> step.

Problem: predict stack voltage from operating conditions (current density,
temperature, humidity) for a healthy stack. A polarisation-curve-like
relationship, nearly linear in this narrow operating window.
"""

# %%
import torch
from torch import nn

torch.manual_seed(0)

# %% Synthetic data --------------------------------------------------------------
N = 2000
J = 0.5 + 0.4 * torch.rand(N)                 # A/cm2
T = 50 + 15 * torch.rand(N)                   # C
RH = 40 + 40 * torch.rand(N)                  # %
V = 3.9 - 0.85 * J + 0.0025 * (T - 55) + 0.0008 * (RH - 60) + 0.004 * torch.randn(N)
X = torch.stack([J, T, RH], dim=1)            # [N, 3]
y = V.unsqueeze(1)                            # [N, 1] (matches model output shape)

# Standardise inputs (lesson 01 broadcasting) - always fit stats on training data.
mu, sd = X.mean(0), X.std(0)
Xs = (X - mu) / sd
# Scale the TARGET too. Optimisers like Adam move each weight by roughly `lr` per
# step, so a bias that must travel to 3.3 V takes 3.3/lr steps. Centring y makes
# every parameter start "close". (The final project divides RUL by a y_scale.)
y_mu, y_sd = y.mean(), y.std()
ys = (y - y_mu) / y_sd

# %% Defining a model as an nn.Module ---------------------------------------------
class LinearModel(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)       # holds weight [1, n_in] and bias [1]

    def forward(self, x):                      # describe the computation; PyTorch handles gradients
        return self.linear(x)

model = LinearModel(3)
print(model)
for name, p in model.named_parameters():
    print(name, tuple(p.shape))

# %% The canonical training loop --------------------------------------------------
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200):
    pred = model(Xs)               # 1. forward
    loss = loss_fn(pred, ys)       # 2. loss (in scaled units)
    optimizer.zero_grad()          # 3. clear old gradients
    loss.backward()                # 4. backprop
    optimizer.step()               # 5. update parameters
    if epoch % 40 == 0:
        print(f"epoch {epoch:3d} MSE {loss.item():.6f}  RMSE {loss.sqrt().item() * y_sd * 1000:.2f} mV")

# Convert learned weights back to physical units to sanity-check them.
w = model.linear.weight.detach().squeeze() * y_sd / sd
print("dV/dJ  (true -0.85) :", round(w[0].item(), 4))
print("dV/dT  (true 0.0025):", round(w[1].item(), 5))
print("dV/dRH (true 0.0008):", round(w[2].item(), 5))

# %% Optimisers compared ----------------------------------------------------------
# SGD uses one learning rate for all parameters. Adam keeps running averages of
# each parameter's gradient and gradient^2, and adapts the step per parameter.
# It's the default choice for deep learning. AdamW = Adam with "decoupled" weight decay.
def fit(opt_cls, lr, steps=100, scaled=True):
    torch.manual_seed(1)
    m = LinearModel(3)
    opt = opt_cls(m.parameters(), lr=lr)
    inputs, target = (Xs, ys) if scaled else (X, y)
    for _ in range(steps):
        loss = loss_fn(m(inputs), target)
        opt.zero_grad(); loss.backward(); opt.step()
    rmse_v = loss.sqrt().item() * (y_sd.item() if scaled else 1.0)
    return f"RMSE {rmse_v * 1000:9.2f} mV"

print("SGD  lr=0.1  scaled  :", fit(torch.optim.SGD, 0.1))
print("Adam lr=0.05 scaled  :", fit(torch.optim.Adam, 0.05))
print("SGD  lr=1e-4 UNSCALED:", fit(torch.optim.SGD, 1e-4, scaled=False), " <- why scaling matters")
print("Adam lr=0.05 UNSCALED:", fit(torch.optim.Adam, 0.05, scaled=False), " <- bias can't travel to 3.3 V in 100 steps")

# %% Mini-batches ----------------------------------------------------------------
# Real datasets don't fit in one forward pass, and noisy small-batch gradients
# actually help generalisation. One "epoch" = one pass over all data in batches.
model = LinearModel(3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
batch = 64
for epoch in range(20):
    perm = torch.randperm(N)                    # shuffle every epoch
    for i in range(0, N, batch):
        idx = perm[i:i + batch]
        loss = loss_fn(model(Xs[idx]), ys[idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
with torch.no_grad():
    print(f"mini-batch Adam RMSE: {loss_fn(model(Xs), ys).sqrt().item() * y_sd.item() * 1000:.2f} mV")
# Lesson 05 replaces this manual batching with Dataset/DataLoader.

# %%
# EXERCISES
# 1. Add an interaction J*T as a 4th feature. Does the error change? Why not?
#    (The data was generated without one.) Now generate data WITH an
#    interaction term and check that the linear model can't capture it unless
#    you add the feature: the motivation for neural nets in lesson 04.
# 2. Replace nn.MSELoss with nn.L1Loss and nn.HuberLoss. Add 1% outliers to y
#    (e.g. -0.2 V glitches) and compare how much each loss is pulled by them.
