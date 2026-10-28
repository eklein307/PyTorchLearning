"""
LESSON 05 - Data pipelines: Dataset, DataLoader, splits, scaling, leakage
=========================================================================

Goals
  * Write a torch Dataset and use a DataLoader for batching & shuffling.
  * Split data into train / validation / test, and know what each is for.
  * Scale features using TRAINING statistics only.
  * Recognise data leakage, the #1 cause of "my model was great until production".

Roles of the splits
  train: the model learns parameters from it.
  validation: you make decisions with it (architecture, epochs, learning rate).
  test: touched ONCE at the end to report honest performance.
  If you tune on the test set, it silently becomes a validation set and your
  reported number is optimistic.
"""

# %%
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

torch.manual_seed(0)

# %% A custom Dataset -----------------------------------------------------------
# A Dataset only needs __len__ and __getitem__. The DataLoader does batching,
# shuffling and (optionally) parallel loading with num_workers.
class OperatingPointDataset(Dataset):
    def __init__(self, n=5000):
        J = 0.5 + 0.4 * torch.rand(n)
        T = 50 + 15 * torch.rand(n)
        RH = 40 + 40 * torch.rand(n)
        self.X = torch.stack([J, T, RH], 1)
        self.y = 3.9 - 0.85 * J + 0.0025 * (T - 55) - 0.3 * (J - 0.7) ** 2 * (RH < 50) + 0.004 * torch.randn(n)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

ds = OperatingPointDataset()
print("one sample:", ds[0])

train_ds, val_ds, test_ds = random_split(ds, [3500, 750, 750], generator=torch.Generator().manual_seed(0))
loader = DataLoader(train_ds, batch_size=128, shuffle=True)
xb, yb = next(iter(loader))
print("batch shapes:", xb.shape, yb.shape)

# %% Scaling with training statistics only ---------------------------------------
Xtr = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
mu, sd = Xtr.mean(0), Xtr.std(0)
# A Dataset wrapper that applies the transform on the fly:
class Scaled(Dataset):
    def __init__(self, base, mu, sd): self.base, self.mu, self.sd = base, mu, sd
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return (x - self.mu) / self.sd, y

train_dl = DataLoader(Scaled(train_ds, mu, sd), batch_size=128, shuffle=True)
val_dl = DataLoader(Scaled(val_ds, mu, sd), batch_size=1024)
test_dl = DataLoader(Scaled(test_ds, mu, sd), batch_size=1024)

# %% A reusable train/evaluate pair ---------------------------------------------
# (pemfc_rul/train.py::run_epoch is this same function, slightly extended)
def run_epoch(model, dl, loss_fn, optimizer=None):
    model.train(optimizer is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for xb, yb in dl:
            loss = loss_fn(model(xb).squeeze(-1), yb)
            if optimizer:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(xb); n += len(xb)
    return total / n

model = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for epoch in range(30):
    tl = run_epoch(model, train_dl, nn.MSELoss(), opt)
    vl = run_epoch(model, val_dl, nn.MSELoss())
    if epoch % 5 == 0:
        print(f"epoch {epoch:2d}  train RMSE {tl ** .5 * 1000:6.2f} mV  val RMSE {vl ** .5 * 1000:6.2f} mV")
print(f"TEST RMSE (reported once): {run_epoch(model, test_dl, nn.MSELoss()) ** .5 * 1000:.2f} mV")

# %% LEAKAGE: the time-series trap ----------------------------------------------
# With degradation data, rows from the same stack are strongly correlated and
# ordered in time. If you split ROWS at random, training sees hour 500 and 502
# of a stack while validation gets hour 501. The model can memorise each stack
# and look brilliant.
#
# Demonstration: 20 "stacks", each with its own hidden offset. Target depends on
# the offset; features include a stack fingerprint (here, the initial voltage).
rng = np.random.default_rng(0)
rows = []
for unit in range(20):
    life = rng.uniform(500, 1500)
    v0 = rng.normal(3.25, 0.03)
    for t in np.arange(0, life, 5):
        rul = life - t
        v = v0 - 0.15 * (t / life) ** 2 + rng.normal(0, 0.004)
        rows.append((unit, t, v, v0, rul))
arr = np.array(rows, dtype=np.float32)
units = arr[:, 0]
X = torch.tensor(arr[:, 1:4]); y = torch.tensor(arr[:, 4])
X = (X - X.mean(0)) / X.std(0)

def fit_score(train_mask):
    m = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    o = torch.optim.Adam(m.parameters(), lr=3e-3)
    dl = DataLoader(TensorDataset(X[train_mask], y[train_mask] / 1000), batch_size=128, shuffle=True)
    for _ in range(60):
        run_epoch(m, dl, nn.MSELoss(), o)
    with torch.no_grad():
        p = m(X[~train_mask]).squeeze(-1) * 1000
    return ((p - y[~train_mask]) ** 2).mean().sqrt().item()

row_mask = torch.tensor(rng.random(len(y)) < 0.8)
unit_mask = torch.tensor(np.isin(units, rng.choice(20, 16, replace=False)))
print(f"random ROW split  RMSE: {fit_score(row_mask):6.1f} h   <- leaky, optimistic")
print(f"split by UNIT     RMSE: {fit_score(unit_mask):6.1f} h   <- honest: new, unseen stacks")
# pemfc_rul/data.py::split_units always splits by unit for this reason.
# Other leakage sources to watch for:
#   * scaling with statistics computed on all data (incl. test)
#   * features computed with "future" information (centred rolling windows, full-life normalisation)
#   * duplicate or near-duplicate rows across splits

# %%
# EXERCISES
# 1. Set num_workers=2 in the DataLoader. Is it faster here? (Probably not:
#    the data is in memory; workers help when loading/decoding from disk.)
# 2. In the leakage demo, remove the v0 feature. Does the gap between the row
#    split and the unit split shrink? Why?
# 3. The rolling median in pemfc_rul/data.py::clean uses center=True. That uses
#    4 future samples. Is that leakage at prediction time? (Hint: at the last
#    time step the window just shrinks; think about what the model saw in training
#    vs what it sees live. Small, but real. How would you remove it?)
