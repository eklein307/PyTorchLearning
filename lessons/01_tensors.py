"""
LESSON 01 - Tensors
===================

Goals
  * Create, index, reshape and combine tensors.
  * Understand shape, dtype, device, and broadcasting.
  * Move between NumPy and PyTorch.

A tensor is an n-dimensional array (like a NumPy array) that can also
  (a) live on a GPU and (b) record operations for automatic differentiation (lesson 02).

The shape convention used for time series later in the course:
      [batch, time, features]    e.g. 256 windows x 96 hours x 12 sensors
Get comfortable reading shapes; most bugs in deep learning are shape bugs.
"""

# %%
import numpy as np
import torch

# %% Creating tensors -------------------------------------------------------
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])   # from Python data
z = torch.zeros(2, 3)
o = torch.ones(2, 3)
r = torch.randn(2, 3)                        # standard normal
ar = torch.arange(0, 10, 2)                  # 0,2,4,6,8
lin = torch.linspace(0, 1, 5)
print(a, a.shape, a.dtype, a.device, sep="\n")

# dtype matters: neural nets almost always use float32.
i = torch.tensor([1, 2, 3])
print(i.dtype, i.float().dtype)

# %% Indexing and slicing (same rules as NumPy) ------------------------------
x = torch.arange(24).reshape(2, 3, 4)        # [batch=2, time=3, features=4]
print("x[0]        ->", x[0].shape)          # first sample: [3, 4]
print("x[:, -1]    ->", x[:, -1].shape)      # last time step of every sample: [2, 4]
print("x[..., 0]   ->", x[..., 0].shape)     # feature 0 everywhere: [2, 3]
mask = x > 10
print("boolean mask selects", x[mask].numel(), "elements")

# %% Reshaping ---------------------------------------------------------------
flat = x.reshape(2, -1)                      # -1 = "infer this dim": [2, 12]
print(flat.shape)
print(x.transpose(1, 2).shape)               # swap time and features: [2, 4, 3]
print(x.unsqueeze(0).shape, x.unsqueeze(0).squeeze(0).shape)  # add/remove size-1 dims
# Conv1d expects [batch, channels, time]; our data is [batch, time, features],
# so you'll see x.transpose(1, 2) in pemfc_rul/models.py::CNN1D.

# %% Combining ---------------------------------------------------------------
p, q = torch.randn(3, 4), torch.randn(3, 4)
print(torch.cat([p, q], dim=0).shape)        # [6, 4]  join along an existing dim
print(torch.stack([p, q], dim=0).shape)      # [2, 3, 4] new dim

# %% Math and reductions -----------------------------------------------------
v = torch.randn(5, 3)
print(v.mean(), v.mean(dim=0).shape, v.std(dim=1).shape)
print((v @ v.T).shape)                       # matrix multiply
print(torch.exp(v).min(), v.abs().max(), v.clamp(min=0).sum())

# %% Broadcasting: the most useful (and most dangerous) rule ------------------
# Dimensions are matched from the right; size-1 dims stretch.
data = torch.randn(100, 3) * torch.tensor([1.0, 50.0, 1000.0]) + torch.tensor([0.0, 60.0, 1300.0])
mean, std = data.mean(dim=0), data.std(dim=0)     # shape [3]
z = (data - mean) / std                           # [100,3] op [3] -> [100,3]
print("standardised means ~0:", z.mean(0), "stds ~1:", z.std(0))
# ^ This is exactly feature standardisation (pemfc_rul/data.py::Standardizer).

# Danger: [N] vs [N,1] broadcasting silently creates an [N,N] matrix.
pred, target = torch.randn(4, 1), torch.randn(4)
print("oops shape:", (pred - target).shape, "<- should have been [4]")
# This exact bug shows up in regression losses. Always check shapes.

# %% NumPy interop -----------------------------------------------------------
n = np.arange(6, dtype=np.float32)
t = torch.from_numpy(n)      # shares memory with n
n[0] = 100
print(t[0])                  # 100: same memory!
back = t.numpy()
# GPU tensors must come back to CPU first: t.cpu().numpy()

# %% Devices -----------------------------------------------------------------
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
g = torch.randn(3, device=dev)
print(g.device, (g * 2).cpu())
# Operations require all tensors on the same device; mixing raises an error.

# %%
# EXERCISES
# 1. Make a tensor of shape [10, 96, 12] (10 windows, 96 hours, 12 features).
#    Extract the last hour of every window -> [10, 12]. Compute the per-feature
#    mean over time -> [10, 12]. Flatten each window -> [10, 1152].
# 2. Given voltages V of shape [1000] (one stack, hourly), compute the 24-h
#    moving average using V.unfold(0, 24, 1).mean(-1). What shape do you get, why?
# 3. Fix the broadcasting bug above two different ways (squeeze / reshape).
