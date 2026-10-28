"""
LESSON 00 - Setup and orientation
=================================

Goals
  * Confirm PyTorch is installed and see which compute device you have.
  * Learn how to run these lessons.

How to run a lesson
  From the repository root:
      source .venv/bin/activate
      python lessons/00_setup.py
  Or open the file in VS Code and use "Run Cell" above each `# %%` marker
  (needs the Python + Jupyter extensions; `pip install ipykernel`). Running
  cell by cell and poking at variables is the best way to learn.

The map of the course
  Part A, PyTorch fundamentals (01-06): tensors, autograd, models, data, training,
      generalisation. These use small toy problems so the ideas are clear.
  Part B, Prognostics for PEM fuel cells (07-12): degradation data, RUL labels,
      baselines, sequence models, attention, uncertainty, tuning.
  Part C, Shipping it (13-14): checkpoints, inference, the command-line tool,
      integrating with other programs, fine-tuning on your own stacks.
"""

# %%
import platform
import sys

import torch

print("python :", sys.version.split()[0], platform.platform())
print("torch  :", torch.__version__)
print("CUDA   :", torch.cuda.is_available())
print("MPS    :", torch.backends.mps.is_available(), "(Apple-silicon GPU)")

# %% Choosing a device -------------------------------------------------------
# Tensors and models live on a device. Code is usually written device-agnostic:
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("using  :", device)

x = torch.randn(1000, 1000, device=device)
y = x @ x.T
print("matmul ok, result on", y.device, "shape", tuple(y.shape))

# Rule of thumb: for the small models in this course CPU is often just as fast,
# because moving tiny batches to a GPU has overhead. GPUs pay off when the
# work per step is large (big batches, big models).

# %%
# EXERCISES
# 1. Time `x @ x.T` on CPU vs your GPU for sizes 100, 1000, 4000.
#    (Use time.perf_counter; on GPU call torch.mps.synchronize() / torch.cuda.synchronize()
#    before reading the clock, because GPU work is asynchronous.)
# 2. Read pemfc_rul/train.py::pick_device - it's the same logic as above.
