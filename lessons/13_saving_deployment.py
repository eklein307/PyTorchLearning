"""
LESSON 13 - Saving, loading and deploying models
================================================

Goals
  * state_dict vs pickling a whole model; what belongs in a checkpoint.
  * Reload and verify that predictions are bit-for-bit reproducible.
  * Inference mode, batching, latency.
  * Export options: TorchScript / torch.export for non-Python runtimes.
  * The command-line interface as an integration boundary.

Checkpoint golden rule
  Weights alone are useless. You also need everything that turns raw data
  into model input and model output back into hours:
      architecture name + constructor args, feature list (ORDER matters),
      scaler mean/std, window length, target scale / RUL cap, versions.
  pemfc_rul/train.py saves all of that in one .pt file.
"""

# %%
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pemfc_rul.models import build_model
from pemfc_rul.predict import Predictor
from pemfc_rul.simulate import standard_datasets
from pemfc_rul.train import TrainConfig, train

ds = standard_datasets("data")
CKPT = "models/l13_gru.pt"
if not Path(CKPT).exists():
    train(TrainConfig(model="gru", epochs=30, out=CKPT), verbose=False)

# %% 1. What's in the checkpoint ----------------------------------------------
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("checkpoint keys:", sorted(ck))
print("model:", ck["model_name"], ck["model_kwargs"])
print("features:", ck["features"])
print("state_dict tensors:", {k: tuple(v.shape) for k, v in list(ck["model_state"].items())[:4]}, "...")
# weights_only=False is needed because we store Python dicts/lists; only load
# checkpoints you trust (unpickling can execute code). For untrusted sources,
# store metadata as JSON next to a weights-only file (safetensors format).

# %% 2. Rebuild from scratch and verify -------------------------------------------
model = build_model(ck["model_name"], **ck["model_kwargs"])
model.load_state_dict(ck["model_state"])
model.eval()
p = Predictor(CKPT)
a = p.predict_frame(ds["test"])
b = Predictor(CKPT).predict_frame(ds["test"])
print("reload reproducible:", np.allclose(a.rul_h, b.rul_h))

# %% 3. Inference performance ---------------------------------------------------
x = torch.randn(256, ck["window"], len(ck["features"]))
with torch.inference_mode():          # like no_grad, slightly faster, stricter
    model(x)                           # warm-up
    t0 = time.perf_counter()
    for _ in range(20):
        model(x)
dt = (time.perf_counter() - t0) / 20
print(f"batch of 256 windows: {dt * 1000:.1f} ms  ({dt / 256 * 1e6:.0f} us per stack)")
# Inference is cheap. In the CLI, most time goes to Python start-up, importing
# torch (~1 s) and feature engineering. That's why `serve` exists: load once,
# answer many requests.

# %% 4. Export for other runtimes (optional) ---------------------------------------
# torch.export captures the network as a standalone graph (.pt2) that can be
# reloaded without the model's Python class, compiled ahead of time
# (AOTInductor) for C++ deployment, or converted to ONNX.
# (TorchScript / torch.jit.trace, which older tutorials use, is deprecated.)
batch = torch.export.Dim("batch", min=1, max=4096)          # allow any batch size
exported = torch.export.export(model, (x[:8],), dynamic_shapes={"x": {0: batch}})
torch.export.save(exported, "models/l13_gru_exported.pt2")
loaded = torch.export.load("models/l13_gru_exported.pt2").module()
with torch.no_grad():
    print("exported model matches:", torch.allclose(loaded(x[:33]), model(x[:33]), atol=1e-5))
# (A UserWarning about rnn._flat_weights during export is harmless: PyTorch internals.)
# Note: the exported graph contains ONLY the network. Feature engineering and
# scaling would have to be re-implemented in the host language, which is why a
# CLI around the full Python pipeline is often the pragmatic integration choice.
# ONNX route (pip install onnx onnxscript onnxruntime):
#   torch.onnx.export(model, (x[:1],), "models/model.onnx", dynamo=True)

# %% 5. Calling the CLI from another program ------------------------------------------
# Any language can do this: run a process, read JSON from stdout.
PY = sys.executable
res = subprocess.run([PY, "-m", "pemfc_rul", "predict", "--model", CKPT, "--input", "data/test.csv",
                      "--unit", "101"], capture_output=True, text=True, check=True)
print("CLI stdout:", json.loads(res.stdout))

# Long-running mode: one JSON request per line on stdin, one response per line.
proc = subprocess.Popen([PY, "-m", "pemfc_rul", "serve", "--model", CKPT],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
print("server:", proc.stdout.readline().strip())
hist = ds["test"][ds["test"].unit == 102]
req = {"id": "stack-102", "rows": json.loads(hist.to_json(orient="records"))}
proc.stdin.write(json.dumps(req) + "\n"); proc.stdin.flush()
print("response:", proc.stdout.readline().strip()[:200])
proc.stdin.write('{"cmd": "quit"}\n'); proc.stdin.flush(); proc.wait()

# %%
# EXERCISES
# 1. Save ONLY model.state_dict() with torch.save and try to predict in a new
#    script. List everything you had to re-create by hand. That's the argument
#    for self-contained checkpoints.
# 2. Change the feature ORDER in a copy of the checkpoint (ck["features"][::-1])
#    without changing the scaler. Predictions are silently wrong. Add a check to
#    Predictor that catches mismatched scaler/feature lengths.
# 3. Export the model with torch.onnx.export and run it with onnxruntime.
#    Compare its outputs with the PyTorch model on the same windows.
