"""
Inference: load a checkpoint once, then predict RUL for any number of stacks.

Uncertainty, when available, comes from two sources and is combined:
  * aleatoric  - the Gaussian head (model trained with loss="gaussian")
  * epistemic  - MC dropout: keep dropout ON at inference, run N passes, look
                 at how much the predictions disagree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import Standardizer, make_windows, prepare, scale_windows
from .train import build_from_checkpoint, pick_device


class Predictor:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = pick_device(device)
        self.ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = build_from_checkpoint(self.ckpt, self.device).eval()
        self.scaler = Standardizer.from_dict(self.ckpt["scaler"])
        self.features = self.ckpt["features"]
        self.window = self.ckpt["window"]
        self.y_scale = self.ckpt["y_scale"]
        self.gaussian = self.ckpt["model_kwargs"]["out_dim"] == 2

    def info(self) -> dict:
        c = self.ckpt
        return {k: c[k] for k in ("model_name", "model_kwargs", "features", "window", "rul_cap",
                                  "loss", "val_metrics", "package_version", "torch_version")}

    def _forward(self, X: np.ndarray, mc_samples: int):
        xb = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            if mc_samples <= 1:
                outs = self.model(xb).unsqueeze(0)
            else:
                for m in self.model.modules():  # dropout on, everything else (e.g. BatchNorm) in eval mode
                    if isinstance(m, nn.Dropout):
                        m.train()
                outs = torch.stack([self.model(xb) for _ in range(mc_samples)])
                self.model.eval()
        outs = outs.cpu().numpy()                               # [S, N, out_dim]
        means = outs[..., 0] * self.y_scale                     # [S, N]
        mean = means.mean(0)
        var = means.var(0) if mc_samples > 1 else np.zeros_like(mean)
        if self.gaussian:
            ale = (np.log1p(np.exp(outs[..., 1])) + 1e-6) * self.y_scale ** 2
            var = var + ale.mean(0)
        return mean, np.sqrt(var)

    def predict_frame(self, raw: pd.DataFrame, all_steps: bool = False, mc_samples: int = 0,
                      batch_size: int = 2048) -> pd.DataFrame:
        """
        raw: sensor DataFrame with a 'unit' column (added if missing) and the
             full history of each stack (needed for beginning-of-life baselines).
        Returns one row per unit (latest time) or per time step if all_steps.
        """
        raw = raw.copy()
        if "unit" not in raw:
            raw["unit"] = 1
        frame = prepare(raw)
        X, _, idx = make_windows(frame, self.features, self.window, stride=1, target=None,
                                 last_only=not all_steps)
        X = scale_windows(X, self.scaler)
        means, stds = [], []
        for i in range(0, len(X), batch_size):
            m, s = self._forward(X[i:i + batch_size], mc_samples)
            means.append(m)
            stds.append(s)
        out = idx.copy()
        out["rul_h"] = np.clip(np.concatenate(means), 0.0, None)
        std = np.concatenate(stds)
        if self.gaussian or mc_samples > 1:
            out["rul_std_h"] = std
            out["rul_p05_h"] = np.clip(out["rul_h"] - 1.645 * std, 0.0, None)
            out["rul_p95_h"] = out["rul_h"] + 1.645 * std
        out["history_h"] = out["unit"].map(frame.groupby("unit")["time_h"].max())
        if self.ckpt["rul_cap"]:
            # Predictions near the cap mean "healthy, at least this much life left".
            out["at_cap"] = out["rul_h"] >= 0.95 * self.ckpt["rul_cap"]
        return out
