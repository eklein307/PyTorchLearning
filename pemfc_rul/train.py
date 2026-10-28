"""
Training engine: config -> data -> model -> training loop -> checkpoint.

The checkpoint is SELF-CONTAINED: weights + architecture + feature list +
scaler statistics + window length + target scaling. Anything needed to turn
raw sensor data into a prediction lives in that one file, so predict.py (and
any program calling the CLI) never has to guess how training was done.
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import __version__
from .data import (DEFAULT_FEATURES, Standardizer, WindowDataset, add_rul, make_windows,
                   prepare, scale_windows, split_units)
from .metrics import report
from .models import build_model, count_parameters


@dataclass
class TrainConfig:
    data: str = "data/train.csv"
    out: str = "models/model.pt"
    model: str = "lstm"
    features: list = field(default_factory=lambda: list(DEFAULT_FEATURES))
    window: int = 96            # hours of history the model sees
    stride: int = 4             # step between training windows
    rul_cap: float = 600.0      # piecewise-linear RUL target cap (hours); 0 = no cap
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1
    loss: str = "huber"         # mse | huber | gaussian
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 60
    patience: int = 10          # early stopping
    grad_clip: float = 1.0
    val_frac: float = 0.2
    seed: int = 0
    device: str = "auto"
    init_from: str = ""         # fine-tune: start from an existing checkpoint
    refit_scaler: bool = False  # when fine-tuning, re-fit feature scaling on the new data?
    freeze: str = ""            # when fine-tuning, comma-separated module prefixes to freeze (e.g. "rnn")

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def log(*args):
    """Logs go to stderr so stdout stays clean for machine-readable output."""
    print(*args, file=sys.stderr, flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loss(name: str):
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss(delta=0.1)
    if name == "gaussian":
        nll = nn.GaussianNLLLoss()
        return lambda out, y: nll(out[:, 0], y, nn.functional.softplus(out[:, 1]) + 1e-6)
    raise ValueError(f"unknown loss {name}")


def point(out):
    """Model output -> point prediction (first column)."""
    return out[:, 0]


def run_epoch(model, loader, loss_fn, device, optimizer=None, grad_clip=0.0):
    """One pass over a loader. Trains if an optimizer is given, otherwise evaluates."""
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out if out.shape[1] > 1 else out[:, 0], yb)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
    return total / max(n, 1)


@torch.no_grad()
def predict_array(model, X, device, batch_size=1024):
    model.eval()
    outs = [model(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu()
            for i in range(0, len(X), batch_size)]
    return torch.cat(outs).numpy()


def build_from_checkpoint(ckpt, device):
    model = build_model(ckpt["model_name"], **ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def train(cfg: TrainConfig, df: pd.DataFrame | None = None, verbose: bool = True) -> dict:
    """Train a model; returns a summary dict (also stored inside the checkpoint)."""
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    say = log if verbose else (lambda *a: None)

    base = torch.load(cfg.init_from, map_location="cpu", weights_only=False) if cfg.init_from else None
    if base is not None:
        # Fine-tuning must keep the architecture & input definition of the base model.
        cfg.model, cfg.features, cfg.window = base["model_name"], base["features"], base["window"]
        cfg.loss, cfg.rul_cap = base["loss"], base["rul_cap"] or 0.0
        say(f"fine-tuning from {cfg.init_from} ({cfg.model}, window={cfg.window})")

    raw = pd.read_csv(cfg.data) if df is None else df
    cap = cfg.rul_cap or None
    frame = add_rul(prepare(raw), cap=cap)
    train_u, val_u, _ = split_units(frame["unit"].unique(), val_frac=cfg.val_frac, seed=cfg.seed)
    if not train_u:  # e.g. fine-tuning on a single stack: train and validate on it
        train_u = val_u
        say("WARNING: only one unit; validating on the training stack (metrics are optimistic)")
    tr, va = frame[frame.unit.isin(train_u)], frame[frame.unit.isin(val_u)]

    Xtr, ytr, _ = make_windows(tr, cfg.features, cfg.window, cfg.stride)
    Xva, yva, _ = make_windows(va, cfg.features, cfg.window, cfg.stride)

    if base is not None and not cfg.refit_scaler:
        scaler = Standardizer.from_dict(base["scaler"])
        y_scale = base["y_scale"]
    else:
        scaler = Standardizer().fit(Xtr.reshape(-1, Xtr.shape[-1]))
        y_scale = float(cap or ytr.max())
    Xtr, Xva = scale_windows(Xtr, scaler), scale_windows(Xva, scaler)

    g = torch.Generator().manual_seed(cfg.seed)
    tr_loader = DataLoader(WindowDataset(Xtr, ytr / y_scale), batch_size=cfg.batch_size, shuffle=True, generator=g)
    va_loader = DataLoader(WindowDataset(Xva, yva / y_scale), batch_size=1024)

    out_dim = 2 if cfg.loss == "gaussian" else 1
    model_kwargs = dict(n_features=len(cfg.features), window=cfg.window, hidden=cfg.hidden,
                        layers=cfg.layers, dropout=cfg.dropout, out_dim=out_dim)
    if base is not None:
        model_kwargs = base["model_kwargs"]
        model = build_from_checkpoint(base, device)
        for prefix in filter(None, cfg.freeze.split(",")):
            for name, p in model.named_parameters():
                if name.startswith(prefix):
                    p.requires_grad_(False)
    else:
        model = build_model(cfg.model, **model_kwargs).to(device)

    say(f"device={device} model={cfg.model} params={count_parameters(model):,} "
        f"train_windows={len(Xtr):,} val_windows={len(Xva):,} units={len(train_u)}/{len(val_u)}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * len(tr_loader))
    loss_fn = make_loss(cfg.loss)

    class Stepper:  # make OneCycle step per batch without changing run_epoch
        def __init__(self, o, s): self.o, self.s = o, s
        def zero_grad(self, **kw): self.o.zero_grad(**kw)
        def step(self): self.o.step(); self.s.step()

    best, best_state, bad, history = math.inf, None, 0, []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        tl = run_epoch(model, tr_loader, loss_fn, device, Stepper(opt, sched), cfg.grad_clip)
        vl = run_epoch(model, va_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl})
        flag = ""
        if vl < best - 1e-6:
            best, best_state, bad, flag = vl, copy.deepcopy(model.state_dict()), 0, " *"
        else:
            bad += 1
        say(f"epoch {epoch:3d}  train {tl:.4f}  val {vl:.4f}{flag}")
        if bad >= cfg.patience:
            say(f"early stop (no val improvement for {cfg.patience} epochs)")
            break

    model.load_state_dict(best_state)
    pred = point(torch.from_numpy(predict_array(model, Xva, device))).numpy() * y_scale
    metrics = report(yva, pred)
    say("val metrics: " + json.dumps({k: round(v, 3) for k, v in metrics.items()}))

    ckpt = {
        "format_version": 1,
        "package_version": __version__,
        "torch_version": torch.__version__,
        "model_name": cfg.model,
        "model_kwargs": model_kwargs,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "features": cfg.features,
        "window": cfg.window,
        "rul_cap": cap,
        "y_scale": y_scale,
        "loss": cfg.loss,
        "scaler": scaler.to_dict(),
        "config": asdict(cfg),
        "val_units": [int(u) if isinstance(u, (np.integer,)) else u for u in val_u],
        "val_metrics": metrics,
        "history": history,
        "train_seconds": time.time() - t0,
    }
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, cfg.out)
    say(f"saved {cfg.out}")
    return {"checkpoint": cfg.out, "val_metrics": metrics, "epochs_run": len(history),
            "best_val_loss": best, "train_seconds": ckpt["train_seconds"]}
