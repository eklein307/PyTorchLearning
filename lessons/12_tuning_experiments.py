"""
LESSON 12 - Tuning, cross-validation and running experiments properly
=====================================================================

Goals
  * Measure run-to-run variance (random seeds) before believing any comparison.
  * Grouped K-fold cross-validation by stack.
  * Learning curve vs number of stacks: is more data or more tuning the bottleneck?
  * Random hyper-parameter search (the CLI `tune` command does this).
  * Log experiments so you can compare them later.

Key mindset: with ~40 stacks every number has a big error bar. A change that
improves RMSE by 5 h means nothing if seeds alone move it by 15 h.

Runtime: a few minutes. Reduce EPOCHS if you're impatient.
"""

# %%
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from pemfc_rul.data import add_rul, prepare
from pemfc_rul.metrics import report
from pemfc_rul.predict import Predictor
from pemfc_rul.simulate import simulate_fleet, standard_datasets
from pemfc_rul.train import TrainConfig, train

ds = standard_datasets("data")
EPOCHS = 25
LOG = Path("outputs/experiments.jsonl"); LOG.parent.mkdir(exist_ok=True)

def test_score(ckpt):
    pred = Predictor(ckpt).predict_frame(ds["test"]).merge(ds["test_truth"], on="unit")
    return report(pred.rul_true_h, pred.rul_h)

def run(tag, cfg, df=None):
    """Train + evaluate + append one line to the experiment log."""
    t0 = time.time()
    r = train(cfg, df=df, verbose=False)
    t = test_score(cfg.out)
    rec = {"tag": tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "config": asdict(cfg),
           "val": r["val_metrics"], "test": t, "seconds": round(time.time() - t0, 1)}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec

# %% 1. Seed variance ------------------------------------------------------------
rows = []
for seed in range(3):
    cfg = TrainConfig(model="gru", epochs=EPOCHS, seed=seed, out=f"models/l12_seed{seed}.pt")
    rec = run("seed", cfg)
    rows.append({"seed": seed, "val_rmse": rec["val"]["rmse_h"], "test_rmse": rec["test"]["rmse_h"]})
    print(rows[-1])
df = pd.DataFrame(rows)
print(f"seed spread: val RMSE {df.val_rmse.mean():.1f} +/- {df.val_rmse.std():.1f} h, "
      f"test RMSE {df.test_rmse.mean():.1f} +/- {df.test_rmse.std():.1f} h")
# Note the seed changes BOTH the weight init AND which stacks land in validation
# (split_units uses it). Both are real sources of variance.

# %% 2. Grouped K-fold cross-validation ---------------------------------------------
# Every stack is used for validation exactly once -> a more stable estimate than
# a single split. Expensive (K trainings), so use it for final decisions.
K = 4
train_df = ds["train"]
units = np.array(sorted(train_df.unit.unique()))
rng = np.random.default_rng(0); rng.shuffle(units)
folds = np.array_split(units, K)
cv = []
for k, val_units in enumerate(folds):
    # Trick: reuse train() by putting the validation fold last, then give it val_frac.
    # Simpler & explicit: train on the other folds, evaluate on this fold ourselves.
    tr = train_df[~train_df.unit.isin(val_units)]
    cfg = TrainConfig(model="gru", epochs=EPOCHS, out=f"models/l12_fold{k}.pt", val_frac=0.15)
    train(cfg, df=tr, verbose=False)
    p = Predictor(cfg.out)
    va = train_df[train_df.unit.isin(val_units)]
    pred = p.predict_frame(va, all_steps=True).merge(add_rul(va[["unit", "time_h"]].copy(), cap=cfg.rul_cap),
                                                     on=["unit", "time_h"])
    cv.append(report(pred.RUL, pred.rul_h)["rmse_h"])
    print(f"fold {k}: held-out stacks {val_units.tolist()} RMSE {cv[-1]:.1f} h")
print(f"{K}-fold CV RMSE: {np.mean(cv):.1f} +/- {np.std(cv):.1f} h")

# %% 3. Learning curve: does more data help? ------------------------------------
extra, _ = simulate_fleet(60, seed=999, start_id=1001)          # 60 more stacks
pool = pd.concat([train_df, extra], ignore_index=True)
pool_units = pool.unit.unique()
curve = []
for n in [10, 20, 40, 100]:
    sub = pool[pool.unit.isin(pool_units[:n])]
    rec = run(f"datasize_{n}", TrainConfig(model="gru", epochs=EPOCHS, out=f"models/l12_n{n}.pt"), df=sub)
    curve.append((n, rec["test"]["rmse_h"]))
    print(f"{n:4d} stacks -> test RMSE {curve[-1][1]:.1f} h")
# If the curve is still dropping at the right end, collecting more stacks beats
# any amount of hyper-parameter tuning. In practice: share data across test
# benches, include partial (not-yet-failed) histories, use simulated data for
# pre-training (lesson 14: fine-tuning).

# %% 4. Random search -------------------------------------------------------------
# Random search beats grid search when only a few hyper-parameters really
# matter (Bergstra & Bengio, 2012). Use VALIDATION metrics to choose; report test once.
space = {"model": ["gru", "lstm", "cnn"], "window": [48, 96, 144], "hidden": [32, 64, 128],
         "dropout": [0.0, 0.1, 0.3], "lr": [3e-4, 1e-3, 3e-3]}
rng = np.random.default_rng(1)
trials = []
for i in range(4):
    choice = {k: v[rng.integers(len(v))] for k, v in space.items()}
    choice = {k: (v.item() if hasattr(v, "item") else v) for k, v in choice.items()}
    rec = run("search", TrainConfig(epochs=EPOCHS, out=f"models/l12_search{i}.pt", **choice))
    trials.append({**choice, "val_rmse": rec["val"]["rmse_h"], "test_rmse": rec["test"]["rmse_h"]})
    print(trials[-1])
print(pd.DataFrame(trials).sort_values("val_rmse").to_string(index=False))
# Same thing from the command line (writes best_config.json you can train from):
#   python -m pemfc_rul tune --data data/train.csv --trials 20 --epochs 25 --out-dir models/tune

# %% 5. Reading your experiment log ---------------------------------------------
log = pd.read_json(LOG, lines=True)
flat = pd.json_normalize(log.to_dict("records"))
print(flat[["tag", "config.model", "config.window", "val.rmse_h", "test.rmse_h", "seconds"]].tail(10))
# Tools like MLflow, Weights & Biases or TensorBoard do this with dashboards; a
# JSON-lines file is enough to start and has no dependencies.

# %%
# EXERCISES
# 1. Re-run section 1 with 5 seeds for both "gru" and "cnn". Is the difference
#    between them larger than the seed spread? (A t-test is a good start.)
# 2. Add rul_cap in {300, 450, 600, 900} to the search space. Careful: RMSE vs a
#    capped target isn't comparable across caps. Compare on UNCAPPED test truth.
# 3. Plot the learning curve from section 3 with error bars (3 seeds per size).
