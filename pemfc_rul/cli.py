"""
Command-line interface:  python -m pemfc_rul <command> [options]

Design rules that make it easy to call from other programs:
  * stdout carries ONLY the result (JSON by default, or CSV); logs go to stderr.
  * `-` as a path means stdin/stdout, so it works in pipes.
  * Exit codes: 0 ok, 1 bad input, 2 usage error (argparse).
  * `serve` keeps the model loaded and answers one JSON request per line, for
    programs that call it many times (avoids paying model-load time per call).

Commands:
  simulate   generate synthetic PEMFC run-to-failure data
  train      train (or fine-tune) a model, writes a checkpoint .pt
  evaluate   score a checkpoint on labelled data
  predict    predict RUL for one or more stacks from sensor history
  serve      JSON-lines request/response loop over stdin/stdout
  tune       random hyper-parameter search
  info       print checkpoint metadata
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict, fields

import numpy as np
import pandas as pd


def _read_csv(path):
    return pd.read_csv(sys.stdin if path == "-" else path)


def _emit(obj, fmt="json", out="-"):
    if isinstance(obj, pd.DataFrame):
        text = obj.to_csv(index=False) if fmt == "csv" else json.dumps(_records(obj), indent=2)
    else:
        text = json.dumps(obj, indent=2, default=_json_default)
    if out == "-":
        sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))
    else:
        with open(out, "w") as f:
            f.write(text)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _records(df):
    return json.loads(df.to_json(orient="records"))


# ---------------------------------------------------------------- commands

def cmd_simulate(a):
    from .simulate import simulate_fleet, truncate_fleet
    df, meta = simulate_fleet(a.units, seed=a.seed, eol_drop=a.eol_drop, start_id=a.start_id)
    if a.truncate:
        df, truth = truncate_fleet(df, seed=a.seed)
        if a.truth_out:
            truth.to_csv(a.truth_out, index=False)
    if a.meta_out:
        meta.to_csv(a.meta_out, index=False)
    if a.out == "-":
        df.to_csv(sys.stdout, index=False)
    else:
        df.to_csv(a.out, index=False)
        print(f"wrote {len(df):,} rows / {df.unit.nunique()} units to {a.out}", file=sys.stderr)


def _config_from_args(a):
    from .train import TrainConfig
    cfg = TrainConfig.from_json(a.config) if a.config else TrainConfig()
    for f in fields(TrainConfig):
        v = getattr(a, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    if isinstance(cfg.features, str):
        cfg.features = [s.strip() for s in cfg.features.split(",") if s.strip()]
    return cfg


def cmd_train(a):
    from .train import train
    cfg = _config_from_args(a)
    if a.dump_config:
        _emit(asdict(cfg))
        return
    _emit(train(cfg))


def cmd_evaluate(a):
    from .data import add_rul
    from .metrics import report
    from .predict import Predictor
    p = Predictor(a.model, device=a.device)
    raw = _read_csv(a.data)
    if a.truth:
        # Truncated (in-service) data: score the latest point of each unit vs known RUL
        pred = p.predict_frame(raw, mc_samples=a.mc_samples)
        truth = pd.read_csv(a.truth)
        m = pred.merge(truth, on="unit")
        y_true = m["rul_true_h"].to_numpy()
    else:
        # Run-to-failure data: score every time step, against the (capped) label
        pred = p.predict_frame(raw, all_steps=True, mc_samples=a.mc_samples)
        lab = add_rul(raw[["unit", "time_h"]].copy(), cap=None if a.no_cap else p.ckpt["rul_cap"])
        m = pred.merge(lab, on=["unit", "time_h"])
        y_true = m["RUL"].to_numpy()
    std = m["rul_std_h"].to_numpy() if "rul_std_h" in m else None
    result = report(y_true, m["rul_h"].to_numpy(), std)
    if a.per_unit:
        m["error_h"] = m["rul_h"] - y_true
        result["per_unit"] = _records(m.groupby("unit")["error_h"].agg(["mean", "std", "count"]).reset_index())
    _emit(result)


def cmd_predict(a):
    from .predict import Predictor
    p = Predictor(a.model, device=a.device)
    raw = _read_csv(a.input)
    if a.unit is not None:
        raw = raw[raw["unit"].astype(str) == str(a.unit)]
        if raw.empty:
            print(f"error: no rows for unit {a.unit}", file=sys.stderr)
            sys.exit(1)
    _emit(p.predict_frame(raw, all_steps=a.all_steps, mc_samples=a.mc_samples), fmt=a.format, out=a.out)


def cmd_serve(a):
    """
    One JSON object per input line:
        {"id": "anything", "rows": [{"time_h": 0, "U1": 0.65, ...}, ...]}
        {"id": "x", "csv": "path/to/history.csv"}
        {"cmd": "info"}   {"cmd": "quit"}
    One JSON object per output line: {"id": ..., "ok": true, "predictions": [...]}
    """
    from .predict import Predictor
    p = Predictor(a.model, device=a.device)
    print(json.dumps({"ok": True, "ready": True, "model": p.ckpt["model_name"]}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = {}
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            if req.get("cmd") == "info":
                resp = {"ok": True, "info": p.info()}
            else:
                raw = pd.DataFrame(req["rows"]) if "rows" in req else pd.read_csv(req["csv"])
                pred = p.predict_frame(raw, all_steps=req.get("all_steps", False),
                                       mc_samples=int(req.get("mc_samples", a.mc_samples)))
                resp = {"ok": True, "predictions": _records(pred)}
        except Exception as e:  # never die on one bad request
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        resp["id"] = req.get("id") if isinstance(req, dict) else None
        print(json.dumps(resp, default=_json_default), flush=True)


def cmd_tune(a):
    from .train import TrainConfig, train
    rng = np.random.default_rng(a.seed)
    space = {
        "model": a.models.split(","),
        "window": [48, 72, 96, 144],
        "hidden": [32, 64, 128],
        "layers": [1, 2, 3],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "lr": [3e-4, 1e-3, 3e-3],
        "weight_decay": [0.0, 1e-4, 1e-3],
    }
    base = TrainConfig.from_json(a.config) if a.config else TrainConfig()
    df = pd.read_csv(a.data)
    results = []
    for trial in range(a.trials):
        cfg = TrainConfig(**asdict(base))
        choice = {k: v[rng.integers(len(v))] for k, v in space.items()}
        for k, v in choice.items():
            setattr(cfg, k, v.item() if hasattr(v, "item") else v)
        cfg.data, cfg.epochs, cfg.out = a.data, a.epochs, f"{a.out_dir}/trial_{trial:03d}.pt"
        print(f"--- trial {trial}: {choice}", file=sys.stderr)
        r = train(cfg, df=df, verbose=a.verbose)
        results.append({"trial": trial, **{k: getattr(cfg, k) for k in space}, **r["val_metrics"],
                        "checkpoint": cfg.out})
        print(f"    val rmse {r['val_metrics']['rmse_h']:.1f} h", file=sys.stderr)
    table = pd.DataFrame(results).sort_values(a.metric)
    table.to_csv(f"{a.out_dir}/tune_results.csv", index=False)
    best = table.iloc[0].to_dict()
    best_cfg = asdict(base)
    best_cfg.update({k: best[k] for k in space})
    best_cfg["data"] = a.data
    with open(f"{a.out_dir}/best_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2, default=_json_default)
    _emit({"best": best, "results_csv": f"{a.out_dir}/tune_results.csv",
           "best_config": f"{a.out_dir}/best_config.json"})


def cmd_info(a):
    from .predict import Predictor
    _emit(Predictor(a.model).info())


# ---------------------------------------------------------------- parser

def build_parser():
    ap = argparse.ArgumentParser(prog="pemfc_rul", description="PEM fuel cell RUL prediction")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="generate synthetic run-to-failure data")
    s.add_argument("--units", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--start-id", type=int, default=1)
    s.add_argument("--eol-drop", type=float, default=0.05, help="EOL voltage drop fraction")
    s.add_argument("--truncate", action="store_true", help="cut histories at random 'now' (test set)")
    s.add_argument("--truth-out", help="with --truncate: CSV of true RUL per unit")
    s.add_argument("--meta-out", help="CSV of hidden per-unit simulation parameters")
    s.add_argument("--out", default="-")
    s.set_defaults(func=cmd_simulate)

    t = sub.add_parser("train", help="train or fine-tune a model")
    t.add_argument("--config", help="JSON config file; CLI flags override it")
    t.add_argument("--data")
    t.add_argument("--out")
    t.add_argument("--model", choices=["mlp", "cnn", "lstm", "gru", "transformer"])
    t.add_argument("--features", help="comma-separated feature names")
    for name, typ in [("window", int), ("stride", int), ("rul_cap", float), ("hidden", int),
                      ("layers", int), ("dropout", float), ("lr", float), ("weight_decay", float),
                      ("batch_size", int), ("epochs", int), ("patience", int), ("val_frac", float),
                      ("seed", int), ("grad_clip", float)]:
        t.add_argument("--" + name.replace("_", "-"), dest=name, type=typ)
    t.add_argument("--loss", choices=["mse", "huber", "gaussian"])
    t.add_argument("--device")
    t.add_argument("--init-from", dest="init_from", help="checkpoint to fine-tune from")
    t.add_argument("--refit-scaler", dest="refit_scaler", action="store_true", default=None)
    t.add_argument("--freeze", help="comma-separated parameter-name prefixes to freeze")
    t.add_argument("--dump-config", action="store_true", help="print resolved config and exit")
    t.set_defaults(func=cmd_train)

    for name, fn, hlp in [("evaluate", cmd_evaluate, "score a model on labelled data"),
                          ("predict", cmd_predict, "predict RUL from sensor history"),
                          ("serve", cmd_serve, "JSON-lines server over stdin/stdout"),
                          ("info", cmd_info, "show checkpoint metadata")]:
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--model", required=True, help="checkpoint .pt")
        p.set_defaults(func=fn)
        if name == "info":
            continue
        p.add_argument("--device", default="cpu")
        p.add_argument("--mc-samples", type=int, default=0, help="MC-dropout passes for uncertainty")
        if name == "evaluate":
            p.add_argument("--data", required=True)
            p.add_argument("--truth", help="CSV unit,rul_true_h for truncated data")
            p.add_argument("--no-cap", action="store_true", help="score against uncapped RUL")
            p.add_argument("--per-unit", action="store_true")
        if name == "predict":
            p.add_argument("--input", default="-", help="CSV with sensor history ('-' = stdin)")
            p.add_argument("--unit", help="only this unit")
            p.add_argument("--all-steps", action="store_true", help="predict at every time step")
            p.add_argument("--format", choices=["json", "csv"], default="json")
            p.add_argument("--out", default="-")

    u = sub.add_parser("tune", help="random hyper-parameter search")
    u.add_argument("--data", required=True)
    u.add_argument("--config")
    u.add_argument("--trials", type=int, default=10)
    u.add_argument("--epochs", type=int, default=25)
    u.add_argument("--models", default="cnn,lstm,gru,transformer")
    u.add_argument("--metric", default="rmse_h")
    u.add_argument("--out-dir", default="models/tune")
    u.add_argument("--seed", type=int, default=0)
    u.add_argument("--verbose", action="store_true")
    u.set_defaults(func=cmd_tune)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if getattr(a, "out_dir", None):
        import os
        os.makedirs(a.out_dir, exist_ok=True)
    try:
        a.func(a)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
