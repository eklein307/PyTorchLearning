# Lesson 14: The final project, a PEMFC RUL tool you can call from anything

Everything from lessons 05–13 is packaged in [`pemfc_rul/`](../pemfc_rul/):

| Module | What it does | Lesson |
|---|---|---|
| `simulate.py` | synthetic PEMFC fleet (PHM 2014-style columns) | 07 |
| `adapters.py` | load real data (PHM 2014 FCLAB files, your own CSVs) | 07, 14 |
| `data.py` | cleaning, health-indicator features, RUL labels, scaler, windows | 05, 07, 09 |
| `models.py` | MLP, 1D-CNN, LSTM, GRU, Transformer | 04, 09, 10 |
| `metrics.py` | RMSE, MAE, PHM score, late fraction, coverage | 08, 11 |
| `train.py` | config → training loop → self-contained checkpoint; fine-tuning | 06, 09, 12 |
| `predict.py` | load checkpoint once, predict with optional uncertainty | 11, 13 |
| `cli.py` | `python -m pemfc_rul ...` (or the `pemfc-rul` command) | 13, 14 |

## 1. The workflow end to end

```bash
source .venv/bin/activate

# Data (skip if you already ran the lessons; they create these)
pemfc-rul simulate --units 40 --seed 0 --out data/train.csv
pemfc-rul simulate --units 15 --seed 123 --start-id 101 --truncate \
          --truth-out data/test_truth.csv --out data/test.csv

# Train with the default config, or a config file + overrides
pemfc-rul train --config configs/gru_gaussian.json --out models/rul_gru.pt
pemfc-rul train --model lstm --window 144 --epochs 80 --out models/rul_lstm.pt

# See exactly what config will be used, without training
pemfc-rul train --config configs/gru_gaussian.json --hidden 128 --dump-config

# Evaluate on in-service stacks (one prediction each, vs known truth)
pemfc-rul evaluate --model models/rul_gru.pt --data data/test.csv --truth data/test_truth.csv --mc-samples 30

# Predict
pemfc-rul predict --model models/rul_gru.pt --input data/test.csv             # JSON, latest point per unit
pemfc-rul predict --model models/rul_gru.pt --input data/test.csv --format csv --out preds.csv
pemfc-rul predict --model models/rul_gru.pt --input data/test.csv --unit 103 --all-steps
cat history.csv | pemfc-rul predict --model models/rul_gru.pt                # stdin

pemfc-rul info --model models/rul_gru.pt
pemfc-rul tune --data data/train.csv --trials 20 --epochs 25 --out-dir models/tune
pemfc-rul train --config models/tune/best_config.json --epochs 80 --out models/best.pt
```

## 2. Integration contract

* **Input**: a CSV with one row per time step. Required columns: `time_h`, `U1`–`U5`,
  `Utot`, `J`, `TinWAT`, `ToutWAT`, `HrAIRFC`, `PinAIR`, `PoutAIR`, `PinH2`, `PoutH2`
  (plus `unit` if there is more than one stack). Send the **whole history from beginning of life**:
  the features are relative to each stack's first 24 h.
* **Output** (stdout, JSON list, one object per stack):

  ```json
  {"unit": 101, "time_h": 572.0, "rul_h": 241.8, "rul_std_h": 119.7,
   "rul_p05_h": 44.8, "rul_p95_h": 438.8, "history_h": 572.0, "at_cap": false}
  ```

  `rul_std_h`/`rul_p05_h`/`rul_p95_h` appear when the model has a Gaussian head or `--mc-samples > 1`.
  `at_cap: true` means "healthy: at least about `rul_cap` hours left", not an exact number.
* **Logs** go to stderr. **Exit code** is 0 on success and 1 on bad input or a missing file.
* **Many calls?** Use `serve`: start it once, write one JSON request per line, read one JSON response per line.
  See [examples/](../examples/) for Python, shell and other-language patterns.

## 3. Moving to real fuel cell data

1. **Get data into the standard columns.**
   * IEEE PHM 2014 Data Challenge (FCLAB FC1/FC2): `adapters.load_phm2014("FC1_Ageing.csv", unit=1)`
     strips units from headers (`"Utot (V)"` → `Utot`) and resamples to 1 h.
   * Your bench: `adapters.load_generic("bench.csv", mapping={"StackV": "Utot", ...})`.
   Then save it with `df.to_csv("data/real.csv", index=False)`.
2. **Decide your EOL definition.** It must be the same for labels and for the decision it supports.
   For a run-to-failure history, `add_rul` treats the last row as EOL. If your stack was stopped
   before failure, cut the history at the first time the (smoothed, reference-condition) voltage
   crosses your threshold, or treat it as *censored* and leave it out of training labels.
3. **Very few real stacks? Pre-train on simulation, fine-tune on real data.** That's transfer learning:

   ```bash
   # tune simulate.py parameters so the synthetic fleet resembles your stacks (lifetimes, voltage levels)
   pemfc-rul simulate --units 200 --seed 1 --eol-drop 0.04 --out data/sim_big.csv
   pemfc-rul train --data data/sim_big.csv --model gru --loss gaussian --epochs 60 --out models/pretrained.pt

   # fine-tune: same architecture/features/window, lower LR, optionally freeze the encoder
   pemfc-rul train --data data/real.csv --init-from models/pretrained.pt \
                   --lr 2e-4 --epochs 30 --freeze rnn --out models/finetuned.pt
   #   --refit-scaler   re-fit feature scaling on the real data (useful if sensors differ in offset)
   #   --val-frac 0.5   with only 2 stacks, validate on one and train on the other
   ```

   Parameter-name prefixes for `--freeze`: `rnn` (LSTM/GRU), `body` (MLP/CNN), `proj,encoder` (Transformer).
   The head always stays trainable.
4. **Validate honestly.** With real data, hold out whole stacks. Report RMSE *and* late fraction *and*
   interval coverage. A model that is confidently late is the dangerous one.

## 4. Ideas to extend it (your next lessons)

* **Physics-informed features**: fit a polarisation-curve model to periodic characterisation
  data (ECSA, R_ohm, j_lim estimates over time) and feed the parameters as features.
* **Operating-condition normalisation**: regress voltage on (J, T, RH) for the first 100 h of
  each stack, then use the residual as the health indicator.
* **Forecasting formulation**: predict the future voltage trajectory (seq2seq) and derive RUL from
  the threshold crossing, which gives interpretable outputs at any EOL threshold.
* **Deep ensembles** for better-calibrated uncertainty (lesson 11, exercise 1).
* **Censored data**: stacks that never reached EOL still say "RUL ≥ x". Survival losses
  (e.g. Weibull negative log-likelihood with censoring) use that information.
* **Other datasets for practice**: NASA C-MAPSS turbofan (the classic RUL benchmark;
  same code works after column mapping, with cycles instead of hours), NASA/CALCE battery ageing.

## 5. Final checklist before trusting a model on real stacks

- [ ] Split by stack; scaler fit on training stacks only
- [ ] Features use only past and present data (no centred windows on the most recent point)
- [ ] Beats the trend-extrapolation and fleet-average baselines (lesson 08)
- [ ] Seed spread measured; improvements bigger than the spread (lesson 12)
- [ ] Uncertainty coverage checked on held-out stacks (lesson 11)
- [ ] Checkpoint reproduces predictions after reload; `info` shows the expected features (lesson 13)
- [ ] Operating envelope of deployment data is inside the training envelope (load profile,
      temperature, humidity). Neural nets extrapolate badly (lesson 04).
