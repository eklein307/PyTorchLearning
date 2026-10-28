# PyTorch & Machine Learning: from tensors to PEM fuel cell RUL prediction

A hands-on course that builds up, lesson by lesson, to a working **Remaining Useful Life (RUL)
model for PEM fuel cell stacks**. You can retrain or fine-tune the model and call it from the
command line in other programs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[lessons]"        # torch, numpy, pandas, matplotlib, ipykernel + this package
```

The lessons are Jupyter notebooks in [lessons/](lessons/). In VS Code (with the Python and Jupyter
extensions), open one, click **Select Kernel** at the top right, and pick the `.venv` interpreter.
Then run cells with Shift+Enter. In a browser instead: `pip install jupyterlab && jupyter lab`.

* Most notebooks run in seconds; 09, 10, 11 and 13 take about 1–2 minutes and 12 about 6 (they train many models).
* Run cells **top to bottom**. If things get confusing, use *Restart → Run All*.
* The first cell switches to the repository root, so `data/`, `models/` and `outputs/` (created on demand)
  are shared between notebooks and the CLI. Plots show inline and are also saved to `outputs/`.
* Run one headless: `jupyter nbconvert --to notebook --execute --inplace lessons/07_pemfc_data.ipynb`.
* Before committing to git, consider clearing outputs (*Clear All Outputs*) to keep diffs small.

## Syllabus

| # | Lesson | You learn | Runtime |
|---|---|---|---|
| **A** | **PyTorch fundamentals** | | |
| 00 | [Setup](lessons/00_setup.ipynb) | devices (CPU / CUDA / Apple MPS) | s |
| 01 | [Tensors](lessons/01_tensors.ipynb) | shapes, indexing, broadcasting, `[batch, time, features]` | s |
| 02 | [Autograd](lessons/02_autograd.ipynb) | gradients, backprop, hand-written gradient descent | s |
| 03 | [Linear regression](lessons/03_linear_regression.ipynb) | `nn.Module`, losses, SGD vs Adam, the training loop, scaling | s |
| 04 | [Neural networks](lessons/04_neural_networks.ipynb) | MLPs, activations, classification, polarisation-curve fit, extrapolation | s |
| 05 | [Data pipelines](lessons/05_data_pipeline.ipynb) | `Dataset`/`DataLoader`, splits, scaling, **leakage** | s |
| 06 | [Generalisation](lessons/06_generalization.ipynb) | overfitting, weight decay, dropout, early stopping, LR schedules | ~10 s |
| **B** | **Prognostics for PEM fuel cells** | | |
| 07 | [PEMFC data & RUL](lessons/07_pemfc_data.ipynb) | degradation, EOL, RUL labels, piecewise-linear cap, health indicators | s |
| 08 | [Baselines & metrics](lessons/08_rul_baselines.ipynb) | trend extrapolation, MLP, RMSE/MAE, asymmetric PHM score | s |
| 09 | [Sequence models](lessons/09_sequence_models.ipynb) | windows, 1D-CNN, LSTM, GRU, RUL trajectories | ~2 min |
| 10 | [Attention & Transformers](lessons/10_attention_transformers.ipynb) | attention from scratch, positional encoding, Transformer RUL | ~1 min |
| 11 | [Uncertainty](lessons/11_uncertainty.ipynb) | Gaussian NLL, MC dropout, quantile loss, calibration, maintenance decisions | ~1.5 min |
| 12 | [Tuning & experiments](lessons/12_tuning_experiments.ipynb) | seed variance, grouped K-fold, learning curves, random search, logging | ~5 min |
| **C** | **Shipping it** | | |
| 13 | [Saving & deployment](lessons/13_saving_deployment.ipynb) | checkpoints, reproducibility, `torch.export`, calling the CLI | ~40 s |
| 14 | [Final project guide](lessons/14_final_project.md) | CLI workflow, integration contract, real data, fine-tuning, extensions | — |

Each lesson ends with **exercises**. Do them; that's where most of the learning happens.

## The final tool (`pemfc_rul`)

```bash
pemfc-rul simulate --units 40 --out data/train.csv                       # synthetic fleet
pemfc-rul train    --config configs/gru_gaussian.json --out models/rul_gru.pt
pemfc-rul evaluate --model models/rul_gru.pt --data data/test.csv --truth data/test_truth.csv
pemfc-rul predict  --model models/rul_gru.pt --input history.csv --mc-samples 30   # JSON to stdout
pemfc-rul serve    --model models/rul_gru.pt         # JSON-lines over stdin/stdout for other programs
pemfc-rul train    --data data/real.csv --init-from models/rul_gru.pt --lr 2e-4 --out models/finetuned.pt
pemfc-rul tune     --data data/train.csv --trials 20
pemfc-rul info     --model models/rul_gru.pt
```

(`python -m pemfc_rul ...` is equivalent.) The integration contract (input columns, output JSON,
exit codes) is in [lesson 14](lessons/14_final_project.md#2-integration-contract); runnable integration code is in
[examples/](examples/).

## About the data

Real run-to-failure PEMFC data is scarce, so the course uses a **physics-inspired simulator**
([pemfc_rul/simulate.py](pemfc_rul/simulate.py)). It models Tafel kinetics with ECSA loss, rising
ohmic resistance, mass-transport limits, reversible losses that recover at maintenance stops,
stress from load cycling, temperature and humidity, cell-to-cell variation, sensor noise, glitches
and dropouts. Column names match the **IEEE PHM 2014 Data Challenge** (FCLAB) dataset
(`U1..U5, Utot, J, I, TinH2, ..., HrAIRFC`), and [pemfc_rul/adapters.py](pemfc_rul/adapters.py)
loads those files, or your own bench data, into the same format.

The simulator is a teaching tool, not a validated degradation model. Absolute numbers (lifetimes,
RMSE) won't carry over to your stacks. The *workflow* will: leakage-free splits, baselines,
uncertainty, honest evaluation, and fine-tuning.

## Repository layout

```
lessons/      00-13 notebooks (.ipynb) + 14 final-project guide (.md)
pemfc_rul/    the package the lessons build up to (simulate, data, models, train, predict, cli)
configs/      example training configs (JSON; CLI flags override them)
examples/     calling the model from Python / shell / other programs
data/ models/ outputs/   generated (git-ignored)
```
