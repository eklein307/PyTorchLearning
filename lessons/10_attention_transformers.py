"""
LESSON 10 - Attention and Transformers for time series
======================================================

Goals
  * Implement scaled dot-product attention from scratch and check it against PyTorch.
  * Understand queries/keys/values, multi-head attention, positional encoding.
  * Train the TransformerRegressor on PEMFC data and compare with the RNNs.

The idea
  An RNN passes information step by step. Attention lets every time step look
  DIRECTLY at every other step and decide how much each matters:
      Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V
  Q (queries): what each step is looking for
  K (keys):    what each step offers
  V (values):  the content that gets mixed
  The softmax weights form a [T, T] matrix: row i says how much step i draws
  from each step j.

  Attention has no notion of order by itself (shuffle the steps, the output
  shuffles the same way), so we ADD a positional encoding to the inputs.

Honest note
  Transformers shine with lots of data. With ~30 stacks they're usually not
  better than a GRU, and they need more care (warm-up, pre-norm, dropout). They're
  worth knowing because they're the dominant architecture in modern ML.
"""

# %%
import math

import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(0)

# %% Attention from scratch ----------------------------------------------------
def attention(q, k, v, mask=None):
    """q, k, v: [batch, T, d]. Returns output [batch, T, d] and weights [batch, T, T]."""
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)    # [B, T, T]: similarity of every pair
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    w = scores.softmax(dim=-1)                          # each row sums to 1
    return w @ v, w

B, T, D = 2, 6, 8
x = torch.randn(B, T, D)
Wq, Wk, Wv = (nn.Linear(D, D, bias=False) for _ in range(3))
out, w = attention(Wq(x), Wk(x), Wv(x))
print("output", tuple(out.shape), "weights", tuple(w.shape), "row sums", w[0].sum(-1))

ref = F.scaled_dot_product_attention(Wq(x), Wk(x), Wv(x))   # fused, fast version
print("matches PyTorch:", torch.allclose(out, ref, atol=1e-6))

# Causal mask: step i may only look at steps <= i (needed for forecasting /
# language models, where the "future" of the window must stay hidden).
causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
_, wc = attention(Wq(x), Wk(x), Wv(x), mask=causal)
print("causal weights (row = query step):\n", wc[0].detach().round(decimals=2))

# %% A toy lookup task: attention you can interpret -------------------------------
# Input has 2 channels: channel 0 is a marker (1.0 at ONE random step, else 0),
# channel 1 is random values. Target: the channel-1 value at the marked step.
# That's a content-based lookup ("find the step that looks like X, read its
# value"), which is exactly what one attention layer does, and the learned
# attention weights tell us WHERE the model looked.
def toy_batch(n, T=64):
    xs = torch.zeros(n, T, 2)
    xs[..., 1] = torch.randn(n, T)
    pos = torch.randint(0, T, (n,))
    xs[torch.arange(n), pos, 0] = 1.0
    return xs, xs[torch.arange(n), pos, 1], pos

class TinyAttn(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.inp = nn.Linear(2, d)
        self.query = nn.Parameter(torch.randn(1, 1, d))   # learned "what am I looking for"
        self.mha = nn.MultiheadAttention(d, num_heads=1, batch_first=True)
        self.out = nn.Linear(d, 1)
    def forward(self, x, return_weights=False):
        h = self.inp(x)
        o, w = self.mha(self.query.expand(len(x), -1, -1), h, h)   # w: [B, 1, T]
        y = self.out(o[:, 0]).squeeze(-1)
        return (y, w[:, 0]) if return_weights else y

class TinyGRU(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.gru = nn.GRU(2, d, batch_first=True)
        self.out = nn.Linear(d, 1)
    def forward(self, x):
        return self.out(self.gru(x)[0][:, -1]).squeeze(-1)

for name, m in [("GRU", TinyGRU()), ("attention", TinyAttn())]:
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for step in range(600):
        xb, yb, _ = toy_batch(128)
        loss = F.mse_loss(m(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    xt, yt, pt = toy_batch(1000)
    with torch.no_grad():
        print(f"toy task {name:10s} test MSE {F.mse_loss(m(xt), yt).item():.3f}  (guess-zero MSE ~1.0)")
attn_model = m
with torch.no_grad():
    _, w = attn_model(xt, return_weights=True)
print(f"attention peaks on the marked step in {(w.argmax(-1) == pt).float().mean():.0%} of samples")
print(f"average weight on the marked step: {w[torch.arange(len(pt)), pt].mean():.2f} (uniform would be {1/64:.3f})")
# Both can learn this. The difference: the GRU's reasoning is hidden in its state,
# while attention weights are directly inspectable. With long sequences, RNNs also
# struggle to carry information across hundreds of steps; attention reaches any
# step in one hop.

# %% Transformer on the PEMFC data -----------------------------------------------
# pemfc_rul/models.py::TransformerRegressor = Linear projection + sinusoidal
# positional encoding + nn.TransformerEncoder (pre-norm) + head on the last step.
from pemfc_rul.predict import Predictor
from pemfc_rul.metrics import report
from pemfc_rul.simulate import standard_datasets
from pemfc_rul.train import TrainConfig, train

ds = standard_datasets("data")
cfg = TrainConfig(data="data/train.csv", out="models/l10_transformer.pt", model="transformer",
                  hidden=64, layers=2, dropout=0.1, lr=5e-4, epochs=40, patience=8)  # device auto: GPU if available
train(cfg, verbose=False)
p = Predictor(cfg.out)
pred = p.predict_frame(ds["test"]).merge(ds["test_truth"], on="unit")
r = report(pred.rul_true_h, pred.rul_h)
print(f"\ntransformer: val RMSE {p.ckpt['val_metrics']['rmse_h']:.1f} h | TEST RMSE {r['rmse_h']:.1f} h "
      f"PHM {r['phm_score']:.2f}  (compare with lesson 09's results)")

# %%
# EXERCISES
# 1. Plot the attention weights w[i] for a few samples with the marker position.
# 2. Change the target to the value 3 steps AFTER the marker. The single-layer
#    model now fails. Why? (It needs to know ORDER: add a positional embedding,
#    nn.Parameter(torch.randn(1, T, d)), to h and a self-attention layer before
#    the query. That's why Transformers need positional encodings.)
# 3. Train the PEMFC transformer with heads=1 vs 4, layers 1 vs 4. Compare test RMSE
#    and training time. Which architecture would you deploy given ~30 stacks?
