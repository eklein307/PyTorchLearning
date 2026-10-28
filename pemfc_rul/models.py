"""
Model zoo for windowed RUL regression. Every model maps
    x: [batch, window, n_features]  ->  [batch, out_dim]
out_dim = 1 for a point estimate, 2 for (mean, raw variance) with Gaussian NLL.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class MLP(nn.Module):
    """Flatten the window and use a plain feed-forward net. A strong-enough baseline."""

    def __init__(self, n_features, window, hidden=128, layers=2, dropout=0.1, out_dim=1):
        super().__init__()
        dims = [n_features * window] + [hidden] * layers
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(nn.Flatten(), *blocks)
        self.head = nn.Linear(dims[-1], out_dim)

    def forward(self, x):
        return self.head(self.body(x))


class CNN1D(nn.Module):
    """Temporal convolutions: learn local patterns (slopes, sawtooth recoveries)."""

    def __init__(self, n_features, window, hidden=64, layers=3, dropout=0.1, out_dim=1, kernel=5):
        super().__init__()
        blocks, c_in = [], n_features
        for i in range(layers):
            blocks += [
                nn.Conv1d(c_in, hidden, kernel, padding=kernel // 2, dilation=1),
                nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            ]
            c_in = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden * 2, out_dim)

    def forward(self, x):
        h = self.body(x.transpose(1, 2))           # Conv1d wants [B, C, T]
        pooled = torch.cat([h.mean(-1), h[..., -1]], dim=-1)  # global avg + last step
        return self.head(pooled)


class RNNRegressor(nn.Module):
    """LSTM or GRU over the window; predict from the final hidden state."""

    def __init__(self, n_features, window, hidden=64, layers=2, dropout=0.1, out_dim=1, cell="lstm"):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(n_features, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1]))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class TransformerRegressor(nn.Module):
    """Self-attention encoder: every time step can look at every other one."""

    def __init__(self, n_features, window, hidden=64, layers=2, dropout=0.1, out_dim=1, heads=4):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden)
        self.pos = PositionalEncoding(hidden, max_len=max(window, 16))
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=hidden * 2, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, x):
        h = self.encoder(self.pos(self.proj(x)))
        return self.head(h[:, -1])


MODELS = {
    "mlp": MLP,
    "cnn": CNN1D,
    "lstm": lambda **kw: RNNRegressor(cell="lstm", **kw),
    "gru": lambda **kw: RNNRegressor(cell="gru", **kw),
    "transformer": TransformerRegressor,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}', choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
