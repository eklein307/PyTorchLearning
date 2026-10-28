"""
LESSON 02 - Autograd: how PyTorch computes gradients
====================================================

Goals
  * Understand what a gradient is and why training needs it.
  * Use requires_grad, .backward(), .grad, torch.no_grad().
  * Do gradient descent by hand.

Training a model = find parameters w that minimise a loss L(w).
Gradient descent: repeatedly step  w <- w - lr * dL/dw.
Computing dL/dw by hand for a million parameters is impossible, so PyTorch
records every operation on tensors that have requires_grad=True (the
"computational graph") and then walks it backwards with the chain rule. That's
backpropagation; .backward() runs it.
"""

# %%
import torch

# %% A tiny example you can check on paper -----------------------------------
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x          # dy/dx = 2x + 2 = 8 at x = 3
y.backward()
print("dy/dx =", x.grad.item())

# %% Gradients accumulate! ---------------------------------------------------
y = x ** 2 + 2 * x
y.backward()
print("after a second backward:", x.grad.item(), "(8 + 8)")
x.grad.zero_()              # this is why training loops call optimizer.zero_grad()

# %% Vectors / multiple parameters --------------------------------------------
w = torch.randn(3, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
inp = torch.tensor([1.0, 2.0, 3.0])
out = (w * inp).sum() + b    # a one-neuron linear model
loss = (out - 10.0) ** 2
loss.backward()
print("dL/dw:", w.grad, " dL/db:", b.grad)
# Check: dL/dw = 2*(out-10)*inp. Verify it yourself:
print("manual:", 2 * (out.detach() - 10) * inp)

# %% Gradient descent by hand: fit a degradation slope ------------------------
# Synthetic: stack voltage drops linearly, V = 3.30 - 0.00012 * t + noise
torch.manual_seed(0)
t = torch.linspace(0, 1000, 200)
V = 3.30 - 0.00012 * t + 0.005 * torch.randn(200)

# Scale inputs! With t in [0, 1000] the gradient for the slope is ~1000x larger
# than for the intercept, and one learning rate can't suit both. Scaling t to
# [0, 1] fixes that. (Feature scaling matters for the same reason in every model.)
ts = t / 1000.0
v0 = torch.tensor(0.0, requires_grad=True)
slope = torch.tensor(0.0, requires_grad=True)
lr = 0.5
for step in range(500):
    pred = v0 + slope * ts
    loss = ((pred - V) ** 2).mean()          # mean squared error
    loss.backward()
    with torch.no_grad():                    # updates must NOT be recorded in the graph
        v0 -= lr * v0.grad
        slope -= lr * slope.grad
    v0.grad.zero_()
    slope.grad.zero_()
    if step % 100 == 0:
        print(f"step {step:3d} loss {loss.item():.6f}")
print(f"fitted V0 = {v0.item():.4f} V, slope = {slope.item() / 1000 * 1e6:.1f} uV/h (true -120)")

# %% detach / no_grad ---------------------------------------------------------
# * torch.no_grad(): block for inference; saves memory, no graph is built.
# * tensor.detach(): same data, cut off from the graph (e.g. to log a value).
with torch.no_grad():
    p = v0 + slope * ts
print("requires_grad inside no_grad:", p.requires_grad)

# %% What can go wrong: the learning rate --------------------------------------
for lr_try in [0.01, 0.5, 1.2]:
    a = torch.tensor(0.0, requires_grad=True)
    s = torch.tensor(0.0, requires_grad=True)
    for _ in range(100):
        loss = ((a + s * ts - V) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            a -= lr_try * a.grad; s -= lr_try * s.grad
        a.grad.zero_(); s.grad.zero_()
    print(f"lr={lr_try:<5} final loss {loss.item():.3e}")
# too small -> slow; too big -> diverges. Optimisers like Adam (lesson 03) help.

# %%
# EXERCISES
# 1. Compute d/dx of sin(x)*exp(x) at x=1 with autograd and compare to the
#    analytic derivative.
# 2. Add a quadratic term (V = v0 + s*t + c*t^2) to model accelerating
#    degradation and fit it. Generate data with accelerating decay to test.
# 3. Remove the `with torch.no_grad()` in the update and see what error you get.
