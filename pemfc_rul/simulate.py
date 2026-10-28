"""
Synthetic PEM fuel cell degradation simulator.

Why a simulator?
    Real run-to-failure PEMFC data is scarce. The best-known public set (IEEE PHM
    2014 Data Challenge, FCLAB) has only two ~1000 h stacks. That's enough to test
    a model but nowhere near enough to *learn* deep learning on. This simulator
    produces a whole fleet of stacks with realistic-looking signals so you can
    practise the full workflow, then swap in real data later (see adapters.py).

What it models (simplified, semi-empirical):
    Cell voltage  V = E_ocv - eta_act - eta_ohm - eta_conc - V_reversible
        eta_act  = A * ln(j / j0)            Tafel kinetics. ECSA loss lowers j0.
        eta_ohm  = j * R                      Membrane/contact resistance grows.
        eta_conc = -B * ln(1 - j / j_lim)     Mass transport; j_lim drops as GDL ages.
        V_reversible: water management / oxide losses that build up between
                      maintenance stops and recover after each stop (the
                      "sawtooth" seen in the PHM 2014 data).
    Degradation rates are driven by operating stress: current density, load
    dynamics, temperature, humidity, plus a random per-stack and per-cell
    "manufacturing" multiplier. Degradation accelerates with age.

End of life (EOL):
    The stack voltage at *reference conditions* (j = 0.7 A/cm2, 55 C, 60 %RH)
    falls below (1 - eol_drop) * its initial value. eol_drop = 0.05 (5 %) by default;
    PHM 2014 used thresholds in the 3.5-5 % range.

Column names deliberately mirror the PHM 2014 dataset (U1..U5, Utot, J, I,
TinH2, ...), so code written against simulated data transfers to real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_CELLS = 5
AREA_CM2 = 100.0
F = 96485.0  # Faraday constant, C/mol
MOLAR_VOL_L = 22.414  # L/mol at STP

# Reference operating point used to define the health indicator / EOL
J_REF, T_REF, RH_REF = 0.70, 55.0, 60.0

# Nominal electrochemical parameters (per cell)
E0 = 1.00          # V, effective open-circuit voltage
A_TAFEL = 0.030    # V, Tafel slope term
J0_0 = 7.5e-4      # A/cm2, exchange current density when new
R0 = 0.150         # ohm*cm2, area-specific resistance when new
B_CONC = 0.050     # V, concentration-loss coefficient
JLIM_0 = 1.60      # A/cm2, limiting current density when new

SENSOR_COLUMNS = [
    "U1", "U2", "U3", "U4", "U5", "Utot", "J", "I",
    "TinH2", "ToutH2", "TinAIR", "ToutAIR", "TinWAT", "ToutWAT",
    "PinAIR", "PoutAIR", "PinH2", "PoutH2",
    "DinH2", "DinAIR", "DWAT", "HrAIRFC",
]


def cell_voltage(j, temp_c, rh, d_ecsa, d_r, d_lim, v_rev=0.0, offset=0.0):
    """Noise-free cell voltage for given conditions and damage state (vectorised)."""
    j = np.clip(j, 1e-3, None)
    j0 = J0_0 * np.exp(-d_ecsa)
    r = R0 * (1.0 + d_r) * (1.0 + 0.004 * (RH_REF - rh))
    jlim = JLIM_0 * (1.0 - d_lim)
    eta_act = A_TAFEL * np.log(j / j0)
    eta_ohm = j * r
    eta_conc = -B_CONC * np.log(np.clip(1.0 - j / jlim, 1e-3, None))
    temp_effect = 0.0005 * (temp_c - T_REF)
    return E0 - eta_act - eta_ohm - eta_conc - v_rev + temp_effect + offset


def _ar1(rng, n, phi, sigma):
    """Smooth random signal: AR(1) process, useful for slowly-wandering setpoints."""
    x = np.zeros(n)
    eps = rng.normal(0.0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def simulate_unit(unit_id, rng: np.random.Generator, eol_drop=0.05, max_hours=4000,
                  dt=1.0, missing_rate=0.001, glitch_rate=0.0005):
    """
    Simulate one stack from new until EOL (run-to-failure).

    Returns (df, meta). df has one row per dt hours; meta is a dict with the
    stack's hidden parameters and its true EOL time (useful for teaching, never
    available in real life).
    """
    n = int(max_hours / dt)
    t = np.arange(n) * dt

    # ---- Per-stack operating profile -------------------------------------
    dynamic = rng.random() < 0.5                    # static vs dynamic load (like FC1 vs FC2)
    t_set = rng.uniform(50.0, 65.0)                 # coolant setpoint, C
    rh_set = rng.uniform(40.0, 80.0)                # inlet air humidity, %
    stack_mult = rng.lognormal(0.0, 0.25)           # manufacturing variability
    cell_mult = np.clip(1.0 + 0.15 * rng.standard_normal(N_CELLS), 0.6, 1.6)
    cell_offset = 0.005 * rng.standard_normal(N_CELLS)
    accel = rng.uniform(0.3, 1.2)                   # how much degradation accelerates with age
    stop_every = rng.uniform(140.0, 220.0)          # hours between maintenance stops

    # ---- Operating conditions over time ----------------------------------
    if dynamic:
        j = J_REF + 0.10 * np.tanh(_ar1(rng, n, 0.7, 0.6))
        low_load = rng.random(n) < 0.03             # occasional low-power periods
        j[low_load] = rng.uniform(0.25, 0.45, low_load.sum())
    else:
        j = J_REF + 0.005 * rng.standard_normal(n)
    temp = t_set + _ar1(rng, n, 0.95, 0.15) + 1.5 * (j - J_REF) / 0.1
    rh = np.clip(rh_set + _ar1(rng, n, 0.98, 0.5), 20.0, 95.0)

    # ---- Degradation stress and damage accumulation ----------------------
    dj = np.abs(np.diff(j, prepend=j[0]))
    stress = (
        (j / J_REF) ** 1.5
        * np.exp(0.04 * (temp - T_REF))
        * (1.0 + 0.01 * (RH_REF - rh))
        * (1.0 + 4.0 * dj)                          # load cycling hurts
        * stack_mult
    )
    stress = np.clip(stress, 0.05, None)
    age_factor = 1.0 + accel * t / 1000.0
    base = stress * age_factor * dt
    # Per-mechanism damage per cell: shape (n, N_CELLS)
    d_ecsa = np.cumsum(base, axis=0)[:, None] * 4.5e-4 * cell_mult[None, :]
    d_r = np.cumsum(base, axis=0)[:, None] * 0.9e-4 * cell_mult[None, :]
    d_lim = np.cumsum(base, axis=0)[:, None] * 0.7e-4 * cell_mult[None, :]

    # Reversible loss: builds after each stop, resets at the next one
    stop_times = np.arange(stop_every, max_hours, stop_every)
    stop_times = stop_times + rng.normal(0, 10, len(stop_times))
    last_stop = np.zeros(n)
    for s in stop_times:
        last_stop[t >= s] = s
    v_rev = 0.004 * (1.0 - np.exp(-(t - last_stop) / 60.0))

    # ---- Health indicator at reference conditions -> EOL -----------------
    v_ref_cells = cell_voltage(J_REF, T_REF, RH_REF, d_ecsa, d_r, d_lim, v_rev[:, None], cell_offset[None, :])
    v_ref_stack = v_ref_cells.sum(axis=1)
    below = np.nonzero(v_ref_stack < (1.0 - eol_drop) * v_ref_stack[0])[0]
    n_eol = int(below[0]) if len(below) else n
    eol_time = t[n_eol - 1] if len(below) else np.nan  # NaN -> never failed (censored)

    # Truncate everything at EOL
    sl = slice(0, n_eol)
    t, j, temp, rh = t[sl], j[sl], temp[sl], rh[sl]
    d_ecsa, d_r, d_lim, v_rev = d_ecsa[sl], d_r[sl], d_lim[sl], v_rev[sl]
    m = len(t)

    # ---- Measured signals -------------------------------------------------
    cells = cell_voltage(j[:, None], temp[:, None], rh[:, None], d_ecsa, d_r, d_lim,
                         v_rev[:, None], cell_offset[None, :])
    cells = cells + rng.normal(0.0, 0.002, cells.shape)  # 2 mV sensor noise
    utot = cells.sum(axis=1) + rng.normal(0.0, 0.003, m)
    current = j * AREA_CM2

    h2_lpm = current * N_CELLS / (2 * F) * MOLAR_VOL_L * 60.0   # consumption, L/min
    air_lpm = current * N_CELLS / (4 * F) / 0.21 * MOLAR_VOL_L * 60.0
    din_h2 = 1.5 * h2_lpm + rng.normal(0, 0.03, m)
    din_air = 2.0 * air_lpm + rng.normal(0, 0.1, m)
    gdl_age = d_lim.mean(axis=1)
    p_air_in = 1300.0 + rng.normal(0, 3, m)
    p_h2_in = 1300.0 + rng.normal(0, 3, m)
    dp_air = 15.0 + 1.2 * din_air * (1.0 + 0.5 * gdl_age / 0.1)  # flow resistance grows with GDL ageing
    dp_h2 = 8.0 + 0.8 * din_h2

    df = pd.DataFrame({
        "unit": unit_id,
        "time_h": t,
        **{f"U{i + 1}": cells[:, i] for i in range(N_CELLS)},
        "Utot": utot,
        "J": j + rng.normal(0, 0.002, m),
        "I": current + rng.normal(0, 0.2, m),
        "TinH2": temp - 3.0 + rng.normal(0, 0.2, m),
        "ToutH2": temp + 0.5 + rng.normal(0, 0.2, m),
        "TinAIR": temp - 4.0 + rng.normal(0, 0.2, m),
        "ToutAIR": temp + 0.8 + rng.normal(0, 0.2, m),
        "TinWAT": temp - 1.5 + rng.normal(0, 0.1, m),
        "ToutWAT": temp + 0.5 + 0.2 * (current / 70.0) + rng.normal(0, 0.1, m),
        "PinAIR": p_air_in,
        "PoutAIR": p_air_in - dp_air + rng.normal(0, 1.5, m),
        "PinH2": p_h2_in,
        "PoutH2": p_h2_in - dp_h2 + rng.normal(0, 1.5, m),
        "DinH2": din_h2,
        "DinAIR": din_air,
        "DWAT": 2.0 + rng.normal(0, 0.02, m),
        "HrAIRFC": rh + rng.normal(0, 1.0, m),
    })

    # ---- Real-world messiness: sensor glitches and missing values ---------
    glitch = rng.random(m) < glitch_rate
    df.loc[glitch, "Utot"] -= rng.uniform(0.05, 0.2, glitch.sum())
    for col in SENSOR_COLUMNS:
        miss = rng.random(m) < missing_rate
        df.loc[miss, col] = np.nan

    meta = {
        "unit": unit_id, "dynamic_load": bool(dynamic), "T_set": t_set, "RH_set": rh_set,
        "stack_mult": stack_mult, "accel": accel, "eol_time_h": eol_time,
        "weakest_cell": int(np.argmax(cell_mult)) + 1,
    }
    return df, meta


def simulate_fleet(n_units=40, seed=0, eol_drop=0.05, max_hours=4000, start_id=1, **kwargs):
    """Simulate many run-to-failure stacks. Returns (data_df, meta_df)."""
    rng = np.random.default_rng(seed)
    frames, metas = [], []
    for k in range(n_units):
        df, meta = simulate_unit(start_id + k, rng, eol_drop=eol_drop, max_hours=max_hours, **kwargs)
        frames.append(df)
        metas.append(meta)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metas)


def truncate_fleet(df, seed=0, min_frac=0.3, max_frac=0.9):
    """
    Cut each run-to-failure history at a random point, the way you'd see a stack
    *in service*: you only have data up to "now", and the question is how much
    life is left. Returns (truncated_df, truth_df[unit, rul_true_h]).
    """
    rng = np.random.default_rng(seed)
    parts, truth = [], []
    for unit, g in df.groupby("unit", sort=False):
        life = g["time_h"].max()
        cut = rng.uniform(min_frac, max_frac) * life
        kept = g[g["time_h"] <= cut]
        parts.append(kept)
        truth.append({"unit": unit, "rul_true_h": float(life - kept["time_h"].max())})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(truth)


def standard_datasets(root="data", n_train=40, n_test=15):
    """
    The fixed datasets used throughout the lessons, created once and cached as CSV:
      train.csv       n_train run-to-failure stacks        (units 1..)
      test.csv        n_test stacks cut at a random "now"  (units 101..)
      test_truth.csv  true RUL of each test stack at its cut point
      test_full.csv   the same test stacks, full run-to-failure (for trajectory plots)
    Returns a dict of DataFrames.
    """
    from pathlib import Path
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    files = {k: root / f"{k}.csv" for k in ("train", "train_meta", "test", "test_truth", "test_full")}
    if not all(p.exists() for p in files.values()):
        train, meta = simulate_fleet(n_train, seed=0)
        full, _ = simulate_fleet(n_test, seed=123, start_id=101)
        test, truth = truncate_fleet(full, seed=123)
        for k, d in [("train", train), ("train_meta", meta), ("test", test),
                     ("test_truth", truth), ("test_full", full)]:
            d.to_csv(files[k], index=False)
    return {k: pd.read_csv(p) for k, p in files.items()}
