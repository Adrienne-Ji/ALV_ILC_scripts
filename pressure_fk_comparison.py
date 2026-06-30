"""
pressure_fk_comparison.py
─────────────────────────────────────────────────────────────────────────────
Trains three pressure-conditioned FK model architectures and uses a gradient-
based optimiser (MPC strategy from PressureFK_MPC.py) with each to generate
actuator trajectories for a desired deformation path.  Results are compared
across architectures and against the desired trajectory.

Architectures
─────────────
  1. DataDriven NN  — plain MLP: [epi, trans, endo, pressure] → [twist, height, vol]
  2. PINN (PCNN)    — PressureConditionedFK: actuator branch + pressure-conditioning branch
  3. SINDy          — sparse polynomial regression on 4-D standardised input

Workflow
────────
  A  Load multi-pressure experimental data (6 pressure levels, ~300 pts each)
  B  Train / fit all three FK models
  C  FK assessment: RMSE on held-out test set (all pressures combined)
  D  Optimiser-based IK (MPC) using each FK model as surrogate
  E  FK verification: plug optimised actuators back through each FK model
  F  Comparison figures and summary table

Data files required (relative to sharedCSVs/)
───────────────────────────────────────────────
  {0,30,60,90,120,150}mmhg_300pt_10ml.csv
  engineered_data_withP.csv   (target trajectory with time-varying pressure)

Run from any directory — all paths are resolved from this file's location.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
PYTHONCODES = os.path.join(BASE, "..")
SHARED      = os.path.join(BASE, "..", "..", "sharedCSVs")

PINN_WEIGHTS = os.path.join(PYTHONCODES, "trained_pressure_fk_model.pth")
ENG_CSV      = os.path.join(PYTHONCODES, "engineered_data_withP.csv")

# ── Configuration ─────────────────────────────────────────────────────────────
PRESSURES         = [0, 30, 60, 90, 120, 150]   # mmHg
COLS              = ['epi', 'trans', 'endo', 'dtwist_deg', 'height_mm', 'volume_endo_mL', 'rBase_mm', 'pressure_meas']
OUTPUT_NAMES      = ['Twist_deg', 'Height_mm', 'Volume_mL']
OUTPUT_UNITS      = ['deg', 'mm', 'mL']
MOTOR_LABELS      = ['Epi', 'Trans', 'Endo']

LOAD_PINN_WEIGHTS = False   # True only if weights were trained with this exact script

POLY_DEGREE       = 3       # SINDy polynomial library degree
SINDY_THRESHOLD   = 0.05    # STLSQ sparsity threshold
SINDY_ALPHA       = 0.05    # STLSQ L2 regularisation

# MPC optimiser settings (shared across all architectures)
MPC_LR          = 1e-2     # Adam learning rate for NN models
LAMBDA_REG      = 0.001    # actuator regularisation (towards 0.5)  ↓ reduced
LAMBDA_SMOOTH   = 0.0      # smoothness penalty — set 0 to diagnose convergence floor
N_STEPS_FIRST   = 2000     # steps for first time-point (cold start)  ↑ increased
N_STEPS_WARM    = 1000     # steps for subsequent time-points (warm start)  ↑ increased

# Physical actuator range limits [epi, trans, endo] in mm
ACT_PHYS_LO = np.array([200., 202., 200.])
ACT_PHYS_HI = np.array([248., 248., 248.])

ARCH_COLORS  = {'DataDriven': '#d62728', 'PINN': '#2ca02c',
                'SINDy': '#1f77b4',     'SINDy2': '#9467bd',
                'pSINDy': '#e377c2'}
ARCH_LINES   = {'DataDriven': '-',       'PINN': '--',
                'SINDy': ':',            'SINDy2': '-.',
                'pSINDy': (0,(3,1,1,1))}

print("=" * 68)
print("  PRESSURE-CONDITIONED FK COMPARISON: DataDriven | PINN | SINDy | SINDy-2Stage")
print("=" * 68)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Multi-pressure data loading and normalisation
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1] Loading multi-pressure data …")

data_dict = {}
for p in PRESSURES:
    fpath = os.path.join(SHARED, f"deformation_by_actuator_V_adjusted_{p}mmhg.csv")
    df = pd.read_csv(fpath).dropna()
    df['nominal_pressure'] = p
    data_dict[p] = df
    print(f"    {p:3d} mmHg → {len(df)} rows")

x_list, y_list = [], []
for p in PRESSURES:
    df = data_dict[p]
    x_list.append(df[['epi', 'trans', 'endo', 'nominal_pressure']].values)
    y_list.append(df[['dtwist_deg', 'height_mm', 'volume_endo_mL']].values)

x_all = np.vstack(x_list).astype(float)
y_all = np.vstack(y_list).astype(float)
print(f"  Combined: {len(x_all)} rows")

# 70 / 20 / 10 split  (matches PressureFK_MPC.py)
x_tr, x_tmp, y_tr, y_tmp = train_test_split(
    x_all, y_all, test_size=0.30, random_state=42, shuffle=True)
x_te, x_va, y_te, y_va = train_test_split(
    x_tmp, y_tmp, test_size=1 / 3, random_state=42, shuffle=True)
print(f"  Train: {len(x_tr)}  |  Val: {len(x_va)}  |  Test: {len(x_te)}")

# Min-max normalisation computed from train split only
x_min = x_tr.min(axis=0);  x_max = x_tr.max(axis=0)
y_min = y_tr.min(axis=0);  y_max = y_tr.max(axis=0)
x_den = np.where(x_max - x_min > 1e-12, x_max - x_min, 1e-12)
y_den = np.where(y_max - y_min > 1e-12, y_max - y_min, 1e-12)

# Normalised actuator bounds (clamp MPC search to training-data actuator range)
ACT_LO_N = np.clip((ACT_PHYS_LO - x_min[:3]) / x_den[:3], 0., 1.)
ACT_HI_N = np.clip((ACT_PHYS_HI - x_min[:3]) / x_den[:3], 0., 1.)
ACT_BOUNDS_LBFGS = [(lo, hi) for lo, hi in zip(ACT_LO_N, ACT_HI_N)]
print(f"  Actuator bounds (normalised): lo={ACT_LO_N.round(3)}  hi={ACT_HI_N.round(3)}")

def norm_x(x):  return (np.asarray(x) - x_min) / x_den
def norm_y(y):  return (np.asarray(y) - y_min) / y_den
def denorm_x(xn): return xn * x_den[:3] + x_min[:3]
def denorm_y(yn): return np.asarray(yn) * y_den + y_min

x_tr_n = norm_x(x_tr);  x_va_n = norm_x(x_va);  x_te_n = norm_x(x_te)
y_tr_n = norm_y(y_tr);  y_va_n = norm_y(y_va);  y_te_n = norm_y(y_te)

# ── Engineered target signal ──────────────────────────────────────────────────
print("\n[1b] Loading engineered target signal …")
eng_data      = pd.read_csv(ENG_CSV).to_numpy()
time_vec      = eng_data[:, 0]
desired_p     = eng_data[:, 1]                     # mmHg
twist_tgt     = eng_data[:, 2]                     # deg
height_tgt    = eng_data[:, 3] + 70               # mm  (matches PressureFK_MPC.py)
volume_tgt    = eng_data[:, 4] + 15      # mL  (matches PressureFK_MPC.py)

tgt_phys = np.stack([twist_tgt, height_tgt, volume_tgt], axis=1)  # (N, 3) physical

# Normalise targets and pressure into model input / output space
targets_norm = np.stack([
    (twist_tgt  - y_min[0]) / y_den[0],
    (height_tgt - y_min[1]) / y_den[1],
    (volume_tgt - y_min[2]) / y_den[2],
], axis=1)                                         # (N, 3) in [0,1] approx
pressure_n = (desired_p - x_min[3]) / x_den[3]   # (N,)   in [0,1]
n_targets  = len(targets_norm)
print(f"  {n_targets} time-steps  |  pressure: [{desired_p.min():.0f}, {desired_p.max():.0f}] mmHg")

# ── Normalisation out-of-bounds check ─────────────────────────────────────────
# Targets outside [0,1] in normalised space = model is extrapolating there.
oob = (targets_norm < 0) | (targets_norm > 1)
print(f"\n  Normalisation coverage check (fraction of trajectory outside [0,1]):")
for i, col in enumerate(OUTPUT_NAMES):
    n_oob = oob[:, i].sum()
    lo = targets_norm[:, i].min(); hi = targets_norm[:, i].max()
    print(f"    {col:<14}: {n_oob:3d}/{n_targets} ({n_oob/n_targets*100:5.1f}% OOB)  "
          f"norm range=[{lo:.3f}, {hi:.3f}]")
if oob.any():
    print("  WARNING: MPC targets outside training range — model will extrapolate there.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2A — DataDriven NN FK
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2A] Training DataDriven NN FK model …")
t0 = time.time()

class DataDrivenFK(nn.Module):
    """Plain MLP: 4-in → 3-out, all pressures concatenated."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 3))
    def forward(self, x): return self.net(x)

dd_model = DataDrivenFK()
dd_opt   = optim.Adam(dd_model.parameters(), lr=1e-3, weight_decay=1e-5)
dd_sch   = optim.lr_scheduler.ReduceLROnPlateau(dd_opt, factor=0.5, patience=10)
dd_crit  = nn.MSELoss()

x_tr_t  = torch.tensor(x_tr_n, dtype=torch.float32)
y_tr_t  = torch.tensor(y_tr_n, dtype=torch.float32)
x_va_t  = torch.tensor(x_va_n, dtype=torch.float32)
y_va_t  = torch.tensor(y_va_n, dtype=torch.float32)
x_te_t  = torch.tensor(x_te_n, dtype=torch.float32)
y_te_t  = torch.tensor(y_te_n, dtype=torch.float32)

tr_ds   = TensorDataset(x_tr_t, y_tr_t)
tr_ld   = DataLoader(tr_ds, batch_size=32, shuffle=True)

best_dd, best_dd_sd, dd_ctr = float('inf'), None, 0
for ep in range(10_000):
    dd_model.train()
    for xb, yb in tr_ld:
        dd_opt.zero_grad()
        dd_crit(dd_model(xb), yb).backward()
        dd_opt.step()
    dd_model.eval()
    with torch.no_grad():
        vl = dd_crit(dd_model(x_va_t), y_va_t).item()
    dd_sch.step(vl)
    if vl < best_dd - 1e-5:
        best_dd, best_dd_sd, dd_ctr = vl, dd_model.state_dict(), 0
    else:
        dd_ctr += 1
        if dd_ctr >= 30: break
    if ep % 500 == 0:
        print(f"    [DD] ep {ep:5d}  val={vl:.6f}")

if best_dd_sd: dd_model.load_state_dict(best_dd_sd)
dd_model.eval()
print(f"  DataDriven FK ready  ({time.time() - t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2B — PINN FK (PressureConditionedFK)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2B] PINN FK (PressureConditionedFK) …")
t0 = time.time()

class PressureConditionedFK(nn.Module):
    """
    Two-branch architecture — embeds the physics prior that pressure
    multiplicatively scales and additively shifts the actuator-driven deformations.
    Architecture identical to PressureFK_MPC.py so saved weights are compatible.
    """
    def __init__(self, hidden_act=256, hidden_p=1024):
        super().__init__()
        self.act_branch = nn.Sequential(
            nn.Linear(3, hidden_act), nn.Tanh(),
            nn.Linear(hidden_act, hidden_act), nn.Tanh(),
            nn.Linear(hidden_act, 3))
        self.pressure_branch = nn.Sequential(
            nn.Linear(1, hidden_p), nn.Tanh(),
            nn.Linear(hidden_p, hidden_p), nn.Tanh(),
            nn.Linear(hidden_p, hidden_p), nn.Tanh(),
            nn.Linear(hidden_p, hidden_p), nn.Tanh(),
            nn.Linear(hidden_p, 6))

    def forward(self, actuators, pressure):
        y_nom  = self.act_branch(actuators)
        cond   = self.pressure_branch(pressure)
        # cond indices match PressureFK_MPC.py ordering exactly
        twist  = torch.sigmoid(cond[:, 0]) * y_nom[:, 0] + cond[:, 1]
        height = torch.sigmoid(cond[:, 3]) * y_nom[:, 1] + cond[:, 2]
        volume = torch.sigmoid(cond[:, 5]) * y_nom[:, 2] + cond[:, 4]
        return torch.stack([twist, height, volume], dim=1)

pinn_model = PressureConditionedFK()

if LOAD_PINN_WEIGHTS and os.path.exists(PINN_WEIGHTS):
    pinn_model.load_state_dict(torch.load(PINN_WEIGHTS, weights_only=True))
    pinn_model.eval()
    print(f"  Loaded PINN weights from {os.path.basename(PINN_WEIGHTS)}")
    print("  (Set LOAD_PINN_WEIGHTS=False to retrain from scratch)")
else:
    print("  Training PINN from scratch (Stage 1: 0 mmHg actuator branch) …")
    df0  = data_dict[0]
    X0   = df0[['epi', 'trans', 'endo']].values.astype(float)
    y0   = df0[['dtwist_deg', 'height_mm', 'volume_endo_mL']].values.astype(float)
    X0_n = norm_x(np.hstack([X0, np.zeros((len(X0), 1))])[:, :])[:, :3]
    y0_n = norm_y(y0)
    X0_tr, X0_va, y0_tr, y0_va = train_test_split(X0_n, y0_n, test_size=0.2, random_state=42)
    X0_tr_t = torch.tensor(X0_tr, dtype=torch.float32)
    y0_tr_t = torch.tensor(y0_tr, dtype=torch.float32)
    X0_va_t = torch.tensor(X0_va, dtype=torch.float32)
    y0_va_t = torch.tensor(y0_va, dtype=torch.float32)
    s1_opt = optim.Adam(pinn_model.parameters(), lr=1e-4)
    s1_crit = nn.MSELoss()
    best_s1, best_s1_sd, s1_ctr = float('inf'), None, 0
    for ep in range(15_000):
        pinn_model.train()
        s1_opt.zero_grad()
        loss = s1_crit(pinn_model.act_branch(X0_tr_t), y0_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), 1.0)
        s1_opt.step()
        pinn_model.eval()
        with torch.no_grad():
            vl = s1_crit(pinn_model.act_branch(X0_va_t), y0_va_t).item()
        if vl < best_s1:
            best_s1, best_s1_sd, s1_ctr = vl, pinn_model.state_dict(), 0
        else:
            s1_ctr += 1
            if s1_ctr >= 20: break
        if ep % 500 == 0: print(f"    [PINN s1] ep {ep:5d}  val={vl:.6f}")
    if best_s1_sd: pinn_model.load_state_dict(best_s1_sd)

    print("  Stage 2: training on all pressures …")
    Xtr_act = torch.tensor(x_tr_n[:, :3], dtype=torch.float32)
    Xtr_p   = torch.tensor(x_tr_n[:, 3:], dtype=torch.float32)
    ytr_t   = torch.tensor(y_tr_n, dtype=torch.float32)
    Xva_act = torch.tensor(x_va_n[:, :3], dtype=torch.float32)
    Xva_p   = torch.tensor(x_va_n[:, 3:], dtype=torch.float32)
    yva_t2  = torch.tensor(y_va_n, dtype=torch.float32)
    s2_opt  = optim.AdamW(pinn_model.parameters(), lr=1e-4, weight_decay=1e-2)
    s2_sch  = optim.lr_scheduler.ReduceLROnPlateau(s2_opt, factor=0.5, patience=3)
    s2_crit = nn.MSELoss()
    best_s2, best_s2_sd, s2_ctr = float('inf'), None, 0
    for ep in range(5_000):
        pinn_model.train()
        s2_opt.zero_grad()
        loss = s2_crit(pinn_model(Xtr_act, Xtr_p), ytr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), 1.0)
        s2_opt.step()
        pinn_model.eval()
        with torch.no_grad():
            vl = s2_crit(pinn_model(Xva_act, Xva_p), yva_t2).item()
        s2_sch.step(vl)
        if (best_s2 - vl) > 1e-4:
            best_s2, best_s2_sd, s2_ctr = vl, pinn_model.state_dict(), 0
        else:
            s2_ctr += 1
            if s2_ctr >= 20: break
        if ep % 500 == 0: print(f"    [PINN s2] ep {ep:5d}  val={vl:.6f}")
    if best_s2_sd: pinn_model.load_state_dict(best_s2_sd)
    torch.save(pinn_model.state_dict(), PINN_WEIGHTS)
    print(f"  PINN weights saved → {os.path.basename(PINN_WEIGHTS)}")

pinn_model.eval()
for param in pinn_model.parameters():
    param.requires_grad_(False)
print(f"  PINN FK ready  ({time.time() - t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2C — SINDy FK (sparse polynomial regression)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[2C] Fitting SINDy FK model  (degree={POLY_DEGREE}) …")
t0 = time.time()

# Standardise inputs so STLSQ threshold is scale-independent
sindy_sc = StandardScaler()
sindy_sc.fit(x_tr)
poly_lib = PolynomialLibrary(degree=POLY_DEGREE, include_bias=True)
poly_lib.fit(sindy_sc.transform(x_tr))
Theta_tr = np.asarray(poly_lib.transform(sindy_sc.transform(x_tr)))
Theta_te = np.asarray(poly_lib.transform(sindy_sc.transform(x_te)))
feat_names = poly_lib.get_feature_names_out(['epi', 'trans', 'endo', 'pressure'])

sindy_results = {}
print(f"\n  {'Output':<18}  {'R² (test)':>10}  {'RMSE':>10}  {'Active terms':>13}")
for i, col in enumerate(OUTPUT_NAMES):
    sy = StandardScaler()
    y_s = sy.fit_transform(y_tr[:, i:i+1]).ravel()
    opt = STLSQ(threshold=SINDY_THRESHOLD, alpha=SINDY_ALPHA, max_iter=100)
    opt.fit(Theta_tr, y_s)
    coef = np.asarray(opt.coef_).ravel()
    y_pred = sy.inverse_transform((Theta_te @ coef).reshape(-1, 1)).ravel()
    r2   = r2_score(y_te[:, i], y_pred)
    rmse = np.sqrt(mean_squared_error(y_te[:, i], y_pred))
    active = int(np.sum(np.abs(coef) > 1e-12))
    sindy_results[col] = dict(coef=coef, sy=sy, r2=r2, rmse=rmse)
    print(f"  {col:<18}  {r2:>10.4f}  {rmse:>10.4f}  {active:>13}")

print(f"\n  SINDy FK ready  ({time.time() - t0:.1f}s)")

def sindy_predict(x_physical):
    """x_physical: (N,4) raw [epi,trans,endo,pressure] → (N,3) physical outputs."""
    x_s   = sindy_sc.transform(np.asarray(x_physical).reshape(-1, 4))
    theta = np.asarray(poly_lib.transform(x_s))
    outs  = []
    for col in OUTPUT_NAMES:
        r = sindy_results[col]
        outs.append(r['sy'].inverse_transform((theta @ r['coef']).reshape(-1, 1)).ravel())
    return np.stack(outs, axis=1)   # (N, 3)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2D — SINDy-2Stage FK
#
# Stage 1: sparse polynomial [epi,trans,endo] → [twist,height,volume]
#           trained on 0 mmHg data only.
# Stage 2: sparse polynomial [epi,trans,endo,pressure] → residuals
#           trained on all pressure data.
# Prediction: y = stage1(act) + stage2(act, pressure)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[2D] Fitting SINDy-2Stage FK model …")
t0 = time.time()

# ── Stage 1: 0 mmHg baseline (3-feature input) ───────────────────────────────
df0        = data_dict[0]
X0_phys    = df0[['epi', 'trans', 'endo']].values.astype(float)
y0_phys    = df0[['dtwist_deg', 'height_mm', 'volume_endo_mL']].values.astype(float)

s2_sc_in1  = StandardScaler().fit(X0_phys)
poly_lib1  = PolynomialLibrary(degree=POLY_DEGREE, include_bias=True)
poly_lib1.fit(s2_sc_in1.transform(X0_phys))
Theta0     = np.asarray(poly_lib1.transform(s2_sc_in1.transform(X0_phys)))

s2_stage1  = {}
print(f"\n  Stage 1 (0 mmHg):")
print(f"  {'Output':<18}  {'R²':>8}  {'RMSE':>10}  {'Active':>8}")
for i, col in enumerate(OUTPUT_NAMES):
    sy1  = StandardScaler().fit(y0_phys[:, i:i+1])
    y_s  = sy1.transform(y0_phys[:, i:i+1]).ravel()
    opt1 = STLSQ(threshold=SINDY_THRESHOLD, alpha=SINDY_ALPHA, max_iter=100)
    opt1.fit(Theta0, y_s)
    coef1 = np.asarray(opt1.coef_).ravel()
    y_pred1 = sy1.inverse_transform((Theta0 @ coef1).reshape(-1, 1)).ravel()
    r2_1    = r2_score(y0_phys[:, i], y_pred1)
    rmse_1  = np.sqrt(mean_squared_error(y0_phys[:, i], y_pred1))
    s2_stage1[col] = dict(coef=coef1, sy=sy1)
    print(f"  {col:<18}  {r2_1:>8.4f}  {rmse_1:>10.4f}  {int(np.sum(np.abs(coef1)>1e-12)):>8}")

def sindy2_stage1(act_phys):
    """act_phys: (N,3) [epi,trans,endo] → (N,3) baseline physical outputs."""
    x_s   = s2_sc_in1.transform(np.asarray(act_phys).reshape(-1, 3))
    theta = np.asarray(poly_lib1.transform(x_s))
    outs  = []
    for col in OUTPUT_NAMES:
        r = s2_stage1[col]
        outs.append(r['sy'].inverse_transform((theta @ r['coef']).reshape(-1, 1)).ravel())
    return np.stack(outs, axis=1)

# ── Stage 2: multiplicative pressure conditioning  (mirrors PINN structure) ───
#
# Basis: [base×p^0, base×p^1, ..., base×p^K, p^0, p^1, ..., p^K]
# Fit:   y ≈ Σ a_k × base × p^k  +  Σ b_k × p^k
#            └── scale terms ──┘     └── offset terms ──┘
# This is linear regression — consistent with SINDy philosophy but
# structurally equivalent to PINN's sigmoid(scale)×base + offset.
# ──────────────────────────────────────────────────────────────────────────────
x_tr_phys = x_tr[:, :3]
x_te_phys = x_te[:, :3]
p_tr      = x_tr[:, 3]
p_te      = x_te[:, 3]

y_base_tr = sindy2_stage1(x_tr_phys)
y_base_te = sindy2_stage1(x_te_phys)

# Standardise pressure alone for polynomial basis
s2_sc_p   = StandardScaler().fit(p_tr.reshape(-1, 1))
p_tr_s    = s2_sc_p.transform(p_tr.reshape(-1, 1)).ravel()
p_te_s    = s2_sc_p.transform(p_te.reshape(-1, 1)).ravel()

def _build_mult_basis(base_col, p_s, degree=POLY_DEGREE):
    """Build [base×p^0 .. base×p^K, 1, p .. p^K] basis for one output."""
    scale_cols  = [base_col * (p_s ** k) for k in range(degree + 1)]
    offset_cols = [(p_s ** k)             for k in range(degree + 1)]
    return np.column_stack(scale_cols + offset_cols)

s2_stage2 = {}
print(f"\n  Stage 2 (multiplicative pressure conditioning):")
print(f"  {'Output':<18}  {'R²':>8}  {'RMSE (total)':>14}  {'Active':>8}")
for i, col in enumerate(OUTPUT_NAMES):
    Theta_tr = _build_mult_basis(y_base_tr[:, i], p_tr_s)
    Theta_te = _build_mult_basis(y_base_te[:, i], p_te_s)

    sy2  = StandardScaler().fit(y_tr[:, i:i+1])
    y_s  = sy2.transform(y_tr[:, i:i+1]).ravel()

    # Scale Theta columns before STLSQ so threshold is meaningful
    col_sc   = StandardScaler().fit(Theta_tr)
    Th_tr_sc = col_sc.transform(Theta_tr)
    Th_te_sc = col_sc.transform(Theta_te)

    opt2  = STLSQ(threshold=SINDY_THRESHOLD, alpha=SINDY_ALPHA, max_iter=200)
    opt2.fit(Th_tr_sc, y_s)
    coef2 = np.asarray(opt2.coef_).ravel()

    y_pred = sy2.inverse_transform((Th_te_sc @ coef2).reshape(-1, 1)).ravel()
    r2_2   = r2_score(y_te[:, i], y_pred)
    rmse_2 = np.sqrt(mean_squared_error(y_te[:, i], y_pred))
    s2_stage2[col] = dict(coef=coef2, sy=sy2, col_sc=col_sc)
    print(f"  {col:<18}  {r2_2:>8.4f}  {rmse_2:>14.4f}  {int(np.sum(np.abs(coef2)>1e-12)):>8}")

def sindy2_predict(x_physical):
    """x_physical: (N,4) raw [epi,trans,endo,pressure] → (N,3) physical outputs."""
    x_physical = np.asarray(x_physical).reshape(-1, 4)
    act_phys   = x_physical[:, :3]
    p_raw      = x_physical[:, 3]
    base       = sindy2_stage1(act_phys)
    p_s        = s2_sc_p.transform(p_raw.reshape(-1, 1)).ravel()
    outs = []
    for i, col in enumerate(OUTPUT_NAMES):
        r      = s2_stage2[col]
        Theta  = _build_mult_basis(base[:, i], p_s)
        Theta_sc = r['col_sc'].transform(Theta)
        outs.append(r['sy'].inverse_transform((Theta_sc @ r['coef']).reshape(-1, 1)).ravel())
    return np.stack(outs, axis=1)

print(f"\n  SINDy-2Stage FK ready  ({time.time() - t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2E — Parameterised SINDy
#
# Fit a separate sparse polynomial at each discrete pressure level using only
# actuator inputs [epi, trans, endo]. Coefficients become smooth functions of
# pressure via linear interpolation between levels.
#
# Prediction at arbitrary pressure p:
#   coef(p)  = linear interpolation of {coef_0, coef_30, ..., coef_150} at p
#   y(act,p) = Theta(act) @ coef(p)
#
# Advantage over single-stage SINDy: polynomial never oscillates between
# pressure levels because each level is fit independently. The interpolation
# of coefficients — not the polynomial itself — handles pressure variation.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[2E] Fitting Parameterised SINDy FK model …")
t0 = time.time()

from scipy.interpolate import interp1d as _coef_interp

# Shared input scaler on actuators only (3 features) — fit on all training data
ps_sc_in  = StandardScaler().fit(x_tr[:, :3])
ps_poly   = PolynomialLibrary(degree=POLY_DEGREE, include_bias=True)
ps_poly.fit(ps_sc_in.transform(x_tr[:, :3]))

# Shared output scaler per output — fit on all training data
ps_sy     = {col: StandardScaler().fit(y_tr[:, i:i+1])
             for i, col in enumerate(OUTPUT_NAMES)}

# Per-pressure coefficient fit
ps_coefs  = {}   # {pressure_level: {output: coef_array}}
print(f"\n  {'Pressure':>10}  {'Twist R²':>10}  {'Height R²':>10}  {'Volume R²':>10}")
for p in PRESSURES:
    df_p  = data_dict[p]
    X_p   = df_p[['epi', 'trans', 'endo']].values.astype(float)
    y_p   = df_p[['dtwist_deg', 'height_mm', 'volume_endo_mL']].values.astype(float)
    Th_p  = np.asarray(ps_poly.transform(ps_sc_in.transform(X_p)))
    ps_coefs[p] = {}
    r2s = []
    for i, col in enumerate(OUTPUT_NAMES):
        y_s  = ps_sy[col].transform(y_p[:, i:i+1]).ravel()
        opt  = STLSQ(threshold=SINDY_THRESHOLD, alpha=SINDY_ALPHA, max_iter=100)
        opt.fit(Th_p, y_s)
        coef = np.asarray(opt.coef_).ravel()
        y_pred = ps_sy[col].inverse_transform((Th_p @ coef).reshape(-1,1)).ravel()
        r2s.append(r2_score(y_p[:, i], y_pred))
        ps_coefs[p][col] = coef
    print(f"  {p:>10} mmHg  {r2s[0]:>10.4f}  {r2s[1]:>10.4f}  {r2s[2]:>10.4f}")

# Build coefficient interpolators (linear interpolation across pressure levels)
_p_arr = np.array(PRESSURES, dtype=float)
ps_interps = {}
for col in OUTPUT_NAMES:
    coef_stack = np.stack([ps_coefs[p][col] for p in PRESSURES], axis=0)  # (n_P, n_coefs)
    ps_interps[col] = _coef_interp(_p_arr, coef_stack, axis=0,
                                   kind='linear', bounds_error=False,
                                   fill_value='extrapolate')

def ps_sindy_predict(x_physical):
    """x_physical: (N,4) raw [epi,trans,endo,pressure] → (N,3) physical outputs."""
    x_physical = np.asarray(x_physical).reshape(-1, 4)
    act_phys   = x_physical[:, :3]
    p_vals     = x_physical[:, 3]
    Theta      = np.asarray(ps_poly.transform(ps_sc_in.transform(act_phys)))  # (N, n_coefs)
    outs = []
    for col in OUTPUT_NAMES:
        coef_at_p = ps_interps[col](p_vals)          # (N, n_coefs)
        y_scaled  = np.sum(Theta * coef_at_p, axis=1)
        outs.append(ps_sy[col].inverse_transform(y_scaled.reshape(-1,1)).ravel())
    return np.stack(outs, axis=1)

# Test set assessment
ps_te = ps_sindy_predict(x_te)
print(f"\n  Parameterised SINDy test RMSE:")
for i, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    rmse = np.sqrt(mean_squared_error(y_te[:, i], ps_te[:, i]))
    print(f"    {col:<14}: {rmse:.4f} {unit}")

print(f"\n  Parameterised SINDy ready  ({time.time() - t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODELS — for use by mpc_ilc.py
# ══════════════════════════════════════════════════════════════════════════════
import pickle, pathlib
_save_dir = pathlib.Path(BASE) / 'saved_models'
_save_dir.mkdir(exist_ok=True)

torch.save(dd_model.state_dict(),   _save_dir / 'dd_model.pth')
torch.save(pinn_model.state_dict(), _save_dir / 'pinn_model.pth')

np.savez(_save_dir / 'norm_constants.npz',
         x_min=x_min, x_max=x_max, x_den=x_den,
         y_min=y_min, y_max=y_max, y_den=y_den)

with open(_save_dir / 'sindy_data.pkl', 'wb') as f:
    pickle.dump({'results': sindy_results, 'sc': sindy_sc,
                 'poly_lib': poly_lib}, f)

with open(_save_dir / 'sindy2_data.pkl', 'wb') as f:
    pickle.dump({'stage1': s2_stage1, 'stage2': s2_stage2,
                 'sc_in1': s2_sc_in1, 'poly_lib1': poly_lib1,
                 'sc_p': s2_sc_p}, f)

print(f"  Models saved → {_save_dir}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FK assessment on held-out test set
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("  FK ASSESSMENT — Test Set RMSE (all pressures combined)")
print(f"{'='*68}")

# DataDriven predictions
with torch.no_grad():
    dd_te_n = dd_model(x_te_t).numpy()
dd_te = denorm_y(dd_te_n)

# PINN predictions
Xte_act = torch.tensor(x_te_n[:, :3], dtype=torch.float32)
Xte_p   = torch.tensor(x_te_n[:, 3:], dtype=torch.float32)
with torch.no_grad():
    pinn_te_n = pinn_model(Xte_act, Xte_p).numpy()
pinn_te = denorm_y(pinn_te_n)

# SINDy predictions
sindy_te  = sindy_predict(x_te)
sindy2_te = sindy2_predict(x_te)
ps_te     = ps_sindy_predict(x_te)

print(f"  {'Output':<18}  {'DataDriven':>12}  {'PINN':>12}  {'SINDy':>12}  {'SINDy2':>12}  {'pSINDy':>12}")
for i, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    r_dd     = np.sqrt(mean_squared_error(y_te[:, i], dd_te[:, i]))
    r_pinn   = np.sqrt(mean_squared_error(y_te[:, i], pinn_te[:, i]))
    r_sindy  = sindy_results[col]['rmse']
    r_sindy2 = np.sqrt(mean_squared_error(y_te[:, i], sindy2_te[:, i]))
    r_ps     = np.sqrt(mean_squared_error(y_te[:, i], ps_te[:, i]))
    label    = f"{col} ({unit})"
    print(f"  {label:<18}  {r_dd:>12.4f}  {r_pinn:>12.4f}  {r_sindy:>12.4f}  {r_sindy2:>12.4f}  {r_ps:>12.4f}")

# Parity plots (3 outputs × 5 architectures)
fig_par, axes_par = plt.subplots(3, 5, figsize=(22, 11))
model_names  = ['DataDriven', 'PINN', 'SINDy', 'SINDy2', 'pSINDy']
all_te_preds = [dd_te, pinn_te, sindy_te, sindy2_te, ps_te]

for oi, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    for mi, (mname, pred) in enumerate(zip(model_names, all_te_preds)):
        ax   = axes_par[oi, mi]
        true = y_te[:, oi]
        p    = pred[:, oi]
        rmse_v = np.sqrt(mean_squared_error(true, p))
        # colour points by pressure level
        p_idx = x_te[:, 3].astype(int)
        sc = ax.scatter(true, p, c=x_te[:, 3], cmap='viridis', alpha=0.3, s=6)
        lo = min(true.min(), p.min()); hi = max(true.max(), p.max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1.2)
        ax.set_title(f'{mname}: {col}\nRMSE={rmse_v:.3f} {unit}', fontsize=9)
        ax.set_xlabel(f'Measured {col}', fontsize=8)
        ax.set_ylabel(f'Predicted {col}', fontsize=8)
        ax.grid(True, alpha=0.3)
        if oi == 0:
            plt.colorbar(sc, ax=ax, label='Pressure (mmHg)', pad=0.01)

plt.suptitle('FK Model Assessment — Predicted vs Measured (Test Set, all pressures)',
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'fk_parity_all.png'), dpi=150, bbox_inches='tight')
print("\n  Parity plot saved → fk_parity_all.png")
plt.close(fig_par)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3.5 — Jacobian accuracy check
#
# MPC steers actuators using ∂FK/∂actuators (the Jacobian).
# Good RMSE does NOT guarantee accurate gradients.
# We compare the analytical Jacobian (autograd / polynomial derivative)
# against a numerical finite-difference Jacobian at N_JAC test points.
# Large disagreement → MPC will walk in wrong directions.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("  JACOBIAN ACCURACY CHECK  (analytical vs finite-difference)")
print(f"{'='*68}")

N_JAC   = 50
FD_EPS  = 1e-4   # finite-difference step in normalised actuator space
rng_idx = np.random.choice(len(x_te_n), N_JAC, replace=False)
X_jac   = x_te_n[rng_idx]          # (N_JAC, 4) normalised

def fd_jacobian_nn(model_fn, x_norm_np):
    """3×3 finite-difference Jacobian of model_fn w.r.t. actuators (cols 0-2)."""
    J = np.zeros((3, 3))
    f0 = model_fn(x_norm_np)
    for j in range(3):
        xp = x_norm_np.copy(); xp[j] += FD_EPS
        J[:, j] = (model_fn(xp) - f0) / FD_EPS
    return J

def analytical_jacobian_nn(model, x_norm_np):
    """3×3 analytical Jacobian via PyTorch autograd."""
    x_t = torch.tensor(x_norm_np, dtype=torch.float32).unsqueeze(0)
    x_t.requires_grad_(True)
    if hasattr(model, 'act_branch'):   # PINN
        act_t = x_t[:, :3]; p_t = x_t[:, 3:]
        y = model(act_t, p_t)
    else:
        y = model(x_t)
    J = np.zeros((3, 3))
    for oi in range(3):
        if x_t.grad is not None: x_t.grad.zero_()
        y[0, oi].backward(retain_graph=True)
        J[oi, :] = x_t.grad[0, :3].numpy()
    return J

# DD model helpers
def dd_fwd(x_np):
    with torch.no_grad():
        return dd_model(torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)).numpy()[0]

# PINN model helpers
def pinn_fwd(x_np):
    xt = torch.tensor(x_np[:3], dtype=torch.float32).unsqueeze(0)
    pt = torch.tensor([[x_np[3]]], dtype=torch.float32)
    with torch.no_grad():
        return pinn_model(xt, pt).numpy()[0]

pinn_model.requires_grad_(True)   # re-enable for Jacobian

jac_results = {}
print(f"\n  {'Model':<12}  {'Mean |J_ana - J_fd|':>22}  {'Relative error':>16}")
for mname, ana_fn, fwd_fn in [
    ('DataDriven', lambda x: analytical_jacobian_nn(dd_model, x),   dd_fwd),
    ('PINN',       lambda x: analytical_jacobian_nn(pinn_model, x), pinn_fwd),
]:
    errs, rel_errs = [], []
    for xi in X_jac:
        J_ana = ana_fn(xi)
        J_fd  = fd_jacobian_nn(fwd_fn, xi)
        err   = np.abs(J_ana - J_fd).mean()
        denom = np.abs(J_fd).mean() + 1e-8
        errs.append(err); rel_errs.append(err / denom)
    jac_results[mname] = {'mae': np.mean(errs), 'rel': np.mean(rel_errs)}
    print(f"  {mname:<12}  {np.mean(errs):>22.6f}  {np.mean(rel_errs):>15.3f}x")

pinn_model.requires_grad_(False)  # freeze again for MPC

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Optimiser-based IK (MPC strategy)
#
# For each time-step: find [epi_n, trans_n, endo_n] in [0,1] that minimises
#   L = ||FK(act, p) - desired||² + λ_reg||act-0.5||² + λ_smooth||act-prev||²
# NN models: Adam gradient descent through the model (projected gradient, clamp)
# SINDy:     scipy L-BFGS-B (model is differentiable but no autograd required)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("  OPTIMISER-BASED IK (MPC STRATEGY)  [multi-start + convergence stop]")
print(f"{'='*68}")

N_RESTARTS  = 3      # random restarts per time-step (plus warm start = N_RESTARTS+1 total)
GRAD_TOL    = 1e-5   # stop early if gradient norm falls below this

def _run_one_start(loss_fn, init, n_steps, prev):
    """Run Adam from a single initialisation; return (best_act, best_loss)."""
    act = init.clone().detach().requires_grad_(True)
    opt = optim.Adam([act], lr=MPC_LR, betas=(0.9, 0.999))
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)
    best_loss, best_act = float('inf'), act.detach().clone()
    for _ in range(n_steps):
        opt.zero_grad()
        loss = loss_fn(act, prev)
        loss.backward()
        grad_norm = act.grad.norm().item()
        opt.step()
        with torch.no_grad():
            lo = torch.tensor(ACT_LO_N, dtype=torch.float32)
            hi = torch.tensor(ACT_HI_N, dtype=torch.float32)
            act.clamp_(lo, hi)
        sch.step(loss)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_act  = act.detach().clone()
        if grad_norm < GRAD_TOL:
            break
    return best_act, best_loss

def _mpc_step_nn(loss_fn, prev, n_steps):
    """Multi-start MPC for one time-step; returns best actuator tensor."""
    inits = []
    if prev is not None:
        inits.append(prev.clone())           # warm start
    else:
        inits.append(torch.full((1, 3), 0.5))
    for _ in range(N_RESTARTS):
        inits.append(torch.rand(1, 3))       # random restarts
    best_act, best_loss = None, float('inf')
    for init in inits:
        act, loss = _run_one_start(loss_fn, init, n_steps, prev)
        if loss < best_loss:
            best_loss, best_act = loss, act
    return best_act, best_loss

# ── 4A: DataDriven MPC ───────────────────────────────────────────────────────
print(f"\n[4A] DataDriven MPC ({n_targets} time-steps, {N_RESTARTS+1} starts each) …")
t0 = time.time()
act_dd_n  = np.empty((n_targets, 3))
prev_dd   = None

for i in range(n_targets):
    tgt_i = torch.tensor(targets_norm[i:i+1], dtype=torch.float32)
    p_i   = torch.tensor([[pressure_n[i]]], dtype=torch.float32)
    n_s   = N_STEPS_FIRST if prev_dd is None else N_STEPS_WARM

    def dd_loss(act, prev):
        x_in = torch.cat([act, p_i], dim=1)
        l = nn.functional.mse_loss(dd_model(x_in), tgt_i)
        if prev is not None:
            l = l + LAMBDA_REG * (act - 0.5).pow(2).sum()
            l = l + LAMBDA_SMOOTH * (act - prev).pow(2).sum()
        return l

    cur, best_loss = _mpc_step_nn(dd_loss, prev_dd, n_s)
    act_dd_n[i] = cur.numpy()[0]
    prev_dd = cur
    if i % 10 == 0:
        print(f"    {i:3d}/{n_targets}  best_loss={best_loss:.6f}", flush=True)

print(f"  DataDriven MPC done  ({time.time()-t0:.1f}s)")

# ── 4B: PINN MPC ─────────────────────────────────────────────────────────────
pinn_model.requires_grad_(True)
print(f"\n[4B] PINN MPC ({n_targets} time-steps, {N_RESTARTS+1} starts each) …")
t0 = time.time()
act_pinn_n = np.empty((n_targets, 3))
prev_pinn  = None

for i in range(n_targets):
    tgt_i = torch.tensor(targets_norm[i:i+1], dtype=torch.float32)
    p_i   = torch.tensor([[pressure_n[i]]], dtype=torch.float32)
    n_s   = N_STEPS_FIRST if prev_pinn is None else N_STEPS_WARM

    def pinn_loss(act, prev):
        l = nn.functional.mse_loss(pinn_model(act, p_i), tgt_i)
        if prev is not None:
            l = l + LAMBDA_REG * (act - 0.5).pow(2).sum()
            l = l + LAMBDA_SMOOTH * (act - prev).pow(2).sum()
        return l

    cur, best_loss = _mpc_step_nn(pinn_loss, prev_pinn, n_s)
    act_pinn_n[i] = cur.numpy()[0]
    prev_pinn = cur
    if i % 10 == 0:
        print(f"    {i:3d}/{n_targets}  best_loss={best_loss:.6f}", flush=True)

print(f"  PINN MPC done  ({time.time()-t0:.1f}s)")
pinn_model.requires_grad_(False)

# ── 4C: SINDy MPC ────────────────────────────────────────────────────────────
print(f"\n[4C] SINDy MPC ({n_targets} time-steps, scipy L-BFGS-B) …")
t0 = time.time()
act_sindy_n = np.empty((n_targets, 3))
x0_s = np.full(3, 0.5)
prev_s = None

for i in range(n_targets):
    p_raw   = desired_p[i]
    tgt_raw = tgt_phys[i]                       # physical [twist, height, volume]

    _prev = prev_s.copy() if prev_s is not None else None

    def sindy_obj(act_n, _p=p_raw, _tgt=tgt_raw, _prev=_prev):
        act_phys = denorm_x(act_n)               # normalised → physical [epi,trans,endo]
        x_phys   = np.append(act_phys, _p).reshape(1, -1)
        y_hat    = sindy_predict(x_phys)[0]
        fk_loss  = np.sum(((y_hat - _tgt) / y_den) ** 2)
        reg      = LAMBDA_REG * np.sum((act_n - 0.5) ** 2)
        smooth   = LAMBDA_SMOOTH * np.sum((act_n - _prev) ** 2) if _prev is not None else 0.
        return fk_loss + reg + smooth

    sol = minimize(sindy_obj, x0_s, method='L-BFGS-B',
                   bounds=ACT_BOUNDS_LBFGS,
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    act_s = np.clip(sol.x, ACT_LO_N, ACT_HI_N)
    act_sindy_n[i] = act_s
    x0_s   = act_s
    prev_s = act_s
    if i % 10 == 0:
        print(f"    {i:3d}/{n_targets}  obj={sol.fun:.6f}", flush=True)

print(f"  SINDy MPC done  ({time.time()-t0:.1f}s)")

# ── 4D: SINDy-2Stage MPC ─────────────────────────────────────────────────────
print(f"\n[4D] SINDy-2Stage MPC ({n_targets} time-steps, scipy L-BFGS-B) …")
t0 = time.time()
act_sindy2_n = np.empty((n_targets, 3))
x0_s2 = np.full(3, 0.5)
prev_s2 = None

for i in range(n_targets):
    p_raw   = desired_p[i]
    tgt_raw = tgt_phys[i]

    _prev = prev_s2.copy() if prev_s2 is not None else None

    def sindy2_obj(act_n, _p=p_raw, _tgt=tgt_raw, _prev=_prev):
        act_phys = denorm_x(act_n)
        x_phys   = np.append(act_phys, _p).reshape(1, -1)
        y_hat    = sindy2_predict(x_phys)[0]
        fk_loss  = np.sum(((y_hat - _tgt) / y_den) ** 2)
        reg      = LAMBDA_REG * np.sum((act_n - 0.5) ** 2)
        smooth   = LAMBDA_SMOOTH * np.sum((act_n - _prev) ** 2) if _prev is not None else 0.
        return fk_loss + reg + smooth

    sol = minimize(sindy2_obj, x0_s2, method='L-BFGS-B',
                   bounds=ACT_BOUNDS_LBFGS,
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    act_s2 = np.clip(sol.x, ACT_LO_N, ACT_HI_N)
    act_sindy2_n[i] = act_s2
    x0_s2   = act_s2
    prev_s2 = act_s2
    if i % 10 == 0:
        print(f"    {i:3d}/{n_targets}  obj={sol.fun:.6f}", flush=True)

print(f"  SINDy-2Stage MPC done  ({time.time()-t0:.1f}s)")

# ── 4E: Parameterised SINDy MPC ──────────────────────────────────────────────
print(f"\n[4E] Parameterised SINDy MPC ({n_targets} time-steps, scipy L-BFGS-B) …")
t0 = time.time()
act_ps_n = np.empty((n_targets, 3))
x0_ps    = np.full(3, 0.5)
prev_ps  = None

for i in range(n_targets):
    p_raw   = desired_p[i]
    tgt_raw = tgt_phys[i]
    _prev   = prev_ps.copy() if prev_ps is not None else None

    def ps_obj(act_n, _p=p_raw, _tgt=tgt_raw, _prev=_prev):
        act_phys = denorm_x(act_n)
        x_phys   = np.append(act_phys, _p).reshape(1, -1)
        y_hat    = ps_sindy_predict(x_phys)[0]
        fk_loss  = np.sum(((y_hat - _tgt) / y_den) ** 2)
        reg      = LAMBDA_REG * np.sum((act_n - 0.5) ** 2)
        smooth   = LAMBDA_SMOOTH * np.sum((act_n - _prev) ** 2) if _prev is not None else 0.
        return fk_loss + reg + smooth

    sol = minimize(ps_obj, x0_ps, method='L-BFGS-B',
                   bounds=ACT_BOUNDS_LBFGS,
                   options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    act_s = np.clip(sol.x, ACT_LO_N, ACT_HI_N)
    act_ps_n[i] = act_s
    x0_ps   = act_s
    prev_ps = act_s
    if i % 10 == 0:
        print(f"    {i:3d}/{n_targets}  obj={sol.fun:.6f}", flush=True)

print(f"  Parameterised SINDy MPC done  ({time.time()-t0:.1f}s)")

# Convert normalised actuator solutions → physical units (mm)
act_dd_phys     = denorm_x(act_dd_n)
act_pinn_phys   = denorm_x(act_pinn_n)
act_sindy_phys  = denorm_x(act_sindy_n)
act_sindy2_phys = denorm_x(act_sindy2_n)
act_ps_phys     = denorm_x(act_ps_n)

# Save actuator CSVs
for arr, name in zip([act_dd_phys, act_pinn_phys, act_sindy_phys, act_sindy2_phys, act_ps_phys],
                     ['DataDriven', 'PINN', 'SINDy', 'SINDy2', 'pSINDy']):
    df_out = pd.DataFrame(arr, columns=['epi', 'trans', 'endo'])
    df_out['pressure'] = desired_p
    df_out['time']     = time_vec
    df_out.to_csv(os.path.join(BASE, f'MPC_actuators_{name}.csv'), index=False)
print("\n  Actuator CSVs saved → MPC_actuators_{DataDriven,PINN,SINDy}.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FK verification: actuators → deformation through each model
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("  FK VERIFICATION — plug actuator solutions back through each FK model")
print(f"{'='*68}")

# DataDriven
x_dd_in = np.hstack([act_dd_n, pressure_n.reshape(-1, 1)])
with torch.no_grad():
    fk_dd = denorm_y(dd_model(torch.tensor(x_dd_in, dtype=torch.float32)).numpy())

# PINN
with torch.no_grad():
    fk_pinn = denorm_y(pinn_model(
        torch.tensor(act_pinn_n, dtype=torch.float32),
        torch.tensor(pressure_n.reshape(-1, 1), dtype=torch.float32)).numpy())

# SINDy
x_sindy_in  = np.hstack([act_sindy_phys,  desired_p.reshape(-1, 1)])
fk_sindy    = sindy_predict(x_sindy_in)

# SINDy-2Stage
x_sindy2_in = np.hstack([act_sindy2_phys, desired_p.reshape(-1, 1)])
fk_sindy2   = sindy2_predict(x_sindy2_in)

x_ps_in     = np.hstack([act_ps_phys, desired_p.reshape(-1, 1)])
fk_ps       = ps_sindy_predict(x_ps_in)

print(f"\n  MPC tracking RMSE (FK model output vs desired):")
print(f"  {'Output':<18}  {'DataDriven':>12}  {'PINN':>12}  {'SINDy':>12}  {'SINDy2':>12}  {'pSINDy':>12}")
mpc_rmse = {}
for i, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    r_dd     = np.sqrt(np.mean((fk_dd[:, i]     - tgt_phys[:, i]) ** 2))
    r_pinn   = np.sqrt(np.mean((fk_pinn[:, i]   - tgt_phys[:, i]) ** 2))
    r_sindy  = np.sqrt(np.mean((fk_sindy[:, i]  - tgt_phys[:, i]) ** 2))
    r_sindy2 = np.sqrt(np.mean((fk_sindy2[:, i] - tgt_phys[:, i]) ** 2))
    r_ps     = np.sqrt(np.mean((fk_ps[:, i]     - tgt_phys[:, i]) ** 2))
    mpc_rmse[col] = {'DataDriven': r_dd, 'PINN': r_pinn,
                     'SINDy': r_sindy,   'SINDy2': r_sindy2, 'pSINDy': r_ps}
    print(f"  {col+' ('+unit+')':<18}  {r_dd:>12.4f}  {r_pinn:>12.4f}  {r_sindy:>12.4f}  {r_sindy2:>12.4f}  {r_ps:>12.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5.5 — KDE-conditional error analysis
#
# Splits the trajectory into high-density and low-density regions (per KDE)
# and computes MPC tracking RMSE separately for each.
# If RMSE_low >> RMSE_high → error is provably data-range limited.
# If RMSE_low ≈ RMSE_high → something else is causing the error.
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*68}")
print("  SECTION 5.5 — KDE-CONDITIONAL ERROR ANALYSIS")
print(f"{'='*68}")

from scipy.stats import gaussian_kde as _kde_fn
from sklearn.preprocessing import StandardScaler as _SS

KDE_THRESH = 0.3   # same threshold as explore_data.py

# Fit 4-D KDE on training data: (volume, height, twist, pressure)
_X_tr_kde = np.column_stack([y_all[:, 2], y_all[:, 1], y_all[:, 0], x_all[:, 3]])
_X_tj_kde = np.column_stack([tgt_phys[:, 2], tgt_phys[:, 1], tgt_phys[:, 0], desired_p])

_sc       = _SS().fit(_X_tr_kde)
_kde      = _kde_fn(_sc.transform(_X_tr_kde).T)
_tr_med   = np.median(_kde(_sc.transform(_X_tr_kde).T))
traj_kde  = _kde(_sc.transform(_X_tj_kde).T) / _tr_med   # relative density along traj

HIGH = traj_kde >= KDE_THRESH
LOW  = traj_kde <  KDE_THRESH

print(f"\n  Trajectory split: {HIGH.sum()} high-density pts  |  {LOW.sum()} low-density pts")
print(f"  (threshold = {KDE_THRESH} × training median)\n")

arches = ['DataDriven', 'PINN', 'SINDy', 'SINDy2', 'pSINDy']
fks    = [fk_dd, fk_pinn, fk_sindy, fk_sindy2, fk_ps]

print(f"  {'':30}  {'High-density':>14}  {'Low-density':>13}  {'Ratio (lo/hi)':>14}")
print(f"  {'':30}  {'(KDE ≥ 0.3)':>14}  {'(KDE < 0.3)':>13}  {'':>14}")
for arch, fk in zip(arches, fks):
    for i, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
        err = (fk[:, i] - tgt_phys[:, i]) ** 2
        rmse_hi = np.sqrt(err[HIGH].mean()) if HIGH.any() else float('nan')
        rmse_lo = np.sqrt(err[LOW].mean())  if LOW.any()  else float('nan')
        ratio   = rmse_lo / rmse_hi if rmse_hi > 1e-9 else float('nan')
        label   = f"{arch} {col} ({unit})"
        print(f"  {label:<30}  {rmse_hi:>14.4f}  {rmse_lo:>13.4f}  {ratio:>14.2f}x")
    print()

print("  Ratio >> 1  →  error is data-range limited (supports ILC strategy)")
print("  Ratio ≈  1  →  error is model/architecture limited")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Figures
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[6] Generating comparison figures …")

# ── Fig 1: Optimised actuator signals ─────────────────────────────────────────
fig1, ax1 = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
for ax, label, mi in zip(ax1, MOTOR_LABELS, range(3)):
    for arch, arr in zip(['DataDriven', 'PINN', 'SINDy', 'SINDy2', 'pSINDy'],
                         [act_dd_phys, act_pinn_phys, act_sindy_phys, act_sindy2_phys, act_ps_phys]):
        ax.plot(time_vec, arr[:, mi],
                color=ARCH_COLORS[arch], lw=1.8,
                linestyle=ARCH_LINES[arch], label=arch)
    ax.set_ylabel(f'{label} (mm)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.35)
ax1[-1].set_xlabel('Time (s)', fontsize=10)
plt.suptitle('MPC-Optimised Actuator Signals: 5 FK Architectures', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'mpc_actuators.png'), dpi=150, bbox_inches='tight')
print("  Fig 1 saved → mpc_actuators.png")
plt.close(fig1)

# ── Fig 2: FK deformation vs desired ─────────────────────────────────────────
from matplotlib.collections import LineCollection

def _pressure_line(ax, x, y, p, lw=2.5):
    """Draw a line coloured continuously by pressure p."""
    pts   = np.array([x, y]).T.reshape(-1, 1, 2)
    segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm_ = plt.Normalize(p.min(), p.max())
    lc    = LineCollection(segs, cmap='coolwarm', norm=norm_, lw=lw, zorder=4)
    lc.set_array(p)
    ax.add_collection(lc)
    return lc

fig2, ax2 = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
_lc_ref = None
for ax, col, unit, oi in zip(ax2, OUTPUT_NAMES, OUTPUT_UNITS, range(3)):
    _lc_ref = _pressure_line(ax, time_vec, tgt_phys[:, oi], desired_p)
    ax.plot([], [], color='grey', lw=2.5, label='Desired (colour=pressure)')
    for arch, fk in zip(['DataDriven', 'PINN', 'SINDy', 'SINDy2', 'pSINDy'],
                        [fk_dd, fk_pinn, fk_sindy, fk_sindy2, fk_ps]):
        r = mpc_rmse[col][arch]
        ax.plot(time_vec, fk[:, oi],
                color=ARCH_COLORS[arch], lw=1.8,
                linestyle=ARCH_LINES[arch],
                label=f'{arch}  RMSE={r:.3f} {unit}')
    ax.set_ylabel(f'{col} ({unit})', fontsize=9)
    ax.autoscale_view()
    ax.legend(fontsize=8); ax.grid(True, alpha=0.35)
ax2[-1].set_xlabel('Time (s)', fontsize=10)
fig2.colorbar(_lc_ref, ax=ax2.tolist(), label='Pressure (mmHg)', shrink=0.6, pad=0.02)
plt.suptitle('FK Verification: Deformation vs Desired Trajectory\n'
             '(each architecture uses its own FK model)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'mpc_fk_verification.png'), dpi=150, bbox_inches='tight')
print("  Fig 2 saved → mpc_fk_verification.png")
plt.close(fig2)

# ── Fig 3: Pressure sensitivity sweep ─────────────────────────────────────────
print("  [6c] Pressure sweep …")
p_sweep   = np.linspace(0, 150, 100)
act_mid   = np.mean(x_all[:, :3], axis=0)   # physical mean actuator values
act_tile  = np.tile(act_mid, (100, 1))      # (100, 3) physical actuators

x_sw_phys = np.hstack([act_tile, p_sweep.reshape(-1, 1)])
x_sw_n    = norm_x(x_sw_phys)

with torch.no_grad():
    fk_sw_dd = denorm_y(
        dd_model(torch.tensor(x_sw_n, dtype=torch.float32)).numpy())

with torch.no_grad():
    fk_sw_pinn = denorm_y(
        pinn_model(torch.tensor(x_sw_n[:, :3], dtype=torch.float32),
                   torch.tensor(x_sw_n[:, 3:], dtype=torch.float32)).numpy())

fk_sw_sindy  = sindy_predict(x_sw_phys)
fk_sw_sindy2 = sindy2_predict(x_sw_phys)
fk_sw_ps     = ps_sindy_predict(x_sw_phys)

fig3, ax3 = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, unit, oi in zip(ax3, OUTPUT_NAMES, OUTPUT_UNITS, range(3)):
    for arch, fk in zip(['DataDriven', 'PINN', 'SINDy', 'SINDy2', 'pSINDy'],
                        [fk_sw_dd, fk_sw_pinn, fk_sw_sindy, fk_sw_sindy2, fk_sw_ps]):
        ax.plot(p_sweep, fk[:, oi],
                color=ARCH_COLORS[arch], lw=2.2,
                linestyle=ARCH_LINES[arch], label=arch)
    ax.set_xlabel('Pressure (mmHg)', fontsize=10)
    ax.set_ylabel(f'{col} ({unit})', fontsize=10)
    ax.set_title(f'{col} vs Pressure\n(fixed mid-range actuators)', fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.35)
plt.suptitle('Pressure Sensitivity Comparison\n'
             f'Actuators fixed at dataset mean: epi={act_mid[0]:.1f}, '
             f'trans={act_mid[1]:.1f}, endo={act_mid[2]:.1f} mm', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'fk_pressure_sensitivity.png'), dpi=150, bbox_inches='tight')
print("  Fig 3 saved → fk_pressure_sensitivity.png")
plt.close(fig3)

# ── Fig 4: RMSE bar chart summary ────────────────────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(13, 6))

# Left: FK test-set RMSE
ax_fk = axes4[0]
arch_keys = ['DataDriven', 'PINN', 'SINDy', 'SINDy2', 'pSINDy']
fk_rmse_vals = {
    col: {
        'DataDriven': np.sqrt(mean_squared_error(y_te[:, i], dd_te[:, i])),
        'PINN':       np.sqrt(mean_squared_error(y_te[:, i], pinn_te[:, i])),
        'SINDy':      sindy_results[col]['rmse'],
        'SINDy2':     np.sqrt(mean_squared_error(y_te[:, i], sindy2_te[:, i])),
        'pSINDy':     np.sqrt(mean_squared_error(y_te[:, i], ps_te[:, i])),
    }
    for i, col in enumerate(OUTPUT_NAMES)
}
x_pos = np.arange(len(OUTPUT_NAMES))
width = 0.15
for ki, arch in enumerate(arch_keys):
    vals = [fk_rmse_vals[col][arch] for col in OUTPUT_NAMES]
    ax_fk.bar(x_pos + ki * width, vals, width, label=arch,
              color=ARCH_COLORS[arch], alpha=0.85)
ax_fk.set_xticks(x_pos + width * 2)
ax_fk.set_xticklabels([f'{col}\n({u})' for col, u in zip(OUTPUT_NAMES, OUTPUT_UNITS)],
                       fontsize=9)
ax_fk.set_ylabel('RMSE', fontsize=10)
ax_fk.set_title('FK Test-Set RMSE\n(static mapping quality)', fontsize=11)
ax_fk.legend(fontsize=9); ax_fk.grid(True, axis='y', alpha=0.35)

# Right: MPC tracking RMSE
ax_mpc = axes4[1]
for ki, arch in enumerate(arch_keys):
    vals = [mpc_rmse[col][arch] for col in OUTPUT_NAMES]
    ax_mpc.bar(x_pos + ki * width, vals, width, label=arch,
               color=ARCH_COLORS[arch], alpha=0.85)
ax_mpc.set_xticks(x_pos + width * 2)
ax_mpc.set_xticklabels([f'{col}\n({u})' for col, u in zip(OUTPUT_NAMES, OUTPUT_UNITS)],
                        fontsize=9)
ax_mpc.set_ylabel('RMSE', fontsize=10)
ax_mpc.set_title('MPC Tracking RMSE\n(trajectory following quality)', fontsize=11)
ax_mpc.legend(fontsize=9); ax_mpc.grid(True, axis='y', alpha=0.35)

plt.suptitle('Architecture Comparison Summary\nDataDriven | PINN | SINDy | SINDy-2Stage', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'rmse_summary.png'), dpi=150, bbox_inches='tight')
print("  Fig 4 saved → rmse_summary.png")
plt.close(fig4)

# ── Fig 5: Error over time for each output ────────────────────────────────────
fig5, ax5 = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
for ax, col, unit, oi in zip(ax5, OUTPUT_NAMES, OUTPUT_UNITS, range(3)):
    for arch, fk in zip(['DataDriven', 'PINN', 'SINDy', 'SINDy2', 'pSINDy'],
                        [fk_dd, fk_pinn, fk_sindy, fk_sindy2, fk_ps]):
        err = np.abs(fk[:, oi] - tgt_phys[:, oi])
        ax.plot(time_vec, err,
                color=ARCH_COLORS[arch], lw=1.5,
                linestyle=ARCH_LINES[arch], label=arch)
    ax.set_ylabel(f'|Error| {col} ({unit})', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.35)
ax5[-1].set_xlabel('Time (s)', fontsize=10)
plt.suptitle('MPC Tracking Error Over Time', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'mpc_error_time.png'), dpi=150, bbox_inches='tight')
print("  Fig 5 saved → mpc_error_time.png")
plt.close(fig5)

print(f"\n{'='*68}")
print("  COMPLETE — all outputs saved to Sindy/ folder")
print(f"{'='*68}")
plt.show()
