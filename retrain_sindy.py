"""
retrain_sindy.py
──────────────────────────────────────────────────────────────────────────────
Retrain the SINDy FK model on updated V_adjusted CSV files, then compare
the volume Jacobian before and after to quantify the impact on ILC.

Run with ML_env (needs pysindy):
  conda run -n ML_env python retrain_sindy.py

What it does
────────────
  1. Load old sindy_data.pkl + norm_constants.npz (for comparison baseline)
  2. Load updated deformation_by_actuator_V_adjusted_*mmhg.csv files
  3. Retrain SINDy (same hyperparameters as pressure_fk_comparison.py)
  4. Back up old models → sindy_data_prev.pkl / norm_constants_prev.npz
  5. Save new sindy_data.pkl + norm_constants.npz
  6. Plot: training data volume distributions + Jacobian comparison
"""

import os, sys, time, pickle, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.ndimage import gaussian_filter1d
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
SHARED    = os.path.normpath(os.path.join(BASE, '..', '..', 'sharedCSVs'))
SAVE_DIR  = pathlib.Path(BASE) / 'saved_models'
SAVE_DIR.mkdir(exist_ok=True)

# ── Data source toggle ───────────────────────────────────────────────────────
# True  → deformation_by_actuator_V_adjusted_{p}mmhg.csv  (volume-ratio corrected)
# False → deformation_by_actuator_{p}mmhg.csv             (original, no volume correction)
USE_V_ADJUSTED = False

# ── Hyperparameters (must match pressure_fk_comparison.py) ───────────────────
PRESSURES       = [0, 30, 60, 90, 120, 150]
POLY_DEGREE     = 3
SINDY_THRESHOLD = 0.05
SINDY_ALPHA     = 0.05
OUTPUT_NAMES    = ['Twist_deg', 'Height_mm', 'Volume_mL']
OUTPUT_UNITS    = ['deg', 'mm', 'mL']
SINDY_SIGMA     = 4          # Gaussian smoothing sigma used in ilcCorrection.py
FD_EPS_PHYS     = 0.01       # finite-diff step in physical actuator mm for Jacobian

# ══════════════════════════════════════════════════════════════════════════════
# 1. Load OLD model (for comparison)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print("  Loading OLD sindy_data.pkl for comparison baseline ...")
print("=" * 64)

old_pkl_path  = SAVE_DIR / 'sindy_data.pkl'
old_nrm_path  = SAVE_DIR / 'norm_constants.npz'

old_sindy = None
old_nrm   = None

if old_pkl_path.exists() and old_nrm_path.exists():
    try:
        with open(old_pkl_path, 'rb') as f:
            old_sindy = pickle.load(f)
        old_nrm = np.load(old_nrm_path)
        print(f"  Old model loaded: poly_lib has "
              f"{old_sindy['poly_lib'].n_output_features_} features")
        print(f"  Old norm: Volume range = [{old_nrm['y_min'][2]:.1f}, "
              f"{old_nrm['y_min'][2]+old_nrm['y_den'][2]:.1f}] mL")
    except Exception as e:
        print(f"  WARNING: could not load old model: {e}")
        old_sindy = None
else:
    print("  No existing model found — will skip comparison plot.")

def _old_sindy_pred(x_physical):
    """Predict with old SINDy model. x_physical: (N,4) → (N,3) physical."""
    x_s   = old_sindy['sc'].transform(np.asarray(x_physical).reshape(-1, 4))
    theta = np.asarray(old_sindy['poly_lib'].transform(x_s))
    outs  = []
    for col in OUTPUT_NAMES:
        r = old_sindy['results'][col]
        outs.append(r['sy'].inverse_transform(
            (theta @ r['coef']).reshape(-1, 1)).ravel())
    return np.stack(outs, axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Load updated V_adjusted CSV files
# ══════════════════════════════════════════════════════════════════════════════
_data_label = 'V_adjusted' if USE_V_ADJUSTED else 'original (no volume correction)'
print(f"\n{'='*64}")
print(f"  Loading {_data_label} CSV files ...")
print(f"{'='*64}")

data_dict = {}
for p in PRESSURES:
    _suffix = f'V_adjusted_{p}mmhg' if USE_V_ADJUSTED else f'{p}mmhg'
    fpath = os.path.join(SHARED, f'deformation_by_actuator_{_suffix}.csv')
    df = pd.read_csv(fpath).dropna()
    df['nominal_pressure'] = p
    data_dict[p] = df
    v = df['volume_endo_mL']
    print(f"  {p:3d} mmHg  n={len(df):4d}  vol=[{v.min():.1f}, {v.max():.1f}]  mean={v.mean():.1f}")

x_list, y_list = [], []
for p in PRESSURES:
    df = data_dict[p]
    x_list.append(df[['epi', 'trans', 'endo', 'nominal_pressure']].values.astype(float))
    y_list.append(df[['dtwist_deg', 'height_mm', 'volume_endo_mL']].values.astype(float))

x_all = np.vstack(x_list)
y_all = np.vstack(y_list)
print(f"\n  Combined: {len(x_all)} rows")
print(f"  Volume range (all pressures): [{y_all[:,2].min():.1f}, {y_all[:,2].max():.1f}] mL")

# 70/20/10 split (same seed as pressure_fk_comparison.py)
x_tr, x_tmp, y_tr, y_tmp = train_test_split(
    x_all, y_all, test_size=0.30, random_state=42, shuffle=True)
x_te, x_va, y_te, y_va = train_test_split(
    x_tmp, y_tmp, test_size=1/3, random_state=42, shuffle=True)
print(f"  Train: {len(x_tr)}  Val: {len(x_va)}  Test: {len(x_te)}")

# New normalisation constants
x_min = x_tr.min(axis=0);  x_max = x_tr.max(axis=0)
y_min = y_tr.min(axis=0);  y_max = y_tr.max(axis=0)
x_den = np.where(x_max - x_min > 1e-12, x_max - x_min, 1e-12)
y_den = np.where(y_max - y_min > 1e-12, y_max - y_min, 1e-12)

def norm_x(x): return (np.asarray(x) - x_min) / x_den
def norm_y(y): return (np.asarray(y) - y_min) / y_den
def denorm_y(yn): return np.asarray(yn) * y_den + y_min

# ══════════════════════════════════════════════════════════════════════════════
# 3. Retrain SINDy
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}")
print(f"  Training new SINDy FK  (degree={POLY_DEGREE}, threshold={SINDY_THRESHOLD}) ...")
print(f"{'='*64}")
t0 = time.time()

sindy_sc  = StandardScaler()
sindy_sc.fit(x_tr)
poly_lib  = PolynomialLibrary(degree=POLY_DEGREE, include_bias=True)
poly_lib.fit(sindy_sc.transform(x_tr))
Theta_tr  = np.asarray(poly_lib.transform(sindy_sc.transform(x_tr)))
Theta_te  = np.asarray(poly_lib.transform(sindy_sc.transform(x_te)))

sindy_results = {}
print(f"\n  {'Output':<18}  {'R2 (test)':>10}  {'RMSE':>10}  {'Active terms':>13}")
for i, col in enumerate(OUTPUT_NAMES):
    sy  = StandardScaler()
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

print(f"\n  SINDy ready  ({time.time()-t0:.1f}s)")

def _new_sindy_pred(x_physical):
    """Predict with new SINDy model. x_physical: (N,4) → (N,3) physical."""
    x_s   = sindy_sc.transform(np.asarray(x_physical).reshape(-1, 4))
    theta = np.asarray(poly_lib.transform(x_s))
    outs  = []
    for col in OUTPUT_NAMES:
        r = sindy_results[col]
        outs.append(r['sy'].inverse_transform(
            (theta @ r['coef']).reshape(-1, 1)).ravel())
    return np.stack(outs, axis=1)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Back up old models + save new
# ══════════════════════════════════════════════════════════════════════════════
import shutil

if old_pkl_path.exists():
    shutil.copy2(old_pkl_path, SAVE_DIR / 'sindy_data_prev.pkl')
    print(f"\n  Backed up → sindy_data_prev.pkl")
if old_nrm_path.exists():
    shutil.copy2(old_nrm_path, SAVE_DIR / 'norm_constants_prev.npz')
    print(f"  Backed up → norm_constants_prev.npz")

with open(SAVE_DIR / 'sindy_data.pkl', 'wb') as f:
    pickle.dump({'results': sindy_results, 'sc': sindy_sc, 'poly_lib': poly_lib}, f)
np.savez(SAVE_DIR / 'norm_constants.npz',
         x_min=x_min, x_max=x_max, x_den=x_den,
         y_min=y_min, y_max=y_max, y_den=y_den)
print(f"  New sindy_data.pkl   saved → {SAVE_DIR}")
print(f"  New norm_constants   saved → {SAVE_DIR}")
print(f"  New Volume norm range: [{y_min[2]:.1f}, {y_min[2]+y_den[2]:.1f}] mL")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Jacobian comparison
# ══════════════════════════════════════════════════════════════════════════════
# Evaluate at a sweep across each actuator, fixing the other two at midpoint.
# Pressure fixed at 90 mmHg.

P_EVAL  = 90.0   # mmHg
ACT_MID = np.array([224.0, 225.0, 224.0])   # epi, trans, endo midpoints
ACT_LO  = np.array([200.0, 202.0, 200.0])
ACT_HI  = np.array([248.0, 248.0, 248.0])
N_SWEEP = 60
ACT_LABELS = ['Epi (mm)', 'Trans (mm)', 'Endo (mm)']

def jacobian_phys(act_phys, p_phys, pred_fn):
    """Physical-space Jacobian: dY_phys / dU_phys (mm or deg / mm)."""
    y0 = pred_fn(np.append(act_phys, p_phys).reshape(1, -1))[0]
    J  = np.zeros((3, 3))
    for j in range(3):
        ap    = act_phys.copy(); ap[j] += FD_EPS_PHYS
        yp    = pred_fn(np.append(ap, p_phys).reshape(1, -1))[0]
        J[:, j] = (yp - y0) / FD_EPS_PHYS
    return J  # (output oi, actuator j)

print(f"\n  Computing Jacobians at P={P_EVAL} mmHg ...")

# For each actuator axis, sweep its value while others = ACT_MID
sweep_jac_old = {}  # {act_idx: array (N_SWEEP, 3, 3)}
sweep_jac_new = {}
sweep_vals    = {}

for j, label in enumerate(ACT_LABELS):
    vals = np.linspace(ACT_LO[j], ACT_HI[j], N_SWEEP)
    sweep_vals[j] = vals
    J_new_raw = np.zeros((N_SWEEP, 3, 3))
    J_old_raw = np.zeros((N_SWEEP, 3, 3))
    for k, v in enumerate(vals):
        act = ACT_MID.copy(); act[j] = v
        J_new_raw[k] = jacobian_phys(act, P_EVAL, _new_sindy_pred)
        if old_sindy is not None:
            J_old_raw[k] = jacobian_phys(act, P_EVAL, _old_sindy_pred)
    # Gaussian smooth (same as ilcCorrection.py)
    for oi in range(3):
        for ci in range(3):
            J_new_raw[:, oi, ci] = gaussian_filter1d(J_new_raw[:, oi, ci],
                                                      sigma=SINDY_SIGMA, mode='nearest')
            if old_sindy is not None:
                J_old_raw[:, oi, ci] = gaussian_filter1d(J_old_raw[:, oi, ci],
                                                          sigma=SINDY_SIGMA, mode='nearest')
    sweep_jac_new[j] = J_new_raw
    if old_sindy is not None:
        sweep_jac_old[j] = J_old_raw

# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 1 — Volume distribution change in training data
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))

# Data distributions
ax = axes1[0]
bins = np.linspace(30, 175, 40)
for p, col in zip(PRESSURES, plt.cm.viridis(np.linspace(0, 1, len(PRESSURES)))):
    v = data_dict[p]['volume_endo_mL'].values
    ax.hist(v, bins=bins, alpha=0.45, color=col, label=f'{p} mmHg')
ax.axvline(60,  color='k',   ls='--', lw=1.5, label='Target ESV=60 mL')
ax.axvline(120, color='red', ls='--', lw=1.5, label='Target EDV=120 mL')
ax.set_xlabel('Volume (mL)')
ax.set_ylabel('Count')
ax.set_title('Training data: volume distribution per pressure level')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Old vs new norm range bar
ax2 = axes1[1]
categories = ['Twist (deg)', 'Height (mm)', 'Volume (mL)']
idx = np.arange(len(categories))
w   = 0.35
if old_sindy is not None:
    ax2.bar(idx - w/2, old_nrm['y_den'],   width=w, label='Old model range', alpha=0.7, color='steelblue')
ax2.bar(idx + w/2, y_den, width=w, label='New model range', alpha=0.7, color='darkorange')
ax2.set_xticks(idx)
ax2.set_xticklabels(categories)
ax2.set_ylabel('Normalisation range (y_den = y_max - y_min)')
ax2.set_title('Normalisation range: old vs new model')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
if old_sindy is not None:
    for i in range(3):
        ax2.text(i - w/2, old_nrm['y_den'][i] + 0.5,
                 f"{old_nrm['y_min'][i]:.1f}–{old_nrm['y_min'][i]+old_nrm['y_den'][i]:.1f}",
                 ha='center', fontsize=7, color='steelblue')
        ax2.text(i + w/2, y_den[i] + 0.5,
                 f"{y_min[i]:.1f}–{y_min[i]+y_den[i]:.1f}",
                 ha='center', fontsize=7, color='darkorange')

plt.suptitle(f'SINDy training data: {_data_label}', fontsize=12, fontweight='bold')
plt.tight_layout()
fig1_path = os.path.join(BASE, 'sindy_retrain_data_distribution.png')
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
print(f"\n  Figure 1 saved → {fig1_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 2 — Jacobian comparison: old vs new (volume row only + key sens)
# ══════════════════════════════════════════════════════════════════════════════
# Show dVolume/d[epi,trans,endo] and dTwist/d[epi,trans,endo]
# One column per swept actuator axis, two output rows (volume + twist)

SHOW_OUTPUTS = [0, 2]  # [Twist, Volume]
SHOW_LABELS  = ['dTwist/dActuator (deg/mm)', 'dVolume/dActuator (mL/mm)']

fig2, axes2 = plt.subplots(len(SHOW_OUTPUTS), 3, figsize=(15, 5 * len(SHOW_OUTPUTS)), sharex='col')

for col_j in range(3):
    vals = sweep_vals[col_j]
    for row_i, oi in enumerate(SHOW_OUTPUTS):
        ax = axes2[row_i, col_j]
        new_sens = sweep_jac_new[col_j][:, oi, col_j]  # d(output oi) / d(actuator col_j)
        ax.plot(vals, new_sens, color='darkorange', lw=2.2, label='New model')
        if old_sindy is not None:
            old_sens = sweep_jac_old[col_j][:, oi, col_j]
            ax.plot(vals, old_sens, color='steelblue', lw=2.2, ls='--', label='Old model')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(ACT_LABELS[col_j], fontsize=9)
        if col_j == 0:
            ax.set_ylabel(SHOW_LABELS[row_i], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        # title for top row only
        if row_i == 0:
            ax.set_title(f'd/d({ACT_LABELS[col_j]})\n(others held at midpoint, P={P_EVAL} mmHg)',
                         fontsize=9)

plt.suptitle(f'SINDy Jacobian comparison — old vs new model\n'
             f'Pressure = {P_EVAL} mmHg  |  Gaussian smoothed (sigma={SINDY_SIGMA})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig2_path = os.path.join(BASE, 'sindy_retrain_jacobian_comparison.png')
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"  Figure 2 saved → {fig2_path}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. FIGURE 3 — Full Jacobian grid (all 9 sensitivities) for new model
# ══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(3, 3, figsize=(15, 12), sharex='col')
for col_j in range(3):
    vals = sweep_vals[col_j]
    for oi in range(3):
        ax = axes3[oi, col_j]
        new_s = sweep_jac_new[col_j][:, oi, col_j]
        ax.plot(vals, new_s, color='darkorange', lw=2.2, label='New')
        if old_sindy is not None:
            old_s = sweep_jac_old[col_j][:, oi, col_j]
            ax.plot(vals, old_s, color='steelblue', lw=2.2, ls='--', label='Old')
        ax.axhline(0, color='gray', lw=0.8, ls=':')
        ax.set_xlabel(ACT_LABELS[col_j], fontsize=8)
        ax.set_ylabel(f'd{OUTPUT_NAMES[oi]}/d{["epi","trans","endo"][col_j]}', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        if oi == 0:
            ax.set_title(f'Sweep {ACT_LABELS[col_j]}', fontsize=9)

plt.suptitle(f'Full SINDy Jacobian (all 9 sensitivities) — old vs new model  |  P={P_EVAL} mmHg',
             fontsize=11, fontweight='bold')
plt.tight_layout()
fig3_path = os.path.join(BASE, 'sindy_retrain_full_jacobian.png')
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
print(f"  Figure 3 saved → {fig3_path}")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("  RETRAIN COMPLETE")
print(f"{'='*64}")
if old_sindy is not None:
    print(f"  Volume norm range:  OLD [{old_nrm['y_min'][2]:.1f}, "
          f"{old_nrm['y_min'][2]+old_nrm['y_den'][2]:.1f}] mL  →  "
          f"NEW [{y_min[2]:.1f}, {y_min[2]+y_den[2]:.1f}] mL")
    print(f"  dV/dTrans peak (new): {sweep_jac_new[1][:, 2, 1].max():.3f} mL/mm")
    print(f"  dV/dTrans peak (old): {sweep_jac_old[1][:, 2, 1].max():.3f} mL/mm")
print(f"\n  Active files updated:")
print(f"    sindy_data.pkl       → {SAVE_DIR}")
print(f"    norm_constants.npz   → {SAVE_DIR}")
print(f"  Backups:")
print(f"    sindy_data_prev.pkl")
print(f"    norm_constants_prev.npz")

plt.show()
