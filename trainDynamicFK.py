"""
trainDynamicFK.py
──────────────────────────────────────────────────────────────────────────────
Train a SINDy-C (SINDy with Control) dynamic FK model on ILC iteration data.

SINDy-C learns the time evolution of the system states:

  dX/dt = Θ(X, U) · Ξ

  X = [twist, height, volume, pressure]    4 non-trivial states
  U = [epi, trans, endo]                   actuator positions (control inputs)
  Θ = polynomial library of (X, U)         candidate functions
  Ξ = sparse coefficient matrix            identified by STLSQ sparse regression

Pressure is a STATE in this model (not an input) — its time evolution is
learned from reactive pressure measurements in the ILC history data.
This gives ∂(dX/dt)/∂U which includes ∂pressure/∂actuators directly.

Run after ≥ MIN_ITERATIONS of ilcMotionCorrection.py have completed.

Inputs
──────
  ILCFiles/Exp_data/itr0/ILCReadyData.csv
  ILCFiles/Exp_data/itr1/ILCReadyData.csv  ... (one per iteration)
      columns: phase, epi, trans, endo, twist, height, volume, pressure

Outputs
───────
  saved_models/dynamic_sindy_c.pkl
      keys: model, x_scaler, u_scaler, dt, x_cols, u_cols, n_iters_trained
"""

import os, pickle, pathlib, warnings, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import STLSQ

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

POLY_DEGREE         = 2      # polynomial library degree
SPARSITY_THRESHOLD  = 0.005  # STLSQ threshold — lower = keep more terms (important with few data)
MAX_ITER            = 20     # STLSQ max iterations
MIN_ITERATIONS      = 3      # minimum ILC history files required before training
SECONDS_PER_CYCLE   = 20.0   # cardiac cycle period (must match PVTwrite.py)

# State columns (must exist in ILCReadyData.csv)
X_COLS = ['twist', 'height', 'volume', 'pressure']

# Actuator column candidates — first match found in the CSV will be used
_U_CANDIDATES = [['epi', 'trans', 'endo'],
                 ['epi_mm', 'trans_mm', 'endo_mm']]

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE        = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR    = pathlib.Path(BASE) / 'saved_models'
EXP_DATA    = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Exp_data'
OUT_PKL     = SAVE_DIR / 'dynamic_sindy_c.pkl'

assert EXP_DATA.exists(), \
    f"Exp_data folder not found:\n  {EXP_DATA}"

def _itr_num(p):
    m = re.search(r'itr\s*(\d+)', p.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1

iter_files = sorted(
    [p / 'ILCReadyData.csv' for p in EXP_DATA.iterdir()
     if p.is_dir() and (p / 'ILCReadyData.csv').exists()],
    key=lambda p: _itr_num(p.parent)
)
assert len(iter_files) >= MIN_ITERATIONS, \
    f"Only {len(iter_files)} iteration(s) found in {EXP_DATA}\n" \
    f"Need at least {MIN_ITERATIONS} before training."

print(f"trainDynamicFK.py")
print(f"  Found {len(iter_files)} ILC iterations:")
for f in iter_files:
    print(f"    {f.parent.name}/{f.name}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD AND STACK HISTORY DATA
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading history data …")

X_list = []   # list of (T, 4) arrays — one per iteration
U_list = []   # list of (T, 3) arrays — one per iteration
dt_list = []

for fpath in iter_files:
    df = pd.read_csv(fpath)

    # Detect actuator column names on first file
    if 'U_COLS' not in dir():
        for candidates in _U_CANDIDATES:
            if all(c in df.columns for c in candidates):
                U_COLS = candidates
                break
        else:
            raise KeyError(f"No actuator columns found in {fpath}.\n"
                           f"Expected one of {_U_CANDIDATES}\n"
                           f"Got: {list(df.columns)}")
        print(f"  Actuator columns: {U_COLS}")

    # Phase: use existing column or generate uniformly
    if 'phase' in df.columns:
        phase = df['phase'].values
        if phase.max() > 1.5:
            phase = (phase - phase[0]) / (phase[-1] - phase[0])
    else:
        phase = np.linspace(0.0, 1.0, len(df))

    dt_phase = np.mean(np.diff(phase))
    dt = dt_phase * SECONDS_PER_CYCLE
    dt_list.append(dt)

    X = df[X_COLS].values.astype(float)   # (T, 4)
    U = df[U_COLS].values.astype(float)   # (T, 3)

    # Remove last point to avoid boundary finite-difference artifacts
    X_list.append(X[:-1])
    U_list.append(U[:-1])

    print(f"  {fpath.name}: {len(X)-1} samples  dt={dt:.4f}s  "
          f"P=[{df['pressure'].min():.0f},{df['pressure'].max():.0f}] mmHg")

dt = float(np.mean(dt_list))
print(f"\n  Using dt = {dt:.4f} s  ({SECONDS_PER_CYCLE} s / cycle)")

# ── Normalise X and U with StandardScaler (zero mean, unit variance) ──────────
# Fit scalers on ALL data combined so normalisation is consistent across iterations
X_all_raw = np.vstack(X_list)   # (N_total, 4)
U_all_raw = np.vstack(U_list)   # (N_total, 3)

sc_x = StandardScaler()
sc_u = StandardScaler()
sc_x.fit(X_all_raw)
sc_u.fit(U_all_raw)

# Normalise each iteration separately (preserves trajectory boundaries)
X_norm_list = [sc_x.transform(X) for X in X_list]
U_norm_list = [sc_u.transform(U) for U in U_list]

print(f"\n  Normalisation (StandardScaler):")
for col, mean, std in zip(X_COLS, sc_x.mean_, sc_x.scale_):
    print(f"    {col:<10}: mean={mean:.3f}  std={std:.3f}")
for col, mean, std in zip(U_COLS, sc_u.mean_, sc_u.scale_):
    print(f"    {col:<10}: mean={mean:.3f}  std={std:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# FIT SINDY-C
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nFitting SINDy-C (degree={POLY_DEGREE}, threshold={SPARSITY_THRESHOLD}) …")

lib   = PolynomialLibrary(degree=POLY_DEGREE, include_bias=True)
opt   = STLSQ(threshold=SPARSITY_THRESHOLD, max_iter=MAX_ITER)
model = SINDy(feature_library=lib, optimizer=opt)

# Concatenate all iterations and fit as one long trajectory
X_norm_all = np.concatenate(X_norm_list, axis=0)
U_norm_all = np.concatenate(U_norm_list, axis=0)
model.fit(X_norm_all, u=U_norm_all, t=dt)

print("\n  SINDy-C equations (normalised space):")
model.print()

# Count non-zero terms per state
coef = model.coefficients()   # (n_states, n_features)
n_terms = np.sum(np.abs(coef) > 1e-10, axis=1)
print(f"\n  Non-zero terms per state:")
for col, n in zip(X_COLS, n_terms):
    print(f"    d({col})/dt : {n} terms")

# ══════════════════════════════════════════════════════════════════════════════
# RESIDUAL EVALUATION ON TRAINING DATA
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nResiduals on training data:")

all_rmse = {col: [] for col in X_COLS}

for k, (X_n, U_n, X_raw) in enumerate(zip(X_norm_list, U_norm_list, X_list)):
    dX_pred_n = model.predict(X_n, u=U_n)                     # (T, 4) normalised dX/dt

    # Compute actual dX/dt via finite differences on physical data
    dX_actual_phys = np.diff(X_raw, axis=0) / dt              # (T-1, 4)
    dX_pred_phys   = dX_pred_n[:-1] * sc_x.scale_             # scale back to physical

    for ci, col in enumerate(X_COLS):
        rmse = np.sqrt(np.mean((dX_actual_phys[:, ci] - dX_pred_phys[:, ci])**2))
        all_rmse[col].append(rmse)

print(f"  {'State':<12}  {'Mean RMSE':>12}  {'Std RMSE':>10}  units/s")
for col in X_COLS:
    unit = {'twist': 'deg/s', 'height': 'mm/s',
            'volume': 'mL/s', 'pressure': 'mmHg/s'}[col]
    print(f"  {col:<12}  {np.mean(all_rmse[col]):>12.4f}  "
          f"{np.std(all_rmse[col]):>10.4f}  {unit}")

# ══════════════════════════════════════════════════════════════════════════════
# LEAVE-ONE-OUT CROSS-VALIDATION  (only if ≥ 3 iterations)
# ══════════════════════════════════════════════════════════════════════════════
if len(iter_files) >= 3:
    print(f"\nLeave-one-out cross-validation ({len(iter_files)} folds) …")
    loo_pressure_rmse = []

    for k in range(len(X_norm_list)):
        # Train on all except iteration k
        X_train = [X_norm_list[i] for i in range(len(X_norm_list)) if i != k]
        U_train = [U_norm_list[i] for i in range(len(U_norm_list)) if i != k]
        X_test  = X_list[k]
        U_test  = U_list[k]

        m_loo = SINDy(feature_library=PolynomialLibrary(degree=POLY_DEGREE),
                      optimizer=STLSQ(threshold=SPARSITY_THRESHOLD, max_iter=MAX_ITER))
        m_loo.fit(np.concatenate(X_train, axis=0),
                  u=np.concatenate(U_train, axis=0), t=dt)

        # Open-loop rollout on held-out iteration
        n_steps = len(X_test)
        X_sim_n = np.zeros((n_steps, len(X_COLS)))
        X_sim_n[0] = sc_x.transform(X_test[0:1])[0]
        U_test_n = sc_u.transform(U_test)

        diverged = False
        for i in range(n_steps - 1):
            dX_n = m_loo.predict(X_sim_n[i:i+1], u=U_test_n[i:i+1])[0]
            X_sim_n[i+1] = np.clip(X_sim_n[i] + dX_n * dt, -5.0, 5.0)
            if not np.all(np.isfinite(X_sim_n[i+1])):
                X_sim_n[i+1:] = np.nan
                diverged = True
                break

        if diverged:
            print(f"    Fold {k+1} ({iter_files[k].parent.name}): rollout diverged — skipped")
            continue

        X_sim_phys = sc_x.inverse_transform(X_sim_n)
        p_idx      = X_COLS.index('pressure')
        p_rmse     = np.sqrt(np.nanmean((X_sim_phys[:, p_idx] - X_test[:, p_idx])**2))
        loo_pressure_rmse.append(p_rmse)
        print(f"    Fold {k+1} ({iter_files[k].parent.name}): "
              f"pressure rollout RMSE = {p_rmse:.2f} mmHg")

    print(f"  Mean LOO pressure RMSE: {np.mean(loo_pressure_rmse):.2f} mmHg  "
          f"(std {np.std(loo_pressure_rmse):.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# OPEN-LOOP PRESSURE ROLLOUT  (last iteration as showcase)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nOpen-loop rollout on last iteration …")

X_show   = X_list[-1]
U_show_n = U_norm_list[-1]
n_steps  = len(X_show)

X_sim_n = np.zeros((n_steps, len(X_COLS)))
X_sim_n[0] = sc_x.transform(X_show[0:1])[0]

rollout_diverged = False
for i in range(n_steps - 1):
    dX_n = model.predict(X_sim_n[i:i+1], u=U_show_n[i:i+1])[0]
    X_sim_n[i+1] = np.clip(X_sim_n[i] + dX_n * dt, -5.0, 5.0)
    if not np.all(np.isfinite(X_sim_n[i+1])):
        X_sim_n[i+1:] = np.nan
        rollout_diverged = True
        print("  WARNING: rollout diverged — model may need more training data "
              "or a lower POLY_DEGREE")
        break

X_sim_phys = sc_x.inverse_transform(X_sim_n)

p_idx   = X_COLS.index('pressure')
p_actual = X_show[:, p_idx]
p_pred   = X_sim_phys[:, p_idx]
p_rmse   = np.sqrt(np.mean((p_actual - p_pred)**2))
print(f"  Pressure rollout RMSE = {p_rmse:.2f} mmHg")

# Phase for x-axis
_df_show = pd.read_csv(iter_files[-1])
if 'phase' in _df_show.columns:
    phase_show = _df_show['phase'].values[:-1]
else:
    phase_show = np.linspace(0.0, 1.0, len(_df_show) - 1)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════
payload = {
    'model':            model,
    'x_scaler':         sc_x,
    'u_scaler':         sc_u,
    'dt':               dt,
    'x_cols':           X_COLS,
    'u_cols':           U_COLS,
    'poly_degree':      POLY_DEGREE,
    'n_iters_trained':  len(iter_files),
}

with open(OUT_PKL, 'wb') as f:
    pickle.dump(payload, f)

print(f"\n  Dynamic FK model saved → saved_models/dynamic_sindy_c.pkl")
print(f"  Trained on {len(iter_files)} ILC iterations  "
      f"({X_all_raw.shape[0]} total samples)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures …")

# ── Figure 1: Open-loop rollout vs actual ─────────────────────────────────────
units = {'twist': 'deg', 'height': 'mm', 'volume': 'mL', 'pressure': 'mmHg'}
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
axes1 = axes1.ravel()

for ci, col in enumerate(X_COLS):
    ax = axes1[ci]
    ax.plot(phase_show, X_show[:, ci],       'b-',  lw=2.0, label='Actual')
    ax.plot(phase_show, X_sim_phys[:, ci],   'r--', lw=1.8, label='SINDy-C rollout')
    rmse_val = np.sqrt(np.mean((X_show[:, ci] - X_sim_phys[:, ci])**2))
    ax.set_ylabel(f'{col} ({units[col]})', fontsize=10)
    ax.set_title(f'RMSE = {rmse_val:.3f} {units[col]}', fontsize=9, color='grey')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    if ci >= 2:
        ax.set_xlabel('Cycle phase', fontsize=10)

plt.suptitle(f'SINDy-C Open-Loop Rollout — Last Iteration  '
             f'(degree={POLY_DEGREE}, threshold={SPARSITY_THRESHOLD})',
             fontsize=12)
plt.tight_layout()
fig1.savefig(os.path.join(BASE, 'dynamicFK_rollout.png'), dpi=150, bbox_inches='tight')
print(f"  Rollout figure → dynamicFK_rollout.png")

# ── Figure 2: Coefficient heatmap (sparsity pattern) ─────────────────────────
fig2, ax2 = plt.subplots(figsize=(max(12, coef.shape[1]//2), 4))
feat_names = model.get_feature_names()
im = ax2.imshow(np.abs(coef), aspect='auto', cmap='viridis',
                norm=mcolors.LogNorm(vmin=1e-4, vmax=np.abs(coef).max() + 1e-9))
ax2.set_yticks(range(len(X_COLS))); ax2.set_yticklabels([f'd({c})/dt' for c in X_COLS])
ax2.set_xticks(range(len(feat_names)))
ax2.set_xticklabels(feat_names, rotation=90, fontsize=7)
ax2.set_title('SINDy-C Coefficient Magnitude  (log scale — bright = large term)',
              fontsize=11)
plt.colorbar(im, ax=ax2, label='|coefficient|')
plt.tight_layout()
fig2.savefig(os.path.join(BASE, 'dynamicFK_coefficients.png'), dpi=150, bbox_inches='tight')
print(f"  Coefficient figure → dynamicFK_coefficients.png")

# ── Figure 3: Pressure trajectory across all iterations ──────────────────────
fig3, ax3 = plt.subplots(figsize=(12, 5))
colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(X_list)))
def _load_phase(f):
    _d = pd.read_csv(f)
    if 'phase' in _d.columns:
        return _d['phase'].values[:-1]
    return np.linspace(0.0, 1.0, len(_d) - 1)

for k, (X_raw, phase_f) in enumerate(zip(X_list, [_load_phase(f) for f in iter_files])):
    ax3.plot(phase_f, X_raw[:, p_idx],
             color=colors[k], lw=1.5,
             label=f'Iter {k+1} actual')
ax3.plot(phase_show, p_pred, 'r--', lw=2.0, label='SINDy-C rollout (iter last)')
ax3.set_xlabel('Cycle phase', fontsize=10)
ax3.set_ylabel('Pressure (mmHg)', fontsize=10)
ax3.set_title('Pressure across ILC iterations  +  SINDy-C prediction', fontsize=11)
ax3.legend(fontsize=8, loc='upper right'); ax3.grid(True, alpha=0.3)
plt.tight_layout()
fig3.savefig(os.path.join(BASE, 'dynamicFK_pressure_history.png'), dpi=150, bbox_inches='tight')
print(f"  Pressure history figure → dynamicFK_pressure_history.png")

plt.show()

print(f"\n  Training complete.")
print(f"  Next step: run ilcPressureCorrection.py")
