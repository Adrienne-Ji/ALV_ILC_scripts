"""
mpc_ilc.py
──────────────────────────────────────────────────────────────────────────────
Iterative Learning Control (ILC) for the artificial heart simulator.

Inputs
──────
  sharedCSVs/ILCReadyData.csv  — single pre-averaged cycle (produced externally:
      warmup cycles discarded, stable cycles averaged, resampled to uniform phase).
      Auto-detected columns: phase (or time), epi, trans, endo,
                             twist, height, volume, pressure (optional).

  engineered_data_withP.csv  — desired 1-cycle trajectory with pressure profile.

Workflow
────────
  1. Load ILCReadyData.csv — already one averaged stable cycle, no segmentation needed.
  2. Resample onto the desired-trajectory phase grid [0, 1].
  3. Compute tracking error: e(t) = y_desired(t) - y_actual(t)
     Hard objectives: Twist, Height, Volume.
     Pressure: soft boundary check only — reactive system, not tracked.
  4. Q-filtered P-type ILC update:
       v_k   = u_k + α · J(t)⁺ · e_norm(t)     ← P-type step
       u_k+1 = Q[ v_k ]                          ← low-pass Q-filter
       u_k+1 = clip(u_k+1, ACT_MIN, ACT_MAX)
  5. Save ilc_corrected_actuators.csv → feed into Zaber control script.

Run pressure_fk_comparison.py first to populate saved_models/.
"""

import os, pickle, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from pysindy.feature_library import PolynomialLibrary

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS — change these for each trial
# ══════════════════════════════════════════════════════════════════════════════

# Pre-processed single-cycle CSV (averaged over stable cycles, produced externally).
# Place in sharedCSVs/ with the name below, or override the full path.
# Expected columns (auto-detected by name — see COLUMN MAPPING below):
#   phase or time  — cycle phase [0,1] or time within cycle (s)
#   epi, trans, endo — actuator positions (mm)
#   twist, height, volume — measured deformation outputs
#   pressure (optional) — actual measured pressure (mmHg); used for soft boundary
#                          check and Jacobian conditioning if present
ILC_READY_CSV = None   # set to full path, or leave None to use sharedCSVs/ILCReadyData.csv

# ILC learning gain (0 < α ≤ 1; start conservative)
ILC_ALPHA  = 0.5

# Q-filter settings (zero-phase low-pass Butterworth applied to full updated signal)
# Q_CUTOFF: normalised cutoff (0–1 fraction of Nyquist). Lower = smoother, slower.
# Q_ORDER:  filter order (3 is a good default)
Q_CUTOFF   = 0.3
Q_ORDER    = 3

# Physical actuator bounds (mm)
ACT_MIN = np.array([200.0, 202.0, 200.0])
ACT_MAX = np.array([248.0, 248.0, 248.0])

# Height/volume offsets applied to desired trajectory to align with training data
HEIGHT_OFFSET = 70.0    # mm
VOLUME_OFFSET = 0.0     # mL

# FK architecture for ILC Jacobian: 'DataDriven', 'PINN', 'SINDy', 'SINDy2'
ILC_MODEL = 'SINDy'

# ── Objective formulation ──────────────────────────────────────────────────────
# HARD objectives (strict ILC tracking): Twist, Height, Volume.
#   Error and Jacobian correction act ONLY on these three geometric outputs.
# SOFT constraint (boundary check only): Pressure.
#   Pressure is reactive — it cannot be independently commanded. The script
#   warns if actual pressure leaves the clinical window but does NOT include
#   it in the ILC correction signal.
P_SOFT_MIN = 0.0     # mmHg — lower clinical pressure bound
P_SOFT_MAX = 125.0   # mmHg — upper clinical pressure bound

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE        = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR    = pathlib.Path(BASE) / 'saved_models'
PYTHONCODES = os.path.join(BASE, '..')
ENG_CSV     = os.path.join(PYTHONCODES, 'engineered_data_withP.csv')

assert SAVE_DIR.exists(), \
    "Run pressure_fk_comparison.py first to generate saved_models/"

# ── Load normalisation constants ───────────────────────────────────────────────
norm  = np.load(SAVE_DIR / 'norm_constants.npz')
x_min = norm['x_min']; x_den = norm['x_den']
y_min = norm['y_min']; y_den = norm['y_den']

def norm_act(act_phys):
    return (np.asarray(act_phys) - x_min[:3]) / x_den[:3]

def denorm_act(act_n):
    return np.asarray(act_n) * x_den[:3] + x_min[:3]

def norm_y(y_phys):
    return (np.asarray(y_phys) - y_min) / y_den

def denorm_y(y_n):
    return np.asarray(y_n) * y_den + y_min

# ── Load desired trajectory ────────────────────────────────────────────────────
eng        = pd.read_csv(ENG_CSV)
traj_time  = eng['time'].values
traj_p     = eng['pressure'].values
traj_twist = eng['twist'].values
traj_h     = eng['height'].values + HEIGHT_OFFSET
traj_v     = eng['volume'].values + VOLUME_OFFSET
traj_phys  = np.stack([traj_twist, traj_h, traj_v], axis=1)   # (T, 3)
traj_norm  = norm_y(traj_phys)                                  # (T, 3)
pressure_n = (traj_p - x_min[3]) / x_den[3]

# Normalise traj time to [0, 1] for cycle-phase alignment
traj_phase = (traj_time - traj_time[0]) / (traj_time[-1] - traj_time[0])
n_traj     = len(traj_time)

OUTPUT_NAMES = ['Twist_deg', 'Height_mm', 'Volume_mL']
OUTPUT_UNITS = ['deg', 'mm', 'mL']

print(f"Desired trajectory: {n_traj} pts  "
      f"pressure=[{traj_p.min():.0f}, {traj_p.max():.0f}] mmHg")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD FK MODEL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading FK model: {ILC_MODEL} …")
POLY_DEGREE = 3

class DataDrivenFK(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 3))
    def forward(self, x): return self.net(x)

class PressureConditionedFK(nn.Module):
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
        y_nom = self.act_branch(actuators)
        cond  = self.pressure_branch(pressure)
        twist  = torch.sigmoid(cond[:, 0]) * y_nom[:, 0] + cond[:, 1]
        height = torch.sigmoid(cond[:, 3]) * y_nom[:, 1] + cond[:, 2]
        volume = torch.sigmoid(cond[:, 5]) * y_nom[:, 2] + cond[:, 4]
        return torch.stack([twist, height, volume], dim=1)

if ILC_MODEL in ('DataDriven', 'PINN'):
    if ILC_MODEL == 'DataDriven':
        model = DataDrivenFK()
        model.load_state_dict(torch.load(SAVE_DIR / 'dd_model.pth', weights_only=True))
    else:
        model = PressureConditionedFK()
        model.load_state_dict(torch.load(SAVE_DIR / 'pinn_model.pth', weights_only=True))
    model.eval()
    model.requires_grad_(True)

    def fk_predict_norm(act_n_batch, p_n_batch):
        """(N,3) normalised act, (N,) normalised pressure → (N,3) normalised output."""
        act_t = torch.tensor(act_n_batch, dtype=torch.float32)
        p_t   = torch.tensor(p_n_batch,   dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            if ILC_MODEL == 'PINN':
                return model(act_t, p_t).numpy()
            else:
                x = torch.cat([act_t, p_t], dim=1)
                return model(x).numpy()

    def jacobian_at(act_n, p_n):
        """3×3 Jacobian ∂FK_norm/∂act_norm via autograd at one point."""
        act_t = torch.tensor(act_n, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
        p_t   = torch.tensor([[p_n]], dtype=torch.float32)
        if ILC_MODEL == 'PINN':
            y = model(act_t, p_t)
        else:
            y = model(torch.cat([act_t, p_t], dim=1))
        J = np.zeros((3, 3))
        for oi in range(3):
            if act_t.grad is not None: act_t.grad.zero_()
            y[0, oi].backward(retain_graph=True)
            J[oi, :] = act_t.grad[0].numpy()
        return J

else:
    # SINDy / SINDy2
    with open(SAVE_DIR / 'sindy_data.pkl',  'rb') as f: sindy_data  = pickle.load(f)
    with open(SAVE_DIR / 'sindy2_data.pkl', 'rb') as f: sindy2_data = pickle.load(f)
    sindy_results = sindy_data['results']
    sindy_sc      = sindy_data['sc']
    poly_lib      = sindy_data['poly_lib']
    s2_stage1     = sindy2_data['stage1']
    s2_stage2     = sindy2_data['stage2']
    s2_sc_in1     = sindy2_data['sc_in1']
    poly_lib1     = sindy2_data['poly_lib1']
    s2_sc_p       = sindy2_data['sc_p']

    def _sindy2_stage1(act_phys):
        x_s = s2_sc_in1.transform(np.asarray(act_phys).reshape(-1, 3))
        theta = np.asarray(poly_lib1.transform(x_s))
        outs = []
        for col in OUTPUT_NAMES:
            r = s2_stage1[col]
            outs.append(r['sy'].inverse_transform((theta @ r['coef']).reshape(-1,1)).ravel())
        return np.stack(outs, axis=1)

    def _build_mult_basis(base_col, p_s):
        scale_cols  = [base_col * (p_s ** k) for k in range(POLY_DEGREE + 1)]
        offset_cols = [(p_s ** k)             for k in range(POLY_DEGREE + 1)]
        return np.column_stack(scale_cols + offset_cols)

    def _sindy_predict_phys(x_physical):
        x_s   = sindy_sc.transform(np.asarray(x_physical).reshape(-1, 4))
        theta = np.asarray(poly_lib.transform(x_s))
        outs  = []
        for col in OUTPUT_NAMES:
            r = sindy_results[col]
            outs.append(r['sy'].inverse_transform((theta @ r['coef']).reshape(-1,1)).ravel())
        return np.stack(outs, axis=1)

    def _sindy2_predict_phys(x_physical):
        x_physical = np.asarray(x_physical).reshape(-1, 4)
        act_phys   = x_physical[:, :3]
        p_raw      = x_physical[:, 3]
        base       = _sindy2_stage1(act_phys)
        p_s        = s2_sc_p.transform(p_raw.reshape(-1,1)).ravel()
        outs = []
        for i, col in enumerate(OUTPUT_NAMES):
            r     = s2_stage2[col]
            Theta = _build_mult_basis(base[:, i], p_s)
            Theta_sc = r['col_sc'].transform(Theta)
            outs.append(r['sy'].inverse_transform((Theta_sc @ r['coef']).reshape(-1,1)).ravel())
        return np.stack(outs, axis=1)

    _pred_fn = _sindy_predict_phys if ILC_MODEL == 'SINDy' else _sindy2_predict_phys
    FD_EPS   = 1e-4

    def fk_predict_norm(act_n_batch, p_n_batch):
        act_phys = denorm_act(act_n_batch)
        p_phys   = p_n_batch * x_den[3] + x_min[3]
        x_phys   = np.hstack([act_phys, p_phys.reshape(-1,1)])
        return norm_y(_pred_fn(x_phys))

    def jacobian_at(act_n, p_n):
        act_phys = denorm_act(act_n)
        p_phys   = p_n * x_den[3] + x_min[3]
        def fwd(a_n):
            a_p  = denorm_act(a_n)
            x_p  = np.append(a_p, p_phys).reshape(1,-1)
            return norm_y(_pred_fn(x_p)[0])
        f0 = fwd(act_n)
        J  = np.zeros((3, 3))
        for j in range(3):
            ap = act_n.copy(); ap[j] += FD_EPS
            J[:, j] = (fwd(ap) - f0) / FD_EPS
        return J

print("  FK model ready.")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ILC-READY DATA  (single pre-averaged cycle from sharedCSVs)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading ILC-ready data …")

_ilc_path = ILC_READY_CSV or os.path.join(
    BASE, '..', '..', 'sharedCSVs', 'ILCReadyData.csv')
assert os.path.exists(_ilc_path), \
    f"ILCReadyData.csv not found at:\n  {_ilc_path}\n" \
    f"Set ILC_READY_CSV to the correct path."

ilc_df = pd.read_csv(_ilc_path)
ilc_df.columns = ilc_df.columns.str.strip()
print(f"  Loaded {len(ilc_df)} rows — columns: {list(ilc_df.columns)}")

# ── Column auto-detection ─────────────────────────────────────────────────────
def _find_col(df, candidates):
    """Return the first candidate name (case-insensitive) found in df, or None."""
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

_COL_PHASE   = _find_col(ilc_df, ['phase', 'cycle_phase', 'time', 'time_s'])
_COL_EPI     = _find_col(ilc_df, ['epi_mm', 'epi', 'Epi_mm', 'Epi', 'u1', 'Position_U1'])
_COL_TRANS   = _find_col(ilc_df, ['trans_mm', 'trans', 'Trans_mm', 'Trans', 'u2', 'Position_U2'])
_COL_ENDO    = _find_col(ilc_df, ['endo_mm', 'endo', 'Endo_mm', 'Endo', 'u3', 'Position_U3'])
_COL_TWIST   = _find_col(ilc_df, ['twist', 'twist_deg', 'dtwist_deg', 'Twist'])
_COL_HEIGHT  = _find_col(ilc_df, ['height', 'height_mm', 'Height'])
_COL_VOLUME  = _find_col(ilc_df, ['volume', 'volume_mL', 'volume_endo_mL', 'Volume'])
_COL_PRES    = _find_col(ilc_df, ['pressure', 'pressure_mmhg', 'pressure_meas', 'Pressure'])

_missing = [name for name, col in [('epi', _COL_EPI), ('trans', _COL_TRANS),
            ('endo', _COL_ENDO), ('twist', _COL_TWIST),
            ('height', _COL_HEIGHT), ('volume', _COL_VOLUME)] if col is None]
assert not _missing, \
    f"Could not find columns for: {_missing}\nAvailable: {list(ilc_df.columns)}"

print(f"  Column mapping:")
if _COL_PHASE:
    print(f"    phase/time → '{_COL_PHASE}'")
else:
    print(f"    phase/time → not found — generated from row index (uniform spacing)")
print(f"    epi        → '{_COL_EPI}'")
print(f"    trans      → '{_COL_TRANS}'")
print(f"    endo       → '{_COL_ENDO}'")
print(f"    twist      → '{_COL_TWIST}'")
print(f"    height     → '{_COL_HEIGHT}'")
print(f"    volume     → '{_COL_VOLUME}'")
print(f"    pressure   → '{_COL_PRES}'" if _COL_PRES else "    pressure   → not found (will use desired profile)")

# ── Extract arrays and resample onto desired trajectory phase grid ─────────────
if _COL_PHASE is not None:
    _phase_raw = ilc_df[_COL_PHASE].values
    if _phase_raw.max() > 1.5:   # looks like time, not phase — normalise
        _phase_raw = (_phase_raw - _phase_raw[0]) / (_phase_raw[-1] - _phase_raw[0])
else:
    # No time/phase column — rows are assumed uniformly spaced over one cycle
    _phase_raw = np.linspace(0.0, 1.0, len(ilc_df))

def _interp(src_phase, values, dst_phase):
    return interp1d(src_phase, values,
                    kind='linear', bounds_error=False,
                    fill_value='extrapolate')(dst_phase)

act_phys_avg = np.column_stack([
    _interp(_phase_raw, ilc_df[_COL_EPI].values,   traj_phase),
    _interp(_phase_raw, ilc_df[_COL_TRANS].values, traj_phase),
    _interp(_phase_raw, ilc_df[_COL_ENDO].values,  traj_phase),
])   # (T, 3) mm

y_phys_avg = np.column_stack([
    _interp(_phase_raw, ilc_df[_COL_TWIST].values,  traj_phase),
    _interp(_phase_raw, ilc_df[_COL_HEIGHT].values, traj_phase),
    _interp(_phase_raw, ilc_df[_COL_VOLUME].values, traj_phase),
])   # (T, 3)

# ── Pressure: Jacobian conditioning + soft boundary check ─────────────────────
if _COL_PRES is not None:
    actual_p   = _interp(_phase_raw, ilc_df[_COL_PRES].values, traj_phase)
    actual_p_n = (actual_p - x_min[3]) / x_den[3]
    _p_rmse    = np.sqrt(np.mean((actual_p - traj_p) ** 2))
    print(f"\n  Pressure (SOFT constraint — boundary check only, not tracked):")
    print(f"    Desired mean : {traj_p.mean():.1f} mmHg")
    print(f"    Actual  mean : {actual_p.mean():.1f} mmHg   RMSE vs desired = {_p_rmse:.2f} mmHg")
    _p_viol = np.sum((actual_p < P_SOFT_MIN) | (actual_p > P_SOFT_MAX))
    if _p_viol == 0:
        print(f"    Boundary [{P_SOFT_MIN:.0f}, {P_SOFT_MAX:.0f}] mmHg : OK — all samples within bounds")
    else:
        _p_frac = 100 * _p_viol / len(actual_p)
        print(f"    *** BOUNDARY VIOLATION: {_p_viol}/{len(actual_p)} samples ({_p_frac:.1f}%) "
              f"outside [{P_SOFT_MIN:.0f}, {P_SOFT_MAX:.0f}] mmHg ***")
    print(f"    Jacobian will use ACTUAL measured pressure.")
else:
    actual_p   = traj_p
    actual_p_n = pressure_n
    print(f"\n  Pressure (SOFT constraint): column not found — Jacobian using desired profile.")

print(f"\n  ILC hard objectives : Twist, Height, Volume")
print(f"  ILC soft constraint : Pressure [{P_SOFT_MIN:.0f}, {P_SOFT_MAX:.0f}] mmHg (boundary check only)")

# ══════════════════════════════════════════════════════════════════════════════
# ILC UPDATE
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing ILC correction …")

# Error in physical and normalised space
e_phys = traj_phys - y_phys_avg         # (T, 3)
e_norm = e_phys / y_den                  # (T, 3) normalised error

act_n_avg = norm_act(act_phys_avg)       # (T, 3) normalised actuators

# RMSE before correction
print(f"\n  Tracking error BEFORE ILC:")
for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    rmse = np.sqrt(np.mean(e_phys[:, ci]**2))
    print(f"    {col:<14}: RMSE = {rmse:.4f} {unit}")

# ── Step 1: P-type update  v_k = u_k + α · J⁺ · e_k ─────────────────────────
delta_u_n = np.zeros_like(act_n_avg)
for i in range(n_traj):
    J      = jacobian_at(act_n_avg[i], actual_p_n[i])   # (3, 3) — uses actual pressure
    J_pinv = np.linalg.pinv(J)                           # (3, 3) pseudoinverse
    delta_u_n[i] = ILC_ALPHA * J_pinv @ e_norm[i]

v_k = act_n_avg + delta_u_n   # (T, 3) — raw P-type updated signal

# ── Step 2: Q-filter  u_{k+1} = Q[v_k]  (zero-phase low-pass) ───────────────
# filtfilt requires at least padlen+1 samples; n_traj is always >> that.
b_q, a_q  = butter(Q_ORDER, Q_CUTOFF, btype='low')
act_n_new = filtfilt(b_q, a_q, v_k, axis=0)   # (T, 3) — filtered signal

print(f"\n  Q-filter applied  (order={Q_ORDER}, cutoff={Q_CUTOFF})")
print(f"  Max correction before filter: {np.abs(delta_u_n).max():.4f} (normalised)")
print(f"  Max correction after  filter: {np.abs(act_n_new - act_n_avg).max():.4f} (normalised)")

# ── Step 3: Denormalise and clip to physical bounds ───────────────────────────
act_phys_new = denorm_act(act_n_new)
act_phys_new = np.clip(act_phys_new,
                       ACT_MIN.reshape(1, -1),
                       ACT_MAX.reshape(1, -1))

# Verify correction with FK model
y_n_corrected = fk_predict_norm(norm_act(act_phys_new), pressure_n)
y_phys_corrected = denorm_y(y_n_corrected)
e_corrected = traj_phys - y_phys_corrected

print(f"\n  Predicted tracking error AFTER ILC (FK model estimate):")
for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    rmse_before = np.sqrt(np.mean(e_phys[:, ci]**2))
    rmse_after  = np.sqrt(np.mean(e_corrected[:, ci]**2))
    improvement = (rmse_before - rmse_after) / rmse_before * 100
    print(f"    {col:<14}: {rmse_before:.4f} → {rmse_after:.4f} {unit}  "
          f"({improvement:+.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE CORRECTED ACTUATOR CSV
# ══════════════════════════════════════════════════════════════════════════════
# Save to sharedCSVs so PVTwrite.py can pick it up directly.
# Column names 'epi', 'trans', 'endo' match PVTwrite.py's expected headers.
SHARED = os.path.join(BASE, '..', '..', 'sharedCSVs')
out_path = os.path.join(SHARED, 'ilc_corrected_actuators.csv')
df_out = pd.DataFrame({
    'phase':         traj_phase,
    'time_in_cycle': traj_time,
    'pressure_mmhg': traj_p,
    'epi':           act_phys_new[:, 0],
    'trans':         act_phys_new[:, 1],
    'endo':          act_phys_new[:, 2],
    'pred_twist_deg':  y_phys_corrected[:, 0],
    'pred_height_mm':  y_phys_corrected[:, 1],
    'pred_volume_mL':  y_phys_corrected[:, 2],
})
df_out.to_csv(out_path, index=False)
print(f"\n  Corrected actuators saved → sharedCSVs/ilc_corrected_actuators.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures …")

fig, axes = plt.subplots(4, 2, figsize=(16, 15))

for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    # Left column: deformation tracking
    ax = axes[ci, 0]
    ax.plot(traj_phase, traj_phys[:, ci],     'k-',  lw=2.5, label='Desired')
    ax.plot(traj_phase, y_phys_avg[:, ci],    'b--', lw=1.8, label='Actual (ILCReadyData)')
    ax.plot(traj_phase, y_phys_corrected[:, ci], 'r-', lw=1.8,
            label=f'Predicted after ILC  (FK estimate)')
    ax.set_ylabel(f'{col} ({unit})', fontsize=10)
    ax.set_xlabel('Cycle phase', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right column: actuator comparison
    ax2 = axes[ci, 1]
    motor_name = ['Epi', 'Trans', 'Endo'][ci]
    ax2.plot(traj_phase, act_phys_avg[:, ci],  'b--', lw=1.8, label='Current (avg)')
    ax2.plot(traj_phase, act_phys_new[:, ci],  'r-',  lw=1.8, label='ILC corrected')
    ax2.axhline(ACT_MIN[ci], color='grey', lw=1, ls=':', label='Bounds')
    ax2.axhline(ACT_MAX[ci], color='grey', lw=1, ls=':')
    ax2.set_ylabel(f'{motor_name} position (mm)', fontsize=10)
    ax2.set_xlabel('Cycle phase', fontsize=9)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# Row 3: Pressure — soft boundary check panel
ax_p = axes[3, 0]
ax_p.plot(traj_phase, traj_p,    'k--', lw=1.5, label='Desired pressure profile')
ax_p.plot(traj_phase, actual_p,  'b-',  lw=1.8, label='Actual pressure (avg cycles)')
ax_p.axhspan(P_SOFT_MIN, P_SOFT_MAX, color='green', alpha=0.08, label='Clinical bounds')
ax_p.axhline(P_SOFT_MIN, color='green', lw=1.2, ls='--')
ax_p.axhline(P_SOFT_MAX, color='green', lw=1.2, ls='--')
_viol_mask = (actual_p < P_SOFT_MIN) | (actual_p > P_SOFT_MAX)
if _viol_mask.any():
    ax_p.scatter(traj_phase[_viol_mask], actual_p[_viol_mask],
                 color='red', s=18, zorder=5, label='Boundary violation')
ax_p.set_ylabel('Pressure (mmHg)', fontsize=10)
ax_p.set_xlabel('Cycle phase', fontsize=9)
ax_p.set_title('Pressure — SOFT constraint (boundary check only, not tracked)',
               fontsize=10, color='grey')
ax_p.legend(fontsize=8); ax_p.grid(True, alpha=0.3)

# Hide the unused 4th-row right panel
axes[3, 1].axis('off')

axes[0, 0].set_title('Deformation tracking  [HARD objectives]', fontsize=11)
axes[0, 1].set_title('Actuator signals', fontsize=11)

plt.suptitle(f'Q-filtered P-type ILC  (α={ILC_ALPHA}, Q_cutoff={Q_CUTOFF}, '
             f'model={ILC_MODEL})', fontsize=13)
plt.tight_layout()
fig_path = os.path.join(BASE, 'ilc_update.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Figure saved → ilc_update.png")
plt.show()
