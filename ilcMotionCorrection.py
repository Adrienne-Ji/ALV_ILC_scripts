"""
ilcMotionCorrection.py
──────────────────────────────────────────────────────────────────────────────
Phase 1 ILC — corrects geometric tracking error (twist, height, volume).

Pressure is recorded passively each iteration and saved to ilc_history/ as
training data for the dynamic SINDy-C FK model (trainDynamicFK.py).

Run repeatedly until geometry RMSE falls below GEOMETRY_RMSE_THRESHOLD,
then proceed to:
  Step 1 →  trainDynamicFK.py       (train dynamic FK on accumulated history)
  Step 2 →  ilcPressureCorrection.py (correct geometry + pressure together)

Inputs
──────
  sharedCSVs/ILCReadyData.csv         — averaged stable cycle from markerProcessing.py
                                         columns (auto-detected): phase/time,
                                         epi, trans, endo, twist, height, volume, pressure
  engineered_data_withP.csv           — desired 1-cycle trajectory
  saved_models/norm_constants.npz     — normalisation from FK training
  saved_models/sindy_data.pkl etc.    — static FK model weights

Outputs
───────
  sharedCSVs/ilc_corrected_actuators.csv      — corrected actuators for PVTwrite.py
  saved_models/ilc_history/iter{k}_Xraw.csv   — state history for dynamic FK training
                                                  columns: phase, twist, height, volume,
                                                           pressure, epi, trans, endo
"""

import os, pickle, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
from pysindy.feature_library import PolynomialLibrary

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

ILC_READY_CSV = None    # None → uses sharedCSVs/ILCReadyData.csv

ILC_ALPHA  = 0.5        # learning gain (0 < α ≤ 1)
Q_CUTOFF   = 0.3        # Q-filter normalised cutoff (fraction of Nyquist)
Q_ORDER    = 3          # Q-filter Butterworth order

ACT_MIN = np.array([200.0, 202.0, 200.0])   # epi, trans, endo lower bounds (mm)
ACT_MAX = np.array([248.0, 248.0, 248.0])   # epi, trans, endo upper bounds (mm)

HEIGHT_OFFSET = 75.0    # mm  — applied to desired height trajectory
VOLUME_OFFSET = 15.0    # mL  — applied to desired volume trajectory

ILC_MODEL = 'SINDy'     # static FK for Jacobian: 'DataDriven', 'PINN', 'SINDy', 'SINDy2'

# Geometry convergence thresholds (physical units).
# When ALL three are satisfied the script prints a STOP signal.
GEOMETRY_RMSE_THRESHOLD = {
    'Twist_deg':  2.0,    # degrees
    'Height_mm':  3.0,    # mm
    'Volume_mL':  5.0,    # mL
}

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE        = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR    = pathlib.Path(BASE) / 'saved_models'
HISTORY_DIR = SAVE_DIR / 'ilc_history'
PYTHONCODES = os.path.join(BASE, '..')
SHARED      = os.path.normpath(os.path.join(BASE, '..', '..', 'sharedCSVs'))
ENG_CSV     = os.path.join(PYTHONCODES, 'engineered_data_withP.csv')

assert SAVE_DIR.exists(), \
    "Run pressure_fk_comparison.py first to generate saved_models/"
HISTORY_DIR.mkdir(exist_ok=True)

# Auto-detect current iteration number from existing history files
_existing = sorted(HISTORY_DIR.glob('iter*_Xraw.csv'))
ITER_NUMBER = len(_existing) + 1
print(f"ILC Motion Correction — Iteration {ITER_NUMBER}")
print(f"  History folder: {HISTORY_DIR}  ({len(_existing)} previous iterations)")

# ══════════════════════════════════════════════════════════════════════════════
# NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# DESIRED TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════
eng        = pd.read_csv(ENG_CSV)
traj_time  = eng['time'].values
traj_p     = eng['pressure'].values
traj_twist = eng['twist'].values
traj_h     = eng['height'].values + HEIGHT_OFFSET
traj_v     = eng['volume'].values + VOLUME_OFFSET
traj_phys  = np.stack([traj_twist, traj_h, traj_v], axis=1)   # (T, 3)
traj_norm  = norm_y(traj_phys)
pressure_n = (traj_p - x_min[3]) / x_den[3]

traj_phase = (traj_time - traj_time[0]) / (traj_time[-1] - traj_time[0])
n_traj     = len(traj_time)

OUTPUT_NAMES = ['Twist_deg', 'Height_mm', 'Volume_mL']
OUTPUT_UNITS = ['deg',       'mm',        'mL']

print(f"\nDesired trajectory: {n_traj} pts  "
      f"height offset={HEIGHT_OFFSET} mm  volume offset={VOLUME_OFFSET} mL")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD STATIC FK MODEL  (for Jacobian direction)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading static FK model: {ILC_MODEL} …")
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
        act_t = torch.tensor(act_n_batch, dtype=torch.float32)
        p_t   = torch.tensor(p_n_batch,   dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            if ILC_MODEL == 'PINN':
                return model(act_t, p_t).numpy()
            else:
                return model(torch.cat([act_t, p_t], dim=1)).numpy()

    def jacobian_at(act_n, p_n):
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
        x_s   = s2_sc_in1.transform(np.asarray(act_phys).reshape(-1, 3))
        theta  = np.asarray(poly_lib1.transform(x_s))
        outs   = []
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
        theta  = np.asarray(poly_lib.transform(x_s))
        outs   = []
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
            a_p = denorm_act(a_n)
            x_p = np.append(a_p, p_phys).reshape(1,-1)
            return norm_y(_pred_fn(x_p)[0])
        f0 = fwd(act_n)
        J  = np.zeros((3, 3))
        for j in range(3):
            ap = act_n.copy(); ap[j] += FD_EPS
            J[:, j] = (fwd(ap) - f0) / FD_EPS
        return J

print("  Static FK model ready.")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ILC-READY DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading ILC-ready data …")

_ilc_path = ILC_READY_CSV or os.path.join(SHARED, 'ILCReadyData.csv')
assert os.path.exists(_ilc_path), \
    f"ILCReadyData.csv not found at:\n  {_ilc_path}"

ilc_df = pd.read_csv(_ilc_path)
ilc_df.columns = ilc_df.columns.str.strip()
print(f"  Loaded {len(ilc_df)} rows — columns: {list(ilc_df.columns)}")

# ── Column auto-detection ─────────────────────────────────────────────────────
def _find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

_COL_PHASE  = _find_col(ilc_df, ['phase', 'cycle_phase', 'time', 'time_s'])
_COL_EPI    = _find_col(ilc_df, ['epi_mm', 'epi', 'Epi_mm', 'Epi', 'u1'])
_COL_TRANS  = _find_col(ilc_df, ['trans_mm', 'trans', 'Trans_mm', 'Trans', 'u2'])
_COL_ENDO   = _find_col(ilc_df, ['endo_mm', 'endo', 'Endo_mm', 'Endo', 'u3'])
_COL_TWIST  = _find_col(ilc_df, ['twist', 'twist_deg', 'Twist'])
_COL_HEIGHT = _find_col(ilc_df, ['height', 'height_mm', 'Height'])
_COL_VOLUME = _find_col(ilc_df, ['volume', 'volume_mL', 'Volume'])
_COL_PRES   = _find_col(ilc_df, ['pressure', 'pressure_mmhg', 'Pressure'])

_missing = [n for n, c in [('epi', _COL_EPI), ('trans', _COL_TRANS),
            ('endo', _COL_ENDO), ('twist', _COL_TWIST),
            ('height', _COL_HEIGHT), ('volume', _COL_VOLUME)] if c is None]
assert not _missing, \
    f"Could not find columns for: {_missing}\nAvailable: {list(ilc_df.columns)}"

print(f"  Columns: epi→'{_COL_EPI}'  trans→'{_COL_TRANS}'  endo→'{_COL_ENDO}'")
print(f"           twist→'{_COL_TWIST}'  height→'{_COL_HEIGHT}'  volume→'{_COL_VOLUME}'")
print(f"           pressure→'{_COL_PRES}'" if _COL_PRES else
      "           pressure→not found (passive recording skipped)")

# ── Phase / resample ──────────────────────────────────────────────────────────
if _COL_PHASE is not None:
    _phase_raw = ilc_df[_COL_PHASE].values
    if _phase_raw.max() > 1.5:
        _phase_raw = (_phase_raw - _phase_raw[0]) / (_phase_raw[-1] - _phase_raw[0])
else:
    _phase_raw = np.linspace(0.0, 1.0, len(ilc_df))

def _interp(src_phase, values, dst_phase):
    return interp1d(src_phase, values,
                    kind='linear', bounds_error=False,
                    fill_value='extrapolate')(dst_phase)

act_phys_avg = np.column_stack([
    _interp(_phase_raw, ilc_df[_COL_EPI].values,   traj_phase),
    _interp(_phase_raw, ilc_df[_COL_TRANS].values, traj_phase),
    _interp(_phase_raw, ilc_df[_COL_ENDO].values,  traj_phase),
])   # (T, 3)

y_phys_avg = np.column_stack([
    _interp(_phase_raw, ilc_df[_COL_TWIST].values,  traj_phase),
    _interp(_phase_raw, ilc_df[_COL_HEIGHT].values, traj_phase),
    _interp(_phase_raw, ilc_df[_COL_VOLUME].values, traj_phase),
])   # (T, 3)

# Pressure — passive recording for history (not used in ILC correction)
if _COL_PRES is not None:
    actual_p   = _interp(_phase_raw, ilc_df[_COL_PRES].values, traj_phase)
    actual_p_n = (actual_p - x_min[3]) / x_den[3]
    print(f"\n  Pressure (passive):  mean={actual_p.mean():.1f} mmHg  "
          f"range=[{actual_p.min():.0f}, {actual_p.max():.0f}] mmHg")
else:
    actual_p   = traj_p.copy()
    actual_p_n = pressure_n.copy()
    print("\n  Pressure: column not found — using desired profile for Jacobian conditioning")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE PASSIVE STATE HISTORY  (training data for dynamic FK)
# ══════════════════════════════════════════════════════════════════════════════
hist_path = HISTORY_DIR / f'iter{ITER_NUMBER:03d}_Xraw.csv'
df_hist = pd.DataFrame({
    'phase':    traj_phase,
    'twist':    y_phys_avg[:, 0],
    'height':   y_phys_avg[:, 1],
    'volume':   y_phys_avg[:, 2],
    'pressure': actual_p,
    'epi':      act_phys_avg[:, 0],
    'trans':    act_phys_avg[:, 1],
    'endo':     act_phys_avg[:, 2],
})
df_hist.to_csv(hist_path, index=False, float_format='%.4f')
print(f"\n  State history saved → {hist_path.name}  "
      f"(total iterations stored: {ITER_NUMBER})")

# ══════════════════════════════════════════════════════════════════════════════
# ILC UPDATE  (geometry only)
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing ILC correction (geometry: twist, height, volume) …")

e_phys = traj_phys - y_phys_avg          # (T, 3) physical error
e_norm = e_phys / y_den                   # (T, 3) normalised error
act_n_avg = norm_act(act_phys_avg)        # (T, 3) normalised actuators

print(f"\n  Tracking error BEFORE correction (iteration {ITER_NUMBER}):")
rmse_before = {}
for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    rmse = np.sqrt(np.mean(e_phys[:, ci]**2))
    rmse_before[col] = rmse
    print(f"    {col:<14}: RMSE = {rmse:.4f} {unit}")

# ── P-type step:  v_k = u_k + α · J⁺ · e_k ──────────────────────────────────
delta_u_n = np.zeros_like(act_n_avg)
for i in range(n_traj):
    J      = jacobian_at(act_n_avg[i], actual_p_n[i])
    J_pinv = np.linalg.pinv(J)
    delta_u_n[i] = ILC_ALPHA * J_pinv @ e_norm[i]

v_k = act_n_avg + delta_u_n

# ── Q-filter:  u_{k+1} = Q[v_k] ─────────────────────────────────────────────
b_q, a_q  = butter(Q_ORDER, Q_CUTOFF, btype='low')
act_n_new = filtfilt(b_q, a_q, v_k, axis=0)

print(f"\n  Q-filter applied  (order={Q_ORDER}, cutoff={Q_CUTOFF})")
print(f"  Max |Δu| before filter : {np.abs(delta_u_n).max():.4f}  (normalised)")
print(f"  Max |Δu| after  filter : {np.abs(act_n_new - act_n_avg).max():.4f}  (normalised)")

# ── Denormalise + clip ────────────────────────────────────────────────────────
act_phys_new = denorm_act(act_n_new)
act_phys_new = np.clip(act_phys_new, ACT_MIN.reshape(1,-1), ACT_MAX.reshape(1,-1))

# ── FK prediction of corrected output ────────────────────────────────────────
y_n_corrected    = fk_predict_norm(norm_act(act_phys_new), pressure_n)
y_phys_corrected = denorm_y(y_n_corrected)
e_corrected      = traj_phys - y_phys_corrected

print(f"\n  Predicted tracking error AFTER correction (FK estimate):")
for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    r_before = rmse_before[col]
    r_after  = np.sqrt(np.mean(e_corrected[:, ci]**2))
    improvement = (r_before - r_after) / r_before * 100
    print(f"    {col:<14}: {r_before:.4f} → {r_after:.4f} {unit}  ({improvement:+.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE CHECK
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  Convergence check (thresholds: "
      f"twist<{GEOMETRY_RMSE_THRESHOLD['Twist_deg']}°  "
      f"height<{GEOMETRY_RMSE_THRESHOLD['Height_mm']}mm  "
      f"volume<{GEOMETRY_RMSE_THRESHOLD['Volume_mL']}mL):")

converged = True
for col, unit in zip(OUTPUT_NAMES, OUTPUT_UNITS):
    rmse   = rmse_before[col]
    thresh = GEOMETRY_RMSE_THRESHOLD[col]
    ok     = rmse < thresh
    converged = converged and ok
    status = 'PASS' if ok else 'FAIL'
    print(f"    [{status}] {col:<14}: {rmse:.3f} {unit}  (threshold {thresh})")

if converged:
    print("\n" + "═"*62)
    print("  GEOMETRY CONVERGED — Phase 1 complete.")
    print("  Next steps:")
    print("    1.  python trainDynamicFK.py")
    print("    2.  python ilcPressureCorrection.py")
    print("═"*62)
else:
    iters_remaining = max(1, sum(
        rmse_before[c] / GEOMETRY_RMSE_THRESHOLD[c] for c in OUTPUT_NAMES) // 3)
    print(f"\n  Not yet converged — run another iteration.")
    print(f"  History files available for dynamic FK training: {ITER_NUMBER}")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE CORRECTED ACTUATORS
# ══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(SHARED, 'ilc_corrected_actuators.csv')
df_out = pd.DataFrame({
    'phase':           traj_phase,
    'time_in_cycle':   traj_time,
    'epi':             act_phys_new[:, 0],
    'trans':           act_phys_new[:, 1],
    'endo':            act_phys_new[:, 2],
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

fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)

for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    # Left — deformation tracking
    ax = axes[ci, 0]
    ax.plot(traj_phase, traj_phys[:, ci],        'k-',  lw=2.5, label='Desired')
    ax.plot(traj_phase, y_phys_avg[:, ci],        'b--', lw=1.8, label='Measured (current)')
    ax.plot(traj_phase, y_phys_corrected[:, ci],  'r-',  lw=1.8, label='Predicted after ILC')
    ax.set_ylabel(f'{col} ({unit})', fontsize=10)
    rmse = rmse_before[col]
    thresh = GEOMETRY_RMSE_THRESHOLD[col]
    ax.set_title(f'RMSE = {rmse:.3f} {unit}  '
                 f'[{"PASS" if rmse < thresh else "FAIL"} < {thresh}]',
                 fontsize=9, color='green' if rmse < thresh else 'red')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right — actuator signal
    ax2 = axes[ci, 1]
    motor = ['Epi', 'Trans', 'Endo'][ci]
    ax2.plot(traj_phase, act_phys_avg[:, ci],  'b--', lw=1.8, label='Current')
    ax2.plot(traj_phase, act_phys_new[:, ci],  'r-',  lw=1.8, label='Corrected')
    ax2.axhline(ACT_MIN[ci], color='grey', lw=1, ls=':', label='Bounds')
    ax2.axhline(ACT_MAX[ci], color='grey', lw=1, ls=':')
    delta = np.abs(act_phys_new[:, ci] - act_phys_avg[:, ci]).max()
    ax2.set_title(f'Max |Δ| = {delta:.2f} mm', fontsize=9, color='grey')
    ax2.set_ylabel(f'{motor} position (mm)', fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

axes[-1, 0].set_xlabel('Cycle phase', fontsize=10)
axes[-1, 1].set_xlabel('Cycle phase', fontsize=10)
axes[0, 0].set_title(axes[0, 0].get_title(), fontsize=9)

plt.suptitle(f'ILC Motion Correction — Iteration {ITER_NUMBER}  '
             f'(α={ILC_ALPHA}, Q_cutoff={Q_CUTOFF}, model={ILC_MODEL})',
             fontsize=12)
plt.tight_layout()

fig_path = os.path.join(BASE, f'ilcMotion_iter{ITER_NUMBER:03d}.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Figure saved → ilcMotion_iter{ITER_NUMBER:03d}.png")
plt.show()
