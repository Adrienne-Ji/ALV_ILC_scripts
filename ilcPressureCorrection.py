"""
ilcPressureCorrection.py
──────────────────────────────────────────────────────────────────────────────
Phase 2 ILC — corrects geometric tracking error AND pressure.

Jacobian structure (4×3, ILC-normalised space):
  Rows 1-3 : ∂[twist, height, volume] / ∂[epi, trans, endo]
             from the same static FK used in Phase 1
  Row 4    : λ · ∂pressure / ∂[epi, trans, endo]
             fitted by direct linear regression on all available ILC history
             data in ILCFiles/Exp_data/itr*/ILCReadyData.csv

ILC update (same as Phase 1 but with 4×3 J):
  v_k       = u_k + α · J_aug⁺ · e_aug
  u_{k+1}   = Q[v_k]            (zero-phase Butterworth)
  u_{k+1}   = clip(u_{k+1}, ACT_MIN, ACT_MAX)

Prerequisites
─────────────
  ilcMotionCorrection.py  run until geometry converges.
  No dynamic model required.

Inputs
──────
  sharedCSVs/ILCReadyData.csv            — averaged stable cycle
  ILCFiles/Exp_data/itr*/ILCReadyData.csv — history for pressure regression
  engineered_data_withP.csv              — desired trajectory
  saved_models/norm_constants.npz        — ILC normalisation constants
  saved_models/sindy_data.pkl  (or dd / pinn variants) — static FK

Outputs
───────
  sharedCSVs/ilc_corrected_actuators.csv
  saved_models/ilc_history/iter{k}_Xraw.csv
"""

import os, pickle, pathlib, re, warnings
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

ILC_READY_CSV = None     # None → uses sharedCSVs/ILCReadyData.csv

ILC_ALPHA  = 0.5         # learning gain
Q_CUTOFF   = 0.3
Q_ORDER    = 3

ACT_MIN = np.array([200.0, 202.0, 200.0])
ACT_MAX = np.array([248.0, 248.0, 248.0])

HEIGHT_OFFSET = 75.0
VOLUME_OFFSET = 15.0

ILC_MODEL = 'SINDy'      # static FK: 'DataDriven' | 'PINN' | 'SINDy' | 'SINDy2'

# Pressure weight: 0 = geometry only, 1 = equal weight to geometry
LAMBDA_P = 0.5

# Reference for pressure error normalisation (expected peak-to-peak mmHg)
P_NORM_REF = 125.0

GEOMETRY_RMSE_THRESHOLD = {
    'Twist_deg':  2.0,
    'Height_mm':  3.0,
    'Volume_mL':  5.0,
}
PRESSURE_RMSE_THRESHOLD = 15.0   # mmHg

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE        = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR    = pathlib.Path(BASE) / 'saved_models'
HISTORY_DIR = SAVE_DIR / 'ilc_history'
PYTHONCODES = os.path.join(BASE, '..')
SHARED      = os.path.normpath(os.path.join(BASE, '..', '..', 'sharedCSVs'))
ENG_CSV     = os.path.join(PYTHONCODES, 'engineered_data_withP.csv')
EXP_DATA    = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Exp_data'

assert SAVE_DIR.exists(), \
    "saved_models/ not found — run pressure_fk_comparison.py first."
HISTORY_DIR.mkdir(exist_ok=True)

_existing   = sorted(HISTORY_DIR.glob('iter*_Xraw.csv'))
ITER_NUMBER = len(_existing) + 1
print(f"ILC Pressure Correction — Iteration {ITER_NUMBER}")
print(f"  (History: {len(_existing)} previous iterations)")

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
# LOAD STATIC FK MODEL  (identical to Phase 1)
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
        for col in ['Twist_deg', 'Height_mm', 'Volume_mL']:
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
        for col in ['Twist_deg', 'Height_mm', 'Volume_mL']:
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
        for i, col in enumerate(['Twist_deg', 'Height_mm', 'Volume_mL']):
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
        p_phys = p_n * x_den[3] + x_min[3]
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
# PRESSURE JACOBIAN — direct regression on ILC history
# ══════════════════════════════════════════════════════════════════════════════
print("\nFitting pressure Jacobian from ILC history data …")

def _itr_num(p):
    m = re.search(r'itr\s*(\d+)', p.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1

def _find_col_df(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

U_reg_list, P_reg_list = [], []

if EXP_DATA.exists():
    hist_csvs = sorted(
        [p / 'ILCReadyData.csv' for p in EXP_DATA.iterdir()
         if p.is_dir() and (p / 'ILCReadyData.csv').exists()],
        key=lambda p: _itr_num(p.parent)
    )
    for csv in hist_csvs:
        df_h = pd.read_csv(csv)
        _ce  = _find_col_df(df_h, ['epi_mm', 'epi'])
        _ct  = _find_col_df(df_h, ['trans_mm', 'trans'])
        _cn  = _find_col_df(df_h, ['endo_mm', 'endo'])
        _cp  = _find_col_df(df_h, ['pressure', 'pressure_mmhg'])
        if None not in (_ce, _ct, _cn, _cp):
            U_reg_list.append(df_h[[_ce, _ct, _cn]].values)
            P_reg_list.append(df_h[_cp].values)
            print(f"  + {csv.parent.name}  ({len(df_h)} pts)")

if not U_reg_list:
    _fallback = os.path.join(SHARED, 'ILCReadyData.csv')
    if os.path.exists(_fallback):
        df_h = pd.read_csv(_fallback)
        _ce  = _find_col_df(df_h, ['epi_mm', 'epi'])
        _ct  = _find_col_df(df_h, ['trans_mm', 'trans'])
        _cn  = _find_col_df(df_h, ['endo_mm', 'endo'])
        _cp  = _find_col_df(df_h, ['pressure', 'pressure_mmhg'])
        if None not in (_ce, _ct, _cn, _cp):
            U_reg_list.append(df_h[[_ce, _ct, _cn]].values)
            P_reg_list.append(df_h[_cp].values)
            print(f"  + sharedCSVs/ILCReadyData.csv  ({len(df_h)} pts) [fallback]")

assert U_reg_list, "No ILC history data found for pressure regression."

U_reg_all = np.concatenate(U_reg_list, axis=0)   # (N_total, 3)  physical mm
P_reg_all = np.concatenate(P_reg_list, axis=0)   # (N_total,)    mmHg

# Normalise to ILC space then fit
U_reg_n   = norm_act(U_reg_all)                   # (N_total, 3)
P_reg_n   = P_reg_all / P_NORM_REF               # (N_total,)

A_reg = np.hstack([U_reg_n, np.ones((len(U_reg_n), 1))])
coeffs, _, _, _ = np.linalg.lstsq(A_reg, P_reg_n, rcond=None)

J_PRES_ROW = coeffs[:3].reshape(1, 3)   # (1, 3)  ∂(P_norm)/∂(u_norm)

n_pts = len(U_reg_all)
P_pred_reg = A_reg @ coeffs
r2 = 1 - np.sum((P_reg_n - P_pred_reg)**2) / np.sum((P_reg_n - P_reg_n.mean())**2)
print(f"\n  Pressure regression  ({n_pts} pts, R²={r2:.3f})")
print(f"    ∂P/∂epi   = {J_PRES_ROW[0,0]:+.4f}  (normalised)")
print(f"    ∂P/∂trans = {J_PRES_ROW[0,1]:+.4f}")
print(f"    ∂P/∂endo  = {J_PRES_ROW[0,2]:+.4f}")
if np.all(J_PRES_ROW > 0):
    print("  WARNING: all pressure Jacobian coefficients are positive — "
          "physically expected to be negative (compression → higher pressure). "
          "Check actuator sign convention.")

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
      f"pressure=[{traj_p.min():.0f}, {traj_p.max():.0f}] mmHg")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ILC-READY DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading ILC-ready data …")

_ilc_path = ILC_READY_CSV or os.path.join(SHARED, 'ILCReadyData.csv')
assert os.path.exists(_ilc_path), f"ILCReadyData.csv not found:\n  {_ilc_path}"

ilc_df = pd.read_csv(_ilc_path)
ilc_df.columns = ilc_df.columns.str.strip()
print(f"  Loaded {len(ilc_df)} rows — columns: {list(ilc_df.columns)}")

def _find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

_COL_PHASE  = _find_col(ilc_df, ['phase', 'cycle_phase', 'time', 'time_s'])
_COL_EPI    = _find_col(ilc_df, ['epi_mm', 'epi', 'Epi_mm', 'Epi'])
_COL_TRANS  = _find_col(ilc_df, ['trans_mm', 'trans', 'Trans_mm', 'Trans'])
_COL_ENDO   = _find_col(ilc_df, ['endo_mm', 'endo', 'Endo_mm', 'Endo'])
_COL_TWIST  = _find_col(ilc_df, ['twist', 'twist_deg', 'Twist'])
_COL_HEIGHT = _find_col(ilc_df, ['height', 'height_mm', 'Height'])
_COL_VOLUME = _find_col(ilc_df, ['volume', 'volume_mL', 'Volume'])
_COL_PRES   = _find_col(ilc_df, ['pressure', 'pressure_mmhg', 'Pressure'])

_missing = [n for n, c in [('epi', _COL_EPI), ('trans', _COL_TRANS),
            ('endo', _COL_ENDO), ('twist', _COL_TWIST), ('height', _COL_HEIGHT),
            ('volume', _COL_VOLUME)] if c is None]
assert not _missing, f"Columns not found: {_missing}\nAvailable: {list(ilc_df.columns)}"

assert _COL_PRES is not None, \
    "Pressure column not found in ILCReadyData.csv — required for Phase 2 ILC.\n" \
    "Ensure markerProcessing.py is writing a pressure column."

print(f"  epi→'{_COL_EPI}'  trans→'{_COL_TRANS}'  endo→'{_COL_ENDO}'")
print(f"  twist→'{_COL_TWIST}'  height→'{_COL_HEIGHT}'  volume→'{_COL_VOLUME}'")
print(f"  pressure→'{_COL_PRES}'  ← active ILC objective")

if _COL_PHASE is not None:
    _phase_raw = ilc_df[_COL_PHASE].values
    if _phase_raw.max() > 1.5:
        _phase_raw = (_phase_raw - _phase_raw[0]) / (_phase_raw[-1] - _phase_raw[0])
else:
    _phase_raw = np.linspace(0.0, 1.0, len(ilc_df))

def _interp(src_phase, values, dst_phase):
    return interp1d(src_phase, values, kind='linear',
                    bounds_error=False, fill_value='extrapolate')(dst_phase)

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

actual_p   = _interp(_phase_raw, ilc_df[_COL_PRES].values, traj_phase)
actual_p_n = (actual_p - x_min[3]) / x_den[3]

print(f"\n  Pressure: mean={actual_p.mean():.1f} mmHg  "
      f"range=[{actual_p.min():.0f}, {actual_p.max():.0f}] mmHg")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE HISTORY
# ══════════════════════════════════════════════════════════════════════════════
hist_path = HISTORY_DIR / f'iter{ITER_NUMBER:03d}_Xraw.csv'
pd.DataFrame({
    'phase':    traj_phase,
    'twist':    y_phys_avg[:, 0],
    'height':   y_phys_avg[:, 1],
    'volume':   y_phys_avg[:, 2],
    'pressure': actual_p,
    'epi':      act_phys_avg[:, 0],
    'trans':    act_phys_avg[:, 1],
    'endo':     act_phys_avg[:, 2],
}).to_csv(hist_path, index=False, float_format='%.4f')
print(f"  History saved → {hist_path.name}")

# ══════════════════════════════════════════════════════════════════════════════
# TRACKING ERROR
# ══════════════════════════════════════════════════════════════════════════════
e_phys   = traj_phys - y_phys_avg     # (T, 3) geometry
e_norm   = e_phys / y_den             # (T, 3) normalised geometry
e_p      = traj_p - actual_p          # (T,)   pressure physical
e_p_norm = e_p / P_NORM_REF           # (T,)   pressure normalised

act_n_avg = norm_act(act_phys_avg)    # (T, 3) normalised actuators

print(f"\n  Tracking error BEFORE correction (iteration {ITER_NUMBER}):")
rmse_before = {}
for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    rmse = np.sqrt(np.mean(e_phys[:, ci]**2))
    rmse_before[col] = rmse
    print(f"    {col:<14}: RMSE = {rmse:.4f} {unit}")
p_rmse_before = np.sqrt(np.mean(e_p**2))
print(f"    {'Pressure':<14}: RMSE = {p_rmse_before:.2f} mmHg  (λ={LAMBDA_P})")

# ══════════════════════════════════════════════════════════════════════════════
# ILC UPDATE  — augmented Jacobian (4×3)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nComputing ILC correction (geometry + pressure, λ={LAMBDA_P}) …")

delta_u_n = np.zeros_like(act_n_avg)

for i in range(n_traj):
    # Geometry Jacobian from static FK (3×3, same as Phase 1)
    J_geom = jacobian_at(act_n_avg[i], actual_p_n[i])

    # Augmented Jacobian (4×3): stack pressure row below geometry rows
    J_aug = np.vstack([J_geom,
                       LAMBDA_P * J_PRES_ROW])        # (4, 3)

    # Augmented error (4,): geometry(3) + pressure(1)
    e_aug = np.append(e_norm[i], e_p_norm[i])         # (4,)

    delta_u_n[i] = ILC_ALPHA * np.linalg.pinv(J_aug) @ e_aug

v_k = act_n_avg + delta_u_n

# ── Q-filter ──────────────────────────────────────────────────────────────────
b_q, a_q  = butter(Q_ORDER, Q_CUTOFF, btype='low')
act_n_new = filtfilt(b_q, a_q, v_k, axis=0)

print(f"  Q-filter applied  (order={Q_ORDER}, cutoff={Q_CUTOFF})")
print(f"  Max |Δu| before filter : {np.abs(delta_u_n).max():.4f}  (normalised)")
print(f"  Max |Δu| after  filter : {np.abs(act_n_new - act_n_avg).max():.4f}  (normalised)")

# ── Denormalise + clip ────────────────────────────────────────────────────────
act_phys_new = denorm_act(act_n_new)
act_phys_new = np.clip(act_phys_new, ACT_MIN.reshape(1,-1), ACT_MAX.reshape(1,-1))

# ── Static FK prediction of corrected geometry ────────────────────────────────
y_n_corrected    = fk_predict_norm(norm_act(act_phys_new), pressure_n)
y_phys_corrected = denorm_y(y_n_corrected)
e_corrected      = traj_phys - y_phys_corrected

print(f"\n  Predicted geometry after correction (FK estimate):")
for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    r_before = rmse_before[col]
    r_after  = np.sqrt(np.mean(e_corrected[:, ci]**2))
    improvement = (r_before - r_after) / r_before * 100 if r_before > 0 else 0.0
    print(f"    {col:<14}: {r_before:.4f} → {r_after:.4f} {unit}  ({improvement:+.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE CHECK
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  Convergence check:")
geo_converged = True
for col, unit in zip(OUTPUT_NAMES, OUTPUT_UNITS):
    rmse   = rmse_before[col]
    thresh = GEOMETRY_RMSE_THRESHOLD[col]
    ok     = rmse < thresh
    geo_converged = geo_converged and ok
    print(f"    [{'PASS' if ok else 'FAIL'}] {col:<14}: "
          f"{rmse:.3f} {unit}  (threshold {thresh})")

p_ok = p_rmse_before < PRESSURE_RMSE_THRESHOLD
print(f"    [{'PASS' if p_ok else 'FAIL'}] {'Pressure':<14}: "
      f"{p_rmse_before:.2f} mmHg  (threshold {PRESSURE_RMSE_THRESHOLD})")

if geo_converged and p_ok:
    print("\n" + "═"*62)
    print("  GEOMETRY + PRESSURE CONVERGED — ILC complete.")
    print("═"*62)
elif geo_converged:
    print(f"\n  Geometry converged — pressure still correcting (RMSE={p_rmse_before:.1f} mmHg).")
    print(f"  Continue running ilcPressureCorrection.py.")
    print(f"  If pressure does not improve, try increasing LAMBDA_P (now {LAMBDA_P}).")
else:
    print(f"\n  Not yet converged — run another iteration.")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE CORRECTED ACTUATORS
# ══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(SHARED, 'ilc_corrected_actuators.csv')
pd.DataFrame({
    'phase':           traj_phase,
    'time_in_cycle':   traj_time,
    'epi':             act_phys_new[:, 0],
    'trans':           act_phys_new[:, 1],
    'endo':            act_phys_new[:, 2],
    'pred_twist_deg':  y_phys_corrected[:, 0],
    'pred_height_mm':  y_phys_corrected[:, 1],
    'pred_volume_mL':  y_phys_corrected[:, 2],
}).to_csv(out_path, index=False)
print(f"\n  Corrected actuators saved → sharedCSVs/ilc_corrected_actuators.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures …")

fig, axes = plt.subplots(4, 2, figsize=(15, 16), sharex=True)

# ── Left column: tracking (desired / actual / FK-predicted) ──────────────────
geo_data = [
    (0, traj_phys[:, 0], y_phys_avg[:, 0], y_phys_corrected[:, 0]),
    (1, traj_phys[:, 1], y_phys_avg[:, 1], y_phys_corrected[:, 1]),
    (2, traj_phys[:, 2], y_phys_avg[:, 2], y_phys_corrected[:, 2]),
]
for ci, (row, desired, actual, predicted) in enumerate(geo_data):
    ax = axes[row, 0]
    ax.plot(traj_phase, desired,   'k-',  lw=2.5, label='Desired')
    ax.plot(traj_phase, actual,    'b--', lw=1.8, label='Measured (current)')
    ax.plot(traj_phase, predicted, 'r-',  lw=1.5, label='FK predicted after ILC', alpha=0.8)
    rmse  = rmse_before[OUTPUT_NAMES[ci]]
    thresh = GEOMETRY_RMSE_THRESHOLD[OUTPUT_NAMES[ci]]
    ax.set_ylabel(f'{OUTPUT_NAMES[ci]} ({OUTPUT_UNITS[ci]})', fontsize=10)
    ax.set_title(f'RMSE = {rmse:.3f} {OUTPUT_UNITS[ci]}  '
                 f'[{"PASS" if rmse < thresh else "FAIL"} < {thresh}]',
                 fontsize=9, color='green' if rmse < thresh else 'red')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Pressure row
ax_p = axes[3, 0]
ax_p.plot(traj_phase, traj_p,  'k-',  lw=2.5, label='Desired')
ax_p.plot(traj_phase, actual_p, 'b--', lw=1.8, label='Measured (current)')
ax_p.set_ylabel('Pressure (mmHg)', fontsize=10)
ax_p.set_xlabel('Cycle phase', fontsize=10)
ax_p.set_title(f'Pressure RMSE = {p_rmse_before:.2f} mmHg  '
               f'[{"PASS" if p_ok else "FAIL"} < {PRESSURE_RMSE_THRESHOLD}]  '
               f'(λ={LAMBDA_P})',
               fontsize=9, color='green' if p_ok else 'red')
ax_p.legend(fontsize=8); ax_p.grid(True, alpha=0.3)

# ── Right column: actuator comparison ────────────────────────────────────────
motor_names = ['Epi', 'Trans', 'Endo']
for ci in range(3):
    ax2 = axes[ci, 1]
    ax2.plot(traj_phase, act_phys_avg[:, ci], 'b--', lw=1.8, label='Current')
    ax2.plot(traj_phase, act_phys_new[:, ci], 'r-',  lw=1.8, label='Corrected')
    ax2.axhline(ACT_MIN[ci], color='grey', lw=1, ls=':')
    ax2.axhline(ACT_MAX[ci], color='grey', lw=1, ls=':')
    delta = np.abs(act_phys_new[:, ci] - act_phys_avg[:, ci]).max()
    ax2.set_title(f'Max |Δ| = {delta:.2f} mm', fontsize=9, color='grey')
    ax2.set_ylabel(f'{motor_names[ci]} position (mm)', fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

# Bottom-right: pressure regression sanity
ax_r = axes[3, 1]
ax_r.scatter(U_reg_n @ J_PRES_ROW[0] * P_NORM_REF,
             P_reg_all, s=10, alpha=0.6, color='steelblue', label='History data')
_mn, _mx = min(P_reg_all.min(), 0), P_reg_all.max() * 1.05
ax_r.plot([_mn, _mx], [_mn, _mx], 'k--', lw=1, label='y=x')
ax_r.set_xlabel('Regression prediction (mmHg)', fontsize=9)
ax_r.set_ylabel('Measured pressure (mmHg)', fontsize=9)
ax_r.set_title(f'Pressure regression  R²={r2:.3f}  ({n_pts} pts)', fontsize=9)
ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.3)
ax_r.set_xlabel('Cycle phase', fontsize=10)

axes[0, 0].set_title('Deformation tracking', fontsize=10)
axes[0, 1].set_title('Actuator signals', fontsize=10)

plt.suptitle(f'ILC Pressure Correction — Iteration {ITER_NUMBER}  '
             f'(α={ILC_ALPHA}, λ={LAMBDA_P}, Q_cutoff={Q_CUTOFF}, '
             f'R²_pres={r2:.3f})',
             fontsize=12)
plt.tight_layout()

fig_path = os.path.join(BASE, f'ilcPressure_iter{ITER_NUMBER:03d}.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Figure saved → ilcPressure_iter{ITER_NUMBER:03d}.png")
plt.show()
