"""
ilcCorrection_clean.py  —  CLEANED COPY OF ilcCorrection.py (original kept unchanged)
──────────────────────────────────────────────────────────────────────────────
Unified ILC — corrects geometry AND pressure in a single pass.

Architecture
────────────
Geometry Jacobian  : SINDy per-point, Gaussian-smoothed (σ=SINDY_SMOOTH_SIGMA).
                     Phase-varying and nonlinear; no data-driven blending.
Pressure Jacobian  : Fourier regression on pooled session history.
                     Active only at IVC/IVR phases (isovolumic — no geometry
                     conflict). Permanently enabled once geometry passes the
                     RMSE gate (one-way latch stored as a flag file).

ILC update
──────────
  v_k     = u_k + α · pinv(W·J_aug)·(W·e_aug)
  u_{k+1} = Q-filter(v_k)
  u_{k+1} = clip(u_{k+1}, ACT_MIN, ACT_MAX)

  W = per-phase adaptive geometry weights (GEOM_WEIGHTS × local sensitivity);
      pressure row weight = λ_eff (only at IVC/IVR, else 0).

Inputs
──────
  sharedCSVs/ILCReadyData.csv
  ILCFiles/Exp_data/<date>/**/ILCReadyData.csv   — session history
  ILCFiles/Engineered_trajs/engineered_data_<case>.csv
  saved_models/norm_constants.npz
  saved_models/sindy_data.pkl

Outputs
───────
  sharedCSVs/ilc_corrected_actuators.csv
  saved_models/ilc_history/iter{k}_Xraw.csv
"""

import os, sys, pickle, pathlib, re, warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

SESSION_DIR = os.environ.get('ILC_SESSION_DIR', '6_28')
SIM_CASE    = os.environ.get('ILC_CASE', 'healthy')

ILC_READY_CSV = None     # None → sharedCSVs/ILCReadyData.csv

ILC_ALPHA          = 0.55   # learning rate during geometry-only phase
ILC_ALPHA_PRESSURE = 0.20   # lower alpha once geometry gate fires — prevents p_itr6-style overshoot
                             # (6_18 data: RMSE 9.7→39.2 at alpha=0.55; step too large near convergence)
Q_CUTOFF   = 0.45   # raised from 0.35 — timing correction is phase-localised (IVC=30% of cycle);
                    # 0.35 cutoff was smearing it across the full cycle, fighting the timing fix
Q_ORDER    = 3

GEOM_WEIGHTS          = np.array([10.0, 1.0, 1.0])  # [twist, height, volume] — geometry-only phase
GEOM_WEIGHTS_PRESSURE = np.array([ 3.0, 1.0, 0.1])  # used once pressure correction activates
                                                      # twist: 10→3 (high conflict with pressure)
                                                      # volume: 0.1 experimental — reduce volume pull during pressure correction

USE_ADAPTIVE_WEIGHTS = True   # scale per-phase weights by local J row norms
SENSITIVITY_FLOOR    = 0.25   # minimum weight fraction at near-zero sensitivity

SINDY_SMOOTH_SIGMA = 4   # Gaussian sigma in phase samples for SINDy Jacobian

# rcond for pinv — zeros singular values below s_max / SINDY_COND_MAX
SINDY_COND_MAX = 300.0

# Delta cap — limits per-iteration correction per phase point to prevent actuator saturation.
# Without cap, ILC drove trans to ACT_MIN at peak systole → physical bounce → spurious 2nd pressure peak.
# Residual above cap is redistributed to other actuators (via REDIST_ACT_WEIGHTS).
MAX_DELTA_U_MM = 2.0   # mm per phase point per iteration

# Residual redistribution preference: higher = preferred receiver
REDIST_ACT_WEIGHTS = np.array([1.0, 0.0, 3.0])  # [epi, trans, endo] — trans=0: never receives redistribution

USE_DELTA_CAP_REDISTRIBUTION  = True    # cap excess → redistribute to other actuators
USE_PHYS_LIMIT_REDISTRIBUTION = True    # physical limit redistribution stays on

# Per-actuator learning-rate scale — trans dampened to limit iterative drift
ACT_ALPHA_SCALE = np.array([1.0, 0.75, 1.0])  # [epi, trans, endo]

ACT_MIN = np.array([200.0, 202.0, 200.0])
ACT_MAX = np.array([248.0, 248.0, 248.0])

from rig_config import HEIGHT_OFFSET, VOLUME_OFFSET

# Pressure weights
LAMBDA_P_EARLY = 0.05   # always-on from itr0 (pre-geometry-gate): prevents geometry ILC from
                         # dragging pressure 17+ mmHg off target before pressure correction activates
LAMBDA_P       = 0.50   # full pressure correction after geometry gate passes
                         # dP/dt matching tried in 7_16 but failed — noise amplification + shape mismatch
                         # P-matching has 97% geometry alignment in IVC → higher lambda safe vs dP/dt

# Pressure correction gated to isovolumic phases only (no geometry conflict)
IVC_PHASE           = (0.00, 0.30)
IVR_PHASE           = (0.30, 0.60)
PRESSURE_FADE_SIGMA = 12.0   # Gaussian fade width (phase samples)

P_NORM_REF          = 125.0   # mmHg, reference for pressure normalisation (kept for regression fitting)
DPDT_NORM_REF       = 1000.0  # mmHg/s, normalisation reference for dP/dt error signal
DPDT_RMSE_THRESHOLD = 200.0   # mmHg/s convergence criterion (~20% of desired peak dP/dt)

N_FOURIER = 1   # empirically validated on 7_16 data against Linear/N=2/N=3:
                #   OOS R²=0.785  LOO R²=0.809  sign_acc=79%  conflict=25% >90deg  ILC_score=0.59
                # N=3 failed: sign_acc=54% (worse than random in IVC), conflict=28% — too many params for 4 train itrs
                # Linear failed: sign_acc=83% but conflict=45% (corrections fight geometry half the time)
                # N=1 captures IVC/IVR phase asymmetry with 1 harmonic; cannot overfit to per-iter patterns

EXCLUDE_ENDO_FROM_JPRES = True   # endo-pressure Jacobian sign is unstable across rig sessions
                                  # (sign flipped between 6_18–7_3 and 7_13–7_16, cause: rig repositioning)
                                  # epi and trans are 100% sign-consistent across all 6 sessions

PRESSURE_MODEL_R2_FLOOR = 0.30   # warn if R² below this — sign accuracy unreliable
                                  # HARD BLOCK if R² < 0 (model is actively worse than mean)

# Unified RMSE pass standard — 10% of desired trajectory span
GEOMETRY_RMSE_THRESHOLD = {
    'Twist_deg': 2.0,   # deg
    'Height_mm': 1.8,   # mm   (10% × 18.0 mm span)
    'Volume_mL': 6.0,   # mL   (10% × 60 mL span)
}
GEOMETRY_R2_THRESHOLD = {
    'Twist_deg': 0.90,
    'Height_mm': 0.95,
    'Volume_mL': 0.95,
}

PRESSURE_GATE_RMSE      = GEOMETRY_RMSE_THRESHOLD   # same standard, no separate gate
PRESSURE_GATE_R2        = {'Twist_deg': 0.80, 'Height_mm': 0.80, 'Volume_mL': 0.85}
PRESSURE_RMSE_THRESHOLD = 5.0   # mmHg

SHAPE_WEIGHT = 0.0   # slope-mismatch penalty; 0 = disabled

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE        = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR    = pathlib.Path(BASE) / 'saved_models'
HISTORY_DIR = SAVE_DIR / 'ilc_history'
SHARED      = os.path.normpath(os.path.join(BASE, '..', '..', 'sharedCSVs'))
ENGINEERED_TRAJS = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Engineered_trajs'

_SIM_CASE_FILES = {
    'healthy':   ENGINEERED_TRAJS / 'engineered_data_healthy.csv',
    'diastolic': ENGINEERED_TRAJS / 'engineered_data_diastolic_dysfunction.csv',
    'systolic':  ENGINEERED_TRAJS / 'engineered_data_systolic_dysfunction.csv',
}
assert SIM_CASE in _SIM_CASE_FILES, \
    f"SIM_CASE must be one of {list(_SIM_CASE_FILES)}, got '{SIM_CASE}'"
ENG_CSV  = _SIM_CASE_FILES[SIM_CASE]
EXP_DATA = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Exp_data' / SESSION_DIR

print(f"Simulation case: {SIM_CASE}  →  {ENG_CSV}")
print(f"Session history scope: {EXP_DATA}")

assert SAVE_DIR.exists(), "saved_models/ not found — run pressure_fk_comparison.py first."
HISTORY_DIR.mkdir(exist_ok=True)

_tag_match = re.search(r'(\d+)\s*$', os.environ.get('ILC_OUTPUT_TAG', ''))
if _tag_match:
    ITER_NUMBER = int(_tag_match.group(1))
else:
    ITER_NUMBER = len(sorted(HISTORY_DIR.glob('iter*_Xraw.csv'))) + 1
print(f"ILC Correction — Iteration {ITER_NUMBER}")

# ══════════════════════════════════════════════════════════════════════════════
# NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
norm  = np.load(SAVE_DIR / 'norm_constants.npz')
x_min = norm['x_min']; x_den = norm['x_den']
y_min = norm['y_min']; y_den = norm['y_den']

def norm_act(act_phys):  return (np.asarray(act_phys) - x_min[:3]) / x_den[:3]
def denorm_act(act_n):   return np.asarray(act_n) * x_den[:3] + x_min[:3]
def norm_y(y_phys):      return (np.asarray(y_phys) - y_min) / y_den
def denorm_y(y_n):       return np.asarray(y_n) * y_den + y_min

# ══════════════════════════════════════════════════════════════════════════════
# STATIC FK MODEL (SINDy)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading static FK model: SINDy …")
POLY_DEGREE = 3

with open(SAVE_DIR / 'sindy_data.pkl', 'rb') as f:
    sindy_data = pickle.load(f)
sindy_results = sindy_data['results']
sindy_sc      = sindy_data['sc']
poly_lib      = sindy_data['poly_lib']

_OUTPUT_NAMES = ['Twist_deg', 'Height_mm', 'Volume_mL']

def _sindy_predict_phys(x_physical):
    x_s   = sindy_sc.transform(np.asarray(x_physical).reshape(-1, 4))
    theta = np.asarray(poly_lib.transform(x_s))
    outs  = []
    for col in _OUTPUT_NAMES:
        r = sindy_results[col]
        outs.append(r['sy'].inverse_transform((theta @ r['coef']).reshape(-1,1)).ravel())
    return np.stack(outs, axis=1)

FD_EPS = 1e-4

def jacobian_at(act_n, p_n):
    p_phys = p_n * x_den[3] + x_min[3]
    def fwd(a_n):
        a_p = denorm_act(a_n)
        return norm_y(_sindy_predict_phys(np.append(a_p, p_phys).reshape(1,-1))[0])
    f0 = fwd(act_n)
    J  = np.zeros((3, 3))
    for j in range(3):
        ap = act_n.copy(); ap[j] += FD_EPS
        J[:, j] = (fwd(ap) - f0) / FD_EPS
    return J

print("  Static FK model ready.")

# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE JACOBIAN — Fourier regression on session history
# ══════════════════════════════════════════════════════════════════════════════
def _find_col_df(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    return None

def _itr_sort_key(csv_path):
    itr_folder = csv_path.parent
    _dm = re.match(r'(\d+)_(\d+)\s*$', itr_folder.parent.name)
    date_key = (int(_dm.group(1)), int(_dm.group(2))) if _dm else (0, 0)
    m = re.search(r'(\d+)\s*$', itr_folder.name)
    num = int(m.group(1)) if m else -1
    is_p = bool(re.match(r'p_itr', itr_folder.name, re.IGNORECASE))
    return (date_key, 1 if is_p else 0, num)

def _fourier_design(phi, u_norm, n_harm):
    cols = []
    for j in range(3):
        cols.append(u_norm[:, j])
        for k in range(1, n_harm + 1):
            cols.append(u_norm[:, j] * np.sin(2 * np.pi * k * phi))
            cols.append(u_norm[:, j] * np.cos(2 * np.pi * k * phi))
    cols.append(np.ones(len(phi)))
    return np.column_stack(cols)

def J_pres_at_phase(phi_scalar, f_coeffs, n_harm):
    J = np.zeros(3)
    idx = 0
    for j in range(3):
        J[j] = f_coeffs[idx]; idx += 1
        for k in range(1, n_harm + 1):
            J[j] += f_coeffs[idx] * np.sin(2 * np.pi * k * phi_scalar); idx += 1
            J[j] += f_coeffs[idx] * np.cos(2 * np.pi * k * phi_scalar); idx += 1
    if EXCLUDE_ENDO_FROM_JPRES:
        J[2] = 0.0   # endo sign unstable across sessions — exclude from pressure correction
    return J.reshape(1, 3)

def J_dpdt_at_phase(phi_scalar, f_coeffs, n_harm, T_cyc):
    """∂(dP/dt)/∂u — time-derivative of J_pres.
    From Fourier model: d/dφ [b_js*sin + b_jc*cos] = b_js*2πk*cos - b_jc*2πk*sin.
    Divide by T_cyc to convert per-phase → per-second."""
    J = np.zeros(3)
    idx = 0
    for j in range(3):
        idx += 1   # b_j0 constant: d/dφ = 0, skip
        for k in range(1, n_harm + 1):
            J[j] += f_coeffs[idx] *  (2*np.pi*k) * np.cos(2*np.pi*k*phi_scalar); idx += 1
            J[j] += f_coeffs[idx] * -(2*np.pi*k) * np.sin(2*np.pi*k*phi_scalar); idx += 1
    return J.reshape(1, 3) / T_cyc

print("\nFitting phase-varying pressure Jacobian from ILC history …")

U_reg_list, P_reg_list, PHI_reg_list = [], [], []

def _load_history_csv(df_h, label):
    _ce  = _find_col_df(df_h, ['epi_mm',  'epi'])
    _ct  = _find_col_df(df_h, ['trans_mm','trans'])
    _cn  = _find_col_df(df_h, ['endo_mm', 'endo'])
    _cp  = _find_col_df(df_h, ['pressure','pressure_mmhg'])
    _cph = _find_col_df(df_h, ['phase','cycle_phase','time','time_s'])
    if None in (_ce, _ct, _cn, _cp): return
    phi = df_h[_cph].values if _cph else np.linspace(0.0, 1.0, len(df_h))
    if phi.max() > 1.5:
        phi = (phi - phi[0]) / (phi[-1] - phi[0])
    U_reg_list.append(df_h[[_ce, _ct, _cn]].values)
    P_reg_list.append(df_h[_cp].values)
    PHI_reg_list.append(phi)
    print(f"  + {label}  ({len(df_h)} pts)")

if EXP_DATA.exists():
    _reg_csvs = (c for c in EXP_DATA.rglob('ILCReadyData.csv')
                 if '_backup' not in c.parts)
    for csv in sorted(_reg_csvs, key=_itr_sort_key):
        _load_history_csv(pd.read_csv(csv), f"{csv.parent.parent.name}/{csv.parent.name}")

if not U_reg_list:
    _fb = os.path.join(SHARED, 'ILCReadyData.csv')
    if os.path.exists(_fb):
        _load_history_csv(pd.read_csv(_fb), 'sharedCSVs/ILCReadyData.csv [fallback]')

REGRESSION_FLOOR = 1

if len(U_reg_list) >= REGRESSION_FLOOR:
    U_reg_all   = np.concatenate(U_reg_list,   axis=0)
    P_reg_all   = np.concatenate(P_reg_list,   axis=0)
    PHI_reg_all = np.concatenate(PHI_reg_list, axis=0)
    U_reg_n     = norm_act(U_reg_all)
    P_reg_n     = P_reg_all / P_NORM_REF

    A_reg          = _fourier_design(PHI_reg_all, U_reg_n, N_FOURIER)
    fourier_coeffs, _, _, _ = np.linalg.lstsq(A_reg, P_reg_n, rcond=None)
    P_pred_reg     = A_reg @ fourier_coeffs
    r2             = 1 - np.sum((P_reg_n - P_pred_reg)**2) / \
                         np.sum((P_reg_n - P_reg_n.mean())**2)
    n_reg_pts      = len(U_reg_all)

    LAMBDA_P_EFF = LAMBDA_P   # fixed — latch handles activation timing

    print(f"  Fourier regression: {len(U_reg_list)} iter files, {n_reg_pts} pts, "
          f"{A_reg.shape[1]} features  R²={r2:.3f}  N_FOURIER={N_FOURIER}  λ={LAMBDA_P_EFF:.3f}")
    for _phi, _lbl in [(0.0,'start'), (0.25,'systole'), (0.5,'mid'), (0.75,'diastole')]:
        _j = J_pres_at_phase(_phi, fourier_coeffs, N_FOURIER)[0]
        _endo_note = ' [endo EXCLUDED]' if EXCLUDE_ENDO_FROM_JPRES else ''
        print(f"    phi={_phi:.2f} ({_lbl}):  "
              f"dP/d_epi={_j[0]:+.4f}  dP/d_trans={_j[1]:+.4f}  dP/d_endo={_j[2]:+.4f}{_endo_note}")

    # ── Pressure model quality gate ───────────────────────────────────────────
    if r2 < 0.0:
        print(f"\n  !! PRESSURE MODEL CRITICAL: R²={r2:.3f}")
        print(f"     Model predicts WORSE than the mean — Jacobian signs are likely wrong.")
        print(f"     This happens when training data is from a different rig configuration.")
        print(f"     Pressure correction BLOCKED this iteration. Run more geometry iterations")
        print(f"     within this session to build enough within-session regression data.")
        LAMBDA_P_EFF = 0.0
    elif r2 < PRESSURE_MODEL_R2_FLOOR:
        print(f"\n  ** PRESSURE MODEL WARNING: R²={r2:.3f} < {PRESSURE_MODEL_R2_FLOOR:.2f}")
        print(f"     Low fit quality — sign accuracy may be unreliable.")
        print(f"     Pressure will run at reduced lambda: "
              f"{LAMBDA_P:.3f} -> {LAMBDA_P * r2 / PRESSURE_MODEL_R2_FLOOR:.3f}")
        LAMBDA_P_EFF = LAMBDA_P * (r2 / PRESSURE_MODEL_R2_FLOOR)
    else:
        print(f"  Pressure model quality OK: R²={r2:.3f} >= {PRESSURE_MODEL_R2_FLOOR:.2f}")
else:
    fourier_coeffs = None
    LAMBDA_P_EFF   = 0.0
    r2, n_reg_pts  = 0.0, 0
    U_reg_all = P_reg_all = PHI_reg_all = U_reg_n = P_pred_reg = np.array([])
    print(f"  Only {len(U_reg_list)} file(s) — need ≥{REGRESSION_FLOOR} — pressure disabled.")

# ══════════════════════════════════════════════════════════════════════════════
# DESIRED TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════
eng        = pd.read_csv(ENG_CSV)
traj_time  = eng['time'].values
traj_p     = eng['pressure'].values
traj_twist = eng['twist'].values
traj_h     = eng['height'].values + HEIGHT_OFFSET
traj_v     = eng['volume'].values + VOLUME_OFFSET
traj_phys  = np.stack([traj_twist, traj_h, traj_v], axis=1)
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
        if c.lower() in lower: return lower[c.lower()]
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
            ('endo', _COL_ENDO), ('twist', _COL_TWIST),
            ('height', _COL_HEIGHT), ('volume', _COL_VOLUME)] if c is None]
assert not _missing, f"Columns not found: {_missing}\nAvailable: {list(ilc_df.columns)}"

print(f"  epi→'{_COL_EPI}'  trans→'{_COL_TRANS}'  endo→'{_COL_ENDO}'")
print(f"  twist→'{_COL_TWIST}'  height→'{_COL_HEIGHT}'  volume→'{_COL_VOLUME}'")
if _COL_PRES:
    print(f"  pressure→'{_COL_PRES}'  "
          f"({'active, λ='+str(LAMBDA_P_EFF) if LAMBDA_P_EFF > 0 else 'passive only'})")
else:
    print("  pressure→not found")

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
])

y_phys_avg = np.column_stack([
    _interp(_phase_raw, ilc_df[_COL_TWIST].values,  traj_phase),
    _interp(_phase_raw, ilc_df[_COL_HEIGHT].values, traj_phase),
    _interp(_phase_raw, ilc_df[_COL_VOLUME].values, traj_phase),
])

if _COL_PRES is not None:
    actual_p   = _interp(_phase_raw, ilc_df[_COL_PRES].values, traj_phase)
    actual_p_n = (actual_p - x_min[3]) / x_den[3]
    print(f"\n  Pressure: mean={actual_p.mean():.1f} mmHg  "
          f"range=[{actual_p.min():.0f}, {actual_p.max():.0f}] mmHg")
else:
    actual_p   = traj_p.copy()
    actual_p_n = pressure_n.copy()
    print("\n  Pressure column not found — using desired profile for FK conditioning")

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
e_phys   = traj_phys - y_phys_avg
e_norm   = e_phys / y_den
e_p      = traj_p - actual_p
e_p_norm = e_p / P_NORM_REF

# dP/dt error — used for pressure correction (sharper than tracking P directly)
T_CYCLE      = float(traj_time[-1] - traj_time[0])   # seconds per cycle
dpdt_des     = np.gradient(traj_p,   traj_phase * T_CYCLE)   # mmHg/s
dpdt_meas    = np.gradient(actual_p, traj_phase * T_CYCLE)   # mmHg/s
e_dpdt       = dpdt_des - dpdt_meas                           # mmHg/s
e_dpdt_norm  = e_dpdt / DPDT_NORM_REF

act_n_avg = norm_act(act_phys_avg)

print(f"\n  Tracking error BEFORE correction (iteration {ITER_NUMBER}):")
rmse_before = {}
r2_before   = {}
for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    des  = traj_phys[:, ci]
    meas = y_phys_avg[:, ci]
    rmse = np.sqrt(np.mean((des - meas)**2))
    r    = float(np.corrcoef(des, meas)[0, 1])
    rmse_before[col] = rmse
    r2_before[col]   = r**2
    print(f"    {col:<14}: RMSE = {rmse:.4f} {unit}   r² = {r2_before[col]:.4f}")
p_rmse_before    = np.sqrt(np.mean(e_p**2))
dpdt_rmse_before = np.sqrt(np.mean(e_dpdt**2))
print(f"    {'Pressure':<14}: RMSE = {p_rmse_before:.2f} mmHg   |   dP/dt RMSE = {dpdt_rmse_before:.1f} mmHg/s")

# ── One-way pressure latch ────────────────────────────────────────────────────
# Stored as a flag file so it persists across invocations within the session.
# Re-evaluating every iteration causes geometry↔pressure oscillation.
_PRESSURE_LATCH = SAVE_DIR / f"pressure_unlocked_{SESSION_DIR}_{SIM_CASE}.flag"

_geom_passes_now = all(rmse_before[col] < PRESSURE_GATE_RMSE[col] for col in OUTPUT_NAMES)
if _geom_passes_now and not _PRESSURE_LATCH.exists():
    _PRESSURE_LATCH.touch()
    print(f"\n  Geometry passed gate — pressure latch SET for {SESSION_DIR}/{SIM_CASE}")

_pressure_unlocked = _PRESSURE_LATCH.exists()

for col in OUTPUT_NAMES:
    rmse_ok = rmse_before[col] < PRESSURE_GATE_RMSE[col]
    print(f"    [{'PASS' if rmse_ok else 'FAIL'}] {col}: "
          f"RMSE={rmse_before[col]:.3f}  (gate {PRESSURE_GATE_RMSE[col]})"
          f"  r²={r2_before[col]:.3f}")

if not _pressure_unlocked:
    if fourier_coeffs is None:
        print(f"\n  Pressure EARLY mode waiting — no regression data yet (need >={REGRESSION_FLOOR} itr files)")
        LAMBDA_P_EFF = 0.0
    elif LAMBDA_P_EFF > 1e-6:
        LAMBDA_P_EFF = LAMBDA_P_EARLY   # cap to early-mode lambda; R² gate did not block
        print(f"\n  Pressure EARLY mode (geometry gate not yet passed):  lambda={LAMBDA_P_EFF:.3f}")
        print(f"     Prevents geometry ILC from dragging pressure off target before full correction activates.")
        print(f"     Full pressure correction (lambda={LAMBDA_P:.3f}, alpha={ILC_ALPHA_PRESSURE}) fires after geometry gate.")
    else:
        print(f"\n  Pressure EARLY mode BLOCKED by R² quality gate this iteration")
else:
    if LAMBDA_P_EFF > 1e-6:
        print(f"\n  Pressure ACTIVE (latch held):  lambda={LAMBDA_P_EFF:.3f}  alpha={ILC_ALPHA_PRESSURE}"
              f"  (IVC {IVC_PHASE} + IVR {IVR_PHASE})")
    else:
        print(f"\n  Pressure BLOCKED by R² quality gate this iteration (lambda=0)")

# ══════════════════════════════════════════════════════════════════════════════
# ILC UPDATE
# ══════════════════════════════════════════════════════════════════════════════
_has_pressure   = LAMBDA_P_EFF > 1e-6 and _COL_PRES is not None
pressure_active = _pressure_unlocked and _has_pressure    # full mode: latch fired, reduced geom weights, low alpha
pressure_early  = (not _pressure_unlocked) and _has_pressure  # early mode: pre-gate, full geom weights, normal alpha

if pressure_active:
    mode_str = (f"geometry + pressure (FULL)  lambda={LAMBDA_P_EFF:.3f}  "
                f"alpha={ILC_ALPHA_PRESSURE}  geom_w={list(GEOM_WEIGHTS_PRESSURE)}")
elif pressure_early:
    mode_str = (f"geometry + pressure (EARLY)  lambda={LAMBDA_P_EFF:.3f}  "
                f"alpha={ILC_ALPHA}  geom_w={list(GEOM_WEIGHTS)}")
elif _pressure_unlocked:
    mode_str = "geometry only — pressure blocked by R² quality gate"
else:
    mode_str = "geometry only — no regression data yet"
print(f"\nComputing ILC correction  [{mode_str}] …")

delta_u_n = np.zeros_like(act_n_avg)

# SINDy Jacobian — phase-varying, Gaussian-smoothed to suppress FD noise
_J_static_raw = np.zeros((n_traj, 3, 3))
for _i in range(n_traj):
    _J_static_raw[_i] = jacobian_at(act_n_avg[_i], actual_p_n[_i])

if SINDY_SMOOTH_SIGMA > 0:
    _J_static_all = np.zeros_like(_J_static_raw)
    for _oi in range(3):
        for _ci in range(3):
            _J_static_all[:, _oi, _ci] = gaussian_filter1d(
                _J_static_raw[:, _oi, _ci], sigma=SINDY_SMOOTH_SIGMA, mode='wrap')
else:
    _J_static_all = _J_static_raw

print(f"\n  Geometry Jacobian: SINDy per-point, smoothed σ={SINDY_SMOOTH_SIGMA}"
      + ("  (adaptive weights ON)" if USE_ADAPTIVE_WEIGHTS else ""))

if USE_ADAPTIVE_WEIGHTS:
    _row_norms  = np.linalg.norm(_J_static_all, axis=2)          # (n_traj, 3)
    _row_max    = _row_norms.max(axis=0, keepdims=True) + 1e-8
    _adaptive_w = (_row_norms / _row_max) * (1 - SENSITIVITY_FLOOR) + SENSITIVITY_FLOOR
    _adaptive_w = _adaptive_w * GEOM_WEIGHTS[None, :]

# Hard IVC/IVR envelope — no Gaussian fade (fade was spreading correction across full cycle)
_lambda_hard = np.array([
    1.0 if (IVC_PHASE[0] <= p <= IVC_PHASE[1] or IVR_PHASE[0] <= p <= IVR_PHASE[1])
    else 0.0
    for p in traj_phase
])
_lambda_envelope = _lambda_hard.copy()

peak_lambda = _lambda_envelope.max()
n_nonzero   = int(np.sum(_lambda_envelope > 0.01))
print(f"  Pressure envelope: peak={peak_lambda:.3f}  "
      f"non-trivial at {n_nonzero}/{n_traj} pts  "
      f"(IVC {IVC_PHASE} + IVR {IVR_PHASE}, σ={PRESSURE_FADE_SIGMA})")

if SHAPE_WEIGHT > 0:
    _y_norm_meas  = norm_y(y_phys_avg)
    _e_slope_norm = np.zeros_like(e_norm)
    for _i in range(n_traj):
        _ip = (_i - 1) % n_traj
        _e_slope_norm[_i] = (traj_norm[_i] - traj_norm[_ip]) - \
                            (_y_norm_meas[_i] - _y_norm_meas[_ip])
    e_norm = e_norm + SHAPE_WEIGHT * _e_slope_norm

# Alpha and geometry weights depend on pressure state:
#   EARLY mode  (pre-gate, lambda=0.05): full geom weights + normal alpha — geometry still converging
#   ACTIVE mode (post-gate, lambda=0.50): reduced geom weights + lower alpha — timing correction, prevent overshoot
_alpha  = ILC_ALPHA_PRESSURE if pressure_active else ILC_ALPHA
_base_w_loop = GEOM_WEIGHTS_PRESSURE if pressure_active else GEOM_WEIGHTS

print(f"  ILC alpha={_alpha}  base_geom_w={list(_base_w_loop)}  "
      f"({'ACTIVE' if pressure_active else 'EARLY' if pressure_early else 'geom-only'})")

for i in range(n_traj):
    geom_w_i = (_adaptive_w[i] / GEOM_WEIGHTS * _base_w_loop) if USE_ADAPTIVE_WEIGHTS else _base_w_loop
    # Apply pressure correction in both early and active modes (lambda differs)
    lambda_i = LAMBDA_P_EFF * _lambda_envelope[i] if _has_pressure else 0.0

    if lambda_i > 1e-6:
        J_pres = J_pres_at_phase(traj_phase[i], fourier_coeffs, N_FOURIER)
        J_aug  = np.vstack([_J_static_all[i], lambda_i * J_pres])  # (4, 3)
        e_aug  = np.append(e_norm[i], e_p_norm[i])
        w_full = np.append(geom_w_i, 1.0)
    else:
        J_aug  = _J_static_all[i]
        e_aug  = e_norm[i]
        w_full = geom_w_i

    J_w = J_aug * w_full[:, None]
    e_w = e_aug * w_full
    delta_u_n[i] = _alpha * np.linalg.pinv(J_w, rcond=1.0/SINDY_COND_MAX) @ e_w

delta_u_n *= ACT_ALPHA_SCALE[None, :]

# ── Hard delta cap + redistribution ──────────────────────────────────────────
delta_u_max_n = MAX_DELTA_U_MM / x_den[:3]
delta_u_n_raw = delta_u_n.copy()
delta_u_n     = np.clip(delta_u_n_raw, -delta_u_max_n, delta_u_max_n)

n_clamped = int(np.sum(np.abs(delta_u_n_raw) > delta_u_max_n))
n_redistributed = 0
if USE_DELTA_CAP_REDISTRIBUTION:
    for i in range(n_traj):
        clipped = np.abs(delta_u_n_raw[i]) > delta_u_max_n
        if not clipped.any(): continue
        free = ~clipped
        if not free.any(): continue
        J_i        = _J_static_all[i]
        e_residual = J_i @ delta_u_n_raw[i] - J_i @ delta_u_n[i]
        w_free     = REDIST_ACT_WEIGHTS[free]
        J_free_w   = J_i[:, free] * w_free[None, :]
        extra_w, _, _, _ = np.linalg.lstsq(J_free_w, e_residual, rcond=None)
        extra = extra_w * w_free
        delta_u_n[i, free] = np.clip(delta_u_n[i, free] + extra,
                                      -delta_u_max_n[free], delta_u_max_n[free])
        n_redistributed += 1

if n_clamped > 0:
    print(f"  Delta cap: {n_clamped} entries clamped  "
          f"({n_redistributed} pts redistributed{'' if USE_DELTA_CAP_REDISTRIBUTION else ', redistribution disabled'})")

# ── Physical limit redistribution ────────────────────────────────────────────
_act_phys_proposed = denorm_act(act_n_avg + delta_u_n)
_act_phys_limited  = np.clip(_act_phys_proposed, ACT_MIN.reshape(1,-1), ACT_MAX.reshape(1,-1))
_phys_viol         = np.abs(_act_phys_proposed - _act_phys_limited) > 0.01
_delta_n_at_limit  = norm_act(_act_phys_limited) - act_n_avg

n_phys_clamped = int(_phys_viol.sum())
n_phys_redist  = 0
if n_phys_clamped > 0:
    if USE_PHYS_LIMIT_REDISTRIBUTION:
        for i in range(n_traj):
            viol = _phys_viol[i]
            if not viol.any(): continue
            free = ~viol
            if not free.any(): continue
            J_i        = _J_static_all[i]
            e_residual = J_i @ (delta_u_n[i] - _delta_n_at_limit[i])
            w_free     = REDIST_ACT_WEIGHTS[free]
            J_free_w   = J_i[:, free] * w_free[None, :]
            extra_w, _, _, _ = np.linalg.lstsq(J_free_w, e_residual, rcond=None)
            delta_u_n[i, free] += extra_w * w_free
            delta_u_n[i, viol]  = _delta_n_at_limit[i, viol]
            n_phys_redist += 1
    else:
        delta_u_n = norm_act(_act_phys_limited) - act_n_avg
    print(f"  Physical limit: {n_phys_clamped} entries hit ACT_MIN/ACT_MAX  "
          f"({n_phys_redist} pts redistributed{'' if USE_PHYS_LIMIT_REDISTRIBUTION else ', redistribution disabled'})")

# ── Smooth delta_u to remove redistribution spikes before Q-filter ───────────
# Point-by-point redistribution can create sharp per-timestep jumps in delta_u.
# Gaussian pre-smoothing removes these without affecting the low-freq correction.
DELTA_SMOOTH_SIGMA = 3   # phase samples (~3% of cycle at 100 pts)
delta_u_n = gaussian_filter1d(delta_u_n, sigma=DELTA_SMOOTH_SIGMA, axis=0, mode='wrap')

v_k = act_n_avg + delta_u_n

# ── Q-filter ─────────────────────────────────────────────────────────────────
b_q, a_q  = butter(Q_ORDER, Q_CUTOFF, btype='low')
act_n_new = filtfilt(b_q, a_q, v_k, axis=0)

print(f"  Delta pre-smooth: Gaussian σ={DELTA_SMOOTH_SIGMA} samples (wrap)")
print(f"  Q-filter applied  (order={Q_ORDER}, cutoff={Q_CUTOFF})")
print(f"  Max |Δu| before filter : {np.abs(delta_u_n).max():.4f}  (normalised)")
print(f"  Max |Δu| after  filter : {np.abs(act_n_new - act_n_avg).max():.4f}  (normalised)")

act_phys_new = np.clip(denorm_act(act_n_new), ACT_MIN.reshape(1,-1), ACT_MAX.reshape(1,-1))

# ── Geometry prediction ───────────────────────────────────────────────────────
delta_u_norm_pred = act_n_new - act_n_avg
y_norm_corrected  = norm_y(y_phys_avg) + np.einsum('noc,nc->no', _J_static_all, delta_u_norm_pred)
y_phys_corrected  = denorm_y(y_norm_corrected)
e_corrected       = traj_phys - y_phys_corrected

print(f"\n  Predicted geometry after correction [SINDy per-point smoothed σ={SINDY_SMOOTH_SIGMA}]:")
for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    r_before = rmse_before[col]
    r_after  = np.sqrt(np.mean(e_corrected[:, ci]**2))
    pct = (r_before - r_after) / r_before * 100 if r_before > 0 else 0.0
    print(f"    {col:<14}: {r_before:.4f} → {r_after:.4f} {unit}  ({pct:+.1f}%)")

if pressure_active:
    # Predict P after correction: p_pred = actual_p + J_pres(φ)·Δu * P_NORM_REF
    _delta_u_n_pred = act_n_new - act_n_avg
    _J_pres_all = np.array([J_pres_at_phase(traj_phase[i], fourier_coeffs, N_FOURIER)[0]
                             for i in range(n_traj)])    # (n_traj, 3)
    _delta_p    = (_J_pres_all * _delta_u_n_pred).sum(axis=1) * P_NORM_REF  # (n_traj,) mmHg
    p_pred_new  = actual_p + _delta_p
    p_pred_rmse = np.sqrt(np.mean((traj_p - p_pred_new)**2))
    dpdt_pred     = np.gradient(p_pred_new, traj_phase * T_CYCLE)
    dpdt_pred_rmse = np.sqrt(np.mean((dpdt_des - dpdt_pred)**2))
    print(f"    {'Pressure':<14}: {p_rmse_before:.2f} → {p_pred_rmse:.2f} mmHg  "
          f"(predicted: actual_p + J_pres·Δu)")
else:
    p_pred_new = None
    dpdt_pred  = None

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

p_ok     = p_rmse_before    < PRESSURE_RMSE_THRESHOLD
dpdt_ok  = dpdt_rmse_before < DPDT_RMSE_THRESHOLD
pres_str = f"{p_rmse_before:.2f} mmHg  (threshold {PRESSURE_RMSE_THRESHOLD})"
dpdt_str = f"{dpdt_rmse_before:.1f} mmHg/s  (threshold {DPDT_RMSE_THRESHOLD})"
if pressure_active:
    print(f"    [{'PASS' if p_ok else 'FAIL'}] {'Pressure':<14}: {pres_str}")
    print(f"    [{'PASS' if dpdt_ok else 'FAIL'}] {'dP/dt':<14}: {dpdt_str}")
else:
    print(f"    [----] {'Pressure':<14}: {pres_str}  (passive — λ=0)")
    print(f"    [----] {'dP/dt':<14}: {dpdt_str}  (passive — λ=0)")

if geo_converged and (p_ok or not pressure_active):
    print("\n" + "═"*62)
    print("  CONVERGED" + (" — geometry + pressure." if pressure_active
                           else " — geometry only."))
    if not pressure_active:
        print("  Pressure correction will activate next iteration.")
    print("═"*62)
else:
    print(f"\n  Not yet converged — run another iteration.")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE CORRECTED ACTUATORS
# ══════════════════════════════════════════════════════════════════════════════
out_path = os.path.join(SHARED, 'ilc_corrected_actuators.csv')
pd.DataFrame({
    'phase':          traj_phase,
    'time_in_cycle':  traj_time,
    'epi':            act_phys_new[:, 0],
    'trans':          act_phys_new[:, 1],
    'endo':           act_phys_new[:, 2],
    'pred_twist_deg': y_phys_corrected[:, 0],
    'pred_height_mm': y_phys_corrected[:, 1],
    'pred_volume_mL': y_phys_corrected[:, 2],
}).to_csv(out_path, index=False)
print(f"\n  Corrected actuators → sharedCSVs/ilc_corrected_actuators.csv")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures …")

n_rows = 4 if _COL_PRES else 3
fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4 * n_rows))

for row in range(1, n_rows):
    axes[row, 0].sharex(axes[0, 0])
for row in range(1, min(3, n_rows)):
    axes[row, 1].sharex(axes[0, 1])

for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    ax = axes[ci, 0]
    ax.plot(traj_phase, traj_phys[:, ci],        'k-',  lw=2.5, label='Desired')
    ax.plot(traj_phase, y_phys_avg[:, ci],        'b--', lw=1.8, label='Measured')
    ax.plot(traj_phase, y_phys_corrected[:, ci],  'r-',  lw=1.5, label='FK predicted after ILC', alpha=0.8)
    rmse   = rmse_before[col]
    thresh = GEOMETRY_RMSE_THRESHOLD[col]
    ok     = rmse < thresh
    ax.set_ylabel(f'{col} ({unit})', fontsize=10)
    ax.text(0.01, 0.02,
            f'RMSE = {rmse:.3f} {unit}  [{"PASS" if ok else "FAIL"} < {thresh}]   r² = {r2_before[col]:.3f}',
            transform=ax.transAxes, fontsize=9, va='bottom',
            color='green' if ok else 'red',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

if _COL_PRES:
    ax_p = axes[3, 0]
    ax_p.plot(traj_phase, traj_p,   'k-',  lw=2.5, label='Desired P')
    ax_p.plot(traj_phase, actual_p, 'b--', lw=1.8, label='Measured P')
    if p_pred_new is not None:
        ax_p.plot(traj_phase, p_pred_new, 'r-', lw=1.5,
                  label=f'Predicted P (actual_p + J_pres·du, RMSE->{p_pred_rmse:.1f} mmHg)',
                  alpha=0.85)
    ax_p.set_ylabel('Pressure (mmHg)', fontsize=10)
    pres_label = (f'P RMSE = {p_rmse_before:.2f} mmHg  [{"PASS" if p_ok else "FAIL"} < {PRESSURE_RMSE_THRESHOLD}]  '
                  f'dP/dt RMSE = {dpdt_rmse_before:.1f} mmHg/s  [{"PASS" if dpdt_ok else "FAIL"} < {DPDT_RMSE_THRESHOLD}]  '
                  f'(λ={LAMBDA_P_EFF})')
    ax_p.set_title(pres_label, fontsize=9, color='green' if (p_ok and dpdt_ok) else 'red')
    ax_p.legend(fontsize=8); ax_p.grid(True, alpha=0.3)

    # dP/dt subplot
    ax_dp = axes[3, 1] if axes.ndim > 1 and axes.shape[1] > 1 else None
    if ax_dp is None:
        # single-column layout — reuse pressure axis twin
        ax_dp = ax_p.twinx()
        ax_dp.plot(traj_phase, dpdt_des,  'k:',  lw=1.5, alpha=0.6, label='dP/dt desired')
        ax_dp.plot(traj_phase, dpdt_meas, 'b:',  lw=1.5, alpha=0.6, label='dP/dt measured')
        if dpdt_pred is not None:
            ax_dp.plot(traj_phase, dpdt_pred, 'r:', lw=1.5, alpha=0.6,
                       label=f'dP/dt predicted  (RMSE→{dpdt_pred_rmse:.1f} mmHg/s)')
        ax_dp.set_ylabel('dP/dt (mmHg/s)', fontsize=9)
        ax_dp.legend(fontsize=7, loc='lower right')
    ax_p.set_xlabel('Cycle phase', fontsize=10)

motor_names = ['Epi', 'Trans', 'Endo']

_raw_corr_mm   = delta_u_n_raw * x_den[:3]
_final_corr_mm = delta_u_n     * x_den[:3]
_was_delta_capped = np.abs(_raw_corr_mm) > MAX_DELTA_U_MM
_clamp_contrib    = np.clip(_raw_corr_mm, -MAX_DELTA_U_MM, MAX_DELTA_U_MM)
_redist_mm        = _final_corr_mm - _clamp_contrib
# only flag actual redistribution — with both flags off, final≠clamp_contrib only at
# physical limit points (hard clip), which is not redistribution
_got_redist = (np.abs(_redist_mm) > 0.05) & \
              (USE_DELTA_CAP_REDISTRIBUTION | USE_PHYS_LIMIT_REDISTRIBUTION)

for ci in range(3):
    ax2 = axes[ci, 1]
    _act_raw = act_phys_avg[:, ci] + _raw_corr_mm[:, ci]
    ax2.plot(traj_phase, _act_raw, color='purple', lw=1.0, ls=':', alpha=0.55,
             label='Raw (pre-cap/limit)')
    ax2.plot(traj_phase, act_phys_avg[:, ci], 'b--', lw=1.8, label='Current')
    ax2.plot(traj_phase, act_phys_new[:, ci], 'r-',  lw=1.8, label='Corrected')

    if _was_delta_capped[:, ci].any():
        ax2.fill_between(traj_phase, ACT_MIN[ci] - 2, ACT_MAX[ci] + 2,
                         where=_was_delta_capped[:, ci],
                         alpha=0.18, color='red', label='Delta cap hit (donor)')
    if _phys_viol[:, ci].any():
        ax2.fill_between(traj_phase, ACT_MIN[ci] - 2, ACT_MAX[ci] + 2,
                         where=_phys_viol[:, ci],
                         alpha=0.25, color='orange', label='Physical limit hit (donor)')
    if _got_redist[:, ci].any():
        ax2.fill_between(traj_phase,
                         act_phys_avg[:, ci] + _clamp_contrib[:, ci],
                         act_phys_new[:, ci],
                         where=_got_redist[:, ci],
                         alpha=0.30, color='limegreen', label='Received redistribution')

    ax2.axhline(ACT_MIN[ci], color='grey', lw=1, ls=':')
    ax2.axhline(ACT_MAX[ci], color='grey', lw=1, ls=':')
    delta  = np.abs(act_phys_new[:, ci] - act_phys_avg[:, ci]).max()
    n_dc   = int(_was_delta_capped[:, ci].sum())
    n_pl   = int(_phys_viol[:, ci].sum())
    n_rd   = int(_got_redist[:, ci].sum())
    title_str = f'Max |Δ| = {delta:.2f} mm'
    if n_dc: title_str += f'  🔴 Δcap@{n_dc}pts'
    if n_pl: title_str += f'  🟠 lim@{n_pl}pts'
    if n_rd: title_str += f'  🟢 +redist@{n_rd}pts'
    ax2.set_title(title_str, fontsize=8, color='grey')
    ax2.set_ylabel(f'{motor_names[ci]} (mm)', fontsize=10)
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

if _COL_PRES and n_rows == 4:
    ax_r = axes[3, 1]
    if pressure_active:
        _phi_plot = np.linspace(0, 1, 200)
        _J_plot   = np.array([J_pres_at_phase(p, fourier_coeffs, N_FOURIER)[0]
                              for p in _phi_plot])
        ax_r.plot(_phi_plot, _J_plot[:, 0], color='tab:blue',   lw=2, label='∂P/∂epi')
        ax_r.plot(_phi_plot, _J_plot[:, 1], color='tab:orange', lw=2, label='∂P/∂trans')
        ax_r.plot(_phi_plot, _J_plot[:, 2], color='tab:green',  lw=2, label='∂P/∂endo')
        ax_r.axhline(0, color='k', lw=0.8, ls='--', alpha=0.4)
        ax_r.set_xlabel('Cycle phase', fontsize=9)
        ax_r.set_ylabel('∂P/∂u  (normalised)', fontsize=9)
        ax_r.set_title(f'Phase-varying pressure Jacobian  R²={r2:.3f}  '
                       f'({n_reg_pts} pts,  N={N_FOURIER})', fontsize=9)
        ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.3)
    else:
        _reason = 'no regression data yet' if fourier_coeffs is None else 'λ=0'
        ax_r.text(0.5, 0.5, f'Pressure correction inactive\n({_reason})',
                  ha='center', va='center', transform=ax_r.transAxes, fontsize=11)
        ax_r.set_axis_off()

axes[n_rows-1, 0].set_xlabel('Cycle phase', fontsize=10)
axes[min(2, n_rows-1), 1].set_xlabel('Cycle phase', fontsize=10)
axes[0, 0].set_title('Deformation tracking', fontsize=10)
axes[0, 1].set_title('Actuator signals', fontsize=10)

_mode_tag = ("FULL" if pressure_active else "EARLY" if pressure_early else "geom-only")
plt.suptitle(
    f'ILC Correction — Iteration {ITER_NUMBER}  '
    f'(alpha={_alpha},  lambda={LAMBDA_P_EFF:.3f},  Q={Q_CUTOFF},  '
    f'P={_mode_tag}{"  R²="+f"{r2:.2f}" if _has_pressure else ""})',
    fontsize=12)
plt.tight_layout()

fig_path = os.path.join(BASE, f'ilcCorrection_iter{ITER_NUMBER:03d}.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Figure → ilcCorrection_iter{ITER_NUMBER:03d}.png")
plt.show()
