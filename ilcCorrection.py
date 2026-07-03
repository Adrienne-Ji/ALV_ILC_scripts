"""
ilcCorrection.py
──────────────────────────────────────────────────────────────────────────────
Unified ILC — corrects geometry AND pressure in a single pass, entirely
data-driven from iteration history (no dynamic model required).

Jacobian (4×3, ILC-normalised space):
  Rows 1-3 : ∂[twist, height, volume] / ∂[epi, trans, endo]
             a CONFIDENCE-BLENDED combination of the static FK Jacobian and an
             empirically-fitted absolute regression Y = J·U + b:

               J_geom = w · J_empirical + (1−w) · J_static

             A hard cutover (100% static FK below a threshold, then suddenly
             100% regression) leaves the regression with no stable floor of
             direction exactly when it's most likely to be wrong — thin or
             collinear session data can give a fit with fine R² but unstable
             individual epi/trans/endo coefficients, producing a confidently
             -wrong Jacobian that makes ILC diverge instead of converge.
             Blending keeps a sensible baseline direction from the static FK
             while gradually trusting the data-driven estimate as it earns it.

             w = w_n · w_cond:
               w_n    ramps 0→1 linearly from MIN_HISTORY_ITERS to
                      FULL_TRUST_ITERS history iterations this session
               w_cond ramps 1→0 (log scale) as the design matrix
                      [epi,trans,endo,1]'s condition number goes from
                      GOOD_COND_NUMBER to MAX_COND_NUMBER (catches actuator
                      collinearity directly, since iteration count alone
                      doesn't guarantee independent actuator combinations)

             J_empirical is fitted (and its conditioning checked) as soon as
             REGRESSION_FLOOR (2) iteration files exist, pooling EVERY file
             found under SESSION_DIR — including other trajectory cases run
             the same day, since actuator→deformation sensitivity is a
             property of the physical device that session, not of which
             target trajectory was being chased. J_static is evaluated ONCE
             at the trajectory's mean operating point and held constant over
             the whole cycle — NOT per phase point — since pointwise
             finite-difference evaluation on the unreliable static model lets
             noise vary erratically along the cycle, producing an oscillatory
             correction instead of a smooth one.
  Row 4    : λ_eff · ∂pressure / ∂[epi, trans, endo]
             phase-varying Fourier regression on the same pooled session
             history. λ_eff = LAMBDA_P · w_n (same iteration-count ramp as
             geometry's w_n, no static-pressure-model equivalent to blend
             against) — smooth, not a hard on/off cutover.

ILC update:
  v_k     = u_k + α · pinv(W·J_aug)·(W·e_aug)   (W = GEOM_WEIGHTS, [+1.0] pressure)
  u_{k+1} = Q-filter(v_k)
  u_{k+1} = clip(u_{k+1}, ACT_MIN, ACT_MAX)

Geometry prediction after correction uses the same blended Jacobian:
  y_pred = y_current + J_geom · Δu   (linearised, valid at any blend weight)

The ILC tracking error itself stays case-specific — it's always computed
against ENG_CSV (the desired trajectory for the case actually being run this
iteration). Only the Jacobian fit pools across cases.

Supersedes ilcMotionCorrection.py, ilcPressureCorrection.py and the SINDy-C
dynamic model in trainDynamicFK.py (abandoned — insufficient trajectory
diversity across ILC iterations for SINDy to identify reliably).

Inputs
──────
  sharedCSVs/ILCReadyData.csv                        — current iteration data
  ILCFiles/Exp_data/<date>/<itr|p_itr>*/ILCReadyData.csv  — history (geometry + pressure)
  ILCFiles/Engineered_trajs/engineered_data_<case>.csv    — desired trajectory (see SIM_CASE)
  saved_models/norm_constants.npz                    — normalisation constants
  saved_models/sindy_data.pkl                        — static FK model (cold-start fallback only)

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
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# Session date folder under ILCFiles/Exp_data — set this EVERY session (new
# date). History is scoped to Exp_data/<SESSION_DIR>/ ONLY — never read across
# dates, because the soft robotic device's behaviour drifts day to day.
# If multiple cases are run the same day (see SIM_CASE below), each case writes
# into its own subfolder (Exp_data/<SESSION_DIR>/<case>/itrN/) but the empirical
# Jacobian fit still pools across ALL case subfolders within this SESSION_DIR —
# actuator→deformation sensitivity is a device property, not case-specific.
# Overridable via env var ILC_SESSION_DIR (set automatically by run_ilc_pipeline.py).
SESSION_DIR = os.environ.get('ILC_SESSION_DIR', '6_28')

# Which trajectory case to simulate — this sets the DESIRED trajectory the ILC
# tracking error is computed against (case-specific, unlike the Jacobian fit
# above). Pick one:
#   'healthy'    → ILCFiles/Engineered_trajs/engineered_data_healthy.csv
#   'diastolic'  → ILCFiles/Engineered_trajs/engineered_data_diastolic_dysfunction.csv
#   'systolic'   → ILCFiles/Engineered_trajs/engineered_data_systolic_dysfunction.csv
# Overridable via env var ILC_CASE (set automatically by run_ilc_pipeline.py).
SIM_CASE = os.environ.get('ILC_CASE', 'healthy')

ILC_READY_CSV = None     # None → sharedCSVs/ILCReadyData.csv

ILC_ALPHA  = 0.55        # learning gain  (0 < α ≤ 1)
Q_CUTOFF   = 0.35        # Q-filter normalised cutoff
Q_ORDER    = 3

GEOM_WEIGHTS = np.array([1.0, 0.5, 1.0])  # [twist, height, volume] — base weights (further modulated per phase)

# When sensitivity to an output collapses at a phase point (e.g. ∂Volume/∂trans
# → 0 at compression), adaptively reduce that output's correction weight at
# that phase so effort is redirected to other actuators that DO have meaningful
# sensitivity there, rather than pushing a near-zero channel.
USE_ADAPTIVE_WEIGHTS = True   # scale per-phase GEOM_WEIGHTS by local J row norms
SENSITIVITY_FLOOR    = 0.25   # minimum weight fraction even when sensitivity is near zero

# Static FK Jacobian evaluation strategy:
# The SINDy model is nonlinear — its gradient varies significantly across the
# cardiac cycle (visualised in plotSINDyLocalGradient.py).  Assuming a single
# constant value (evaluated at the mean operating point) loses this structure
# and specifically misdirects Volume correction, where sensitivity collapses
# to near-zero at peak compression but is large during filling.
# Per-point evaluation restores local accuracy; Gaussian smoothing across
# phase suppresses finite-difference noise that caused oscillation in the old
# architecture without it.
SINDY_SMOOTH_SIGMA = 4   # Gaussian sigma in phase samples (0 = no smoothing, raw per-point)

# ── Confidence-blended geometry Jacobian ────────────────────────────────────
# J_geom = w · J_empirical + (1−w) · J_static  — a SMOOTH blend, not a hard
# cutover. A cliff-edge switch (100% static FK below a threshold, then
# suddenly 100% regression) leaves the regression with no stable floor of
# direction when it's thin/noisy — exactly when it's most likely to be wrong.
# Blending keeps a sensible baseline direction from the static FK while
# gradually trusting the data-driven estimate more as it earns it.
#
# w = w_n · w_cond, where:
#   w_n    ramps 0→1 linearly as session history goes from MIN_HISTORY_ITERS
#          (still mostly FK) to FULL_TRUST_ITERS (mostly/fully empirical)
#   w_cond ramps 1→0 (log scale) as the design matrix conditioning goes from
#          GOOD_COND_NUMBER (trustworthy) to MAX_COND_NUMBER (too collinear
#          to trust at all, regardless of how much data exists)
MIN_HISTORY_ITERS  = 1      # ramp starts here — w_n > 0 as soon as iteration 2 exists
                            # (n_sources=1 itself still gives w_n=0 exactly; REGRESSION_FLOOR
                            # below controls when the fit is even attempted at all)
FULL_TRUST_ITERS   = 10     # at/above this, w_n = 1 (pure empirical, if well-conditioned)
GOOD_COND_NUMBER   = 1e2    # below this, w_cond = 1 (no penalty)
MAX_COND_NUMBER    = 1e4    # at/above this, w_cond = 0 (reject regression entirely)

# Soft safety cap on the per-iteration actuator correction — only triggers for
# numerically extreme corrections (ill-conditioned Jacobian on a degenerate
# session start).  Raised from 8mm: adaptive weights now handle the near-zero-
# sensitivity case that was causing the original overcorrection, so the 8mm
# cap was blocking legitimate large corrections at low-sensitivity phases.
# Physical actuator limits (ACT_MIN/ACT_MAX) remain the true hard backstop.
MAX_DELTA_U_MM = 20.0

ACT_MIN = np.array([200.0, 202.0, 200.0])
ACT_MAX = np.array([248.0, 248.0, 248.0])

# Calibration offsets — imported from rig_config.py (the single source of truth).
# Change them there, not here, so plotILCConvergence.py and any other script
# that compares measured vs desired automatically picks up the update.
from rig_config import HEIGHT_OFFSET, VOLUME_OFFSET

# Pressure weight — scales pressure row relative to geometry rows.
# 0 = geometry only,  0.5 = moderate,  1.0 = equal weight.
# Only active when history data is available; otherwise forced to 0.
# Dropped back from 0.5 — early-session R²_P=1.00 is a sign of overfitting
# (too few diverse iterations for the Fourier regression), so this row's
# correction direction isn't trustworthy yet. Raise again once R²_P looks
# more realistic (well under 1.0) with more session data.
LAMBDA_P = 0.3

# Pressure correction is ONLY active during IVC and IVR — the two isovolumic
# phases where volume is constant but pressure changes sharply.  During
# ejection (0.04–0.42) and filling (0.44–1.0), both volume AND pressure change
# simultaneously: layering pressure correction on top of geometry correction
# creates direct conflicts (the two rows of the augmented Jacobian can push
# actuators in opposite directions for volume vs pressure).  Gating to
# isovolumic phases removes this conflict entirely.
IVC_PHASE          = (0.00, 0.04)   # Isovolumic Contraction: pressure RISES here.
                                    # Conflict ratio 0.656 (above baseline 0.499) — some
                                    # geometry disturbance accepted as the clinical trade-off
                                    # for correct pressure RISE TIMING.  The geometry
                                    # convergence gate ensures geometry is already PASS
                                    # before this ever fires, so the absolute disturbance
                                    # during 4% of the cycle is small and acceptable.
IVR_PHASE          = (0.42, 0.44)   # Isovolumic Relaxation: pressure DROPS here.
                                    # Conflict ratio 0.312 (global minimum) — lowest
                                    # geometry disturbance per unit pressure correction.
PRESSURE_FADE_SIGMA = 1.5           # Gaussian sigma (phase samples) for the smooth
                                    # lambda fade at window boundaries — avoids hard
                                    # steps in delta_u that would kink the trajectory

# Reference for pressure error normalisation (peak-to-peak expected mmHg)
P_NORM_REF = 125.0

# Fourier harmonics for phase-varying pressure Jacobian.
# J_reg(φ)[j] = a_j + Σ_k  b_jk·sin(2πkφ) + c_jk·cos(2πkφ)
# 1 = smooth sinusoid,  2 = captures asymmetric systolic peak (recommended once
# there's enough session data — dropped to 1 for now to halve the parameter
# count (16→10) and reduce overfitting risk while history is still thin)
N_FOURIER = 1

GEOMETRY_RMSE_THRESHOLD = {
    'Twist_deg': 1.0,    # deg
    'Height_mm': 2.0,    # mm
    'Volume_mL': 5.0,    # mL
}
PRESSURE_RMSE_THRESHOLD = 5.0   # mmHg

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE        = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR    = pathlib.Path(BASE) / 'saved_models'
HISTORY_DIR = SAVE_DIR / 'ilc_history'
PYTHONCODES = os.path.join(BASE, '..')
SHARED      = os.path.normpath(os.path.join(BASE, '..', '..', 'sharedCSVs'))
ENGINEERED_TRAJS = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Engineered_trajs'

_SIM_CASE_FILES = {
    'healthy':   ENGINEERED_TRAJS / 'engineered_data_healthy.csv',
    'diastolic': ENGINEERED_TRAJS / 'engineered_data_diastolic_dysfunction.csv',
    'systolic':  ENGINEERED_TRAJS / 'engineered_data_systolic_dysfunction.csv',
}
assert SIM_CASE in _SIM_CASE_FILES, f"SIM_CASE must be one of {list(_SIM_CASE_FILES)}, got '{SIM_CASE}'"
ENG_CSV = _SIM_CASE_FILES[SIM_CASE]
print(f"Simulation case: {SIM_CASE}  →  {ENG_CSV}")

EXP_DATA    = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Exp_data' / SESSION_DIR
print(f"Session history scope: {EXP_DATA}  (history outside this folder is ignored)")

assert SAVE_DIR.exists(), \
    "saved_models/ not found — run pressure_fk_comparison.py first."
HISTORY_DIR.mkdir(exist_ok=True)

# Iteration number for display/filenames: parsed from OUTPUT_TAG when run via
# run_ilc_pipeline.py (e.g. 'P_iter3' → 3) so it matches what's on the figure
# filename and the Zaber file you actually loaded — NOT a global count of every
# file ever written to ilc_history/ across all sessions (that grows forever
# and stops meaning anything session-relative).
_tag_match  = re.search(r'(\d+)\s*$', os.environ.get('ILC_OUTPUT_TAG', ''))
if _tag_match:
    ITER_NUMBER = int(_tag_match.group(1))
else:
    _existing   = sorted(HISTORY_DIR.glob('iter*_Xraw.csv'))
    ITER_NUMBER = len(_existing) + 1
print(f"ILC Correction — Iteration {ITER_NUMBER}")

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
# LOAD STATIC FK MODEL — SINDy, cold-start fallback only (first iteration of a
# session, before USE_EMPIRICAL_JACOBIAN has any history to fit from)
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
        x_p = np.append(a_p, p_phys).reshape(1,-1)
        return norm_y(_sindy_predict_phys(x_p)[0])
    f0 = fwd(act_n)
    J  = np.zeros((3, 3))
    for j in range(3):
        ap = act_n.copy(); ap[j] += FD_EPS
        J[:, j] = (fwd(ap) - f0) / FD_EPS
    return J

print("  Static FK model ready.")

# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE JACOBIAN — Fourier-augmented phase-varying regression on ILC history
# ══════════════════════════════════════════════════════════════════════════════
def _find_col_df(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def _itr_num(p):
    m = re.search(r'itr\s*(\d+)', p.name, re.IGNORECASE)
    return int(m.group(1)) if m else -1

def _itr_sort_key(csv_path):
    """Sort key for an ILCReadyData.csv path: by date folder (if nested as e.g.
    '6_18'), then geometry iters (itr X) before pressure iters (p_itr X), by number."""
    itr_folder = csv_path.parent
    _dm = re.match(r'(\d+)_(\d+)\s*$', itr_folder.parent.name)
    date_key = (int(_dm.group(1)), int(_dm.group(2))) if _dm else (0, 0)
    m = re.search(r'(\d+)\s*$', itr_folder.name)
    num = int(m.group(1)) if m else -1
    is_p = bool(re.match(r'p_itr', itr_folder.name, re.IGNORECASE))
    return (date_key, 1 if is_p else 0, num)

def _fourier_design(phi, u_norm, n_harm):
    """
    Design matrix for Fourier-augmented regression.
    Each actuator j gets: u_j, u_j*sin(2πkφ), u_j*cos(2πkφ) for k=1..n_harm
    Plus scalar bias. Shape: (N, 3*(1+2*n_harm)+1)
    """
    cols = []
    for j in range(3):
        cols.append(u_norm[:, j])
        for k in range(1, n_harm + 1):
            cols.append(u_norm[:, j] * np.sin(2 * np.pi * k * phi))
            cols.append(u_norm[:, j] * np.cos(2 * np.pi * k * phi))
    cols.append(np.ones(len(phi)))
    return np.column_stack(cols)

def J_pres_at_phase(phi_scalar, f_coeffs, n_harm):
    """Phase-varying pressure Jacobian row (1×3) at a single phase value."""
    J = np.zeros(3)
    idx = 0
    for j in range(3):
        J[j] = f_coeffs[idx]
        idx += 1
        for k in range(1, n_harm + 1):
            J[j] += f_coeffs[idx] * np.sin(2 * np.pi * k * phi_scalar); idx += 1
            J[j] += f_coeffs[idx] * np.cos(2 * np.pi * k * phi_scalar); idx += 1
    return J.reshape(1, 3)

def _iter_weight(n_sources):
    """Confidence ramp 0→1 from MIN_HISTORY_ITERS to FULL_TRUST_ITERS."""
    if FULL_TRUST_ITERS <= MIN_HISTORY_ITERS:
        return 1.0 if n_sources >= MIN_HISTORY_ITERS else 0.0
    return float(np.clip((n_sources - MIN_HISTORY_ITERS) /
                          (FULL_TRUST_ITERS - MIN_HISTORY_ITERS), 0.0, 1.0))

def _cond_weight(cond):
    """Confidence ramp 1→0 (log scale) from GOOD_COND_NUMBER to MAX_COND_NUMBER."""
    if cond <= GOOD_COND_NUMBER:
        return 1.0
    if cond >= MAX_COND_NUMBER:
        return 0.0
    log_c, log_good, log_max = np.log10(cond), np.log10(GOOD_COND_NUMBER), np.log10(MAX_COND_NUMBER)
    return float(np.clip(1.0 - (log_c - log_good) / (log_max - log_good), 0.0, 1.0))

print("\nFitting phase-varying pressure Jacobian from ILC history …")

U_reg_list, P_reg_list, PHI_reg_list = [], [], []

def _load_history_csv(df_h, label):
    _ce  = _find_col_df(df_h, ['epi_mm',  'epi'])
    _ct  = _find_col_df(df_h, ['trans_mm','trans'])
    _cn  = _find_col_df(df_h, ['endo_mm', 'endo'])
    _cp  = _find_col_df(df_h, ['pressure','pressure_mmhg'])
    _cph = _find_col_df(df_h, ['phase','cycle_phase','time','time_s'])
    if None in (_ce, _ct, _cn, _cp):
        return
    phi = df_h[_cph].values if _cph else np.linspace(0.0, 1.0, len(df_h))
    if phi.max() > 1.5:
        phi = (phi - phi[0]) / (phi[-1] - phi[0])
    U_reg_list.append(df_h[[_ce, _ct, _cn]].values)
    P_reg_list.append(df_h[_cp].values)
    PHI_reg_list.append(phi)
    print(f"  + {label}  ({len(df_h)} pts)")

if EXP_DATA.exists():
    for csv in sorted(EXP_DATA.rglob('ILCReadyData.csv'), key=_itr_sort_key):
        _load_history_csv(pd.read_csv(csv), f"{csv.parent.parent.name}/{csv.parent.name}")

if not U_reg_list:
    _fb = os.path.join(SHARED, 'ILCReadyData.csv')
    if os.path.exists(_fb):
        _load_history_csv(pd.read_csv(_fb), 'sharedCSVs/ILCReadyData.csv [fallback]')

REGRESSION_FLOOR = 1   # attempt the fit from the first iteration onward — its
                       # WEIGHT in the blend (MIN_HISTORY_ITERS/FULL_TRUST_ITERS
                       # ramp + conditioning check) is what actually controls trust

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

    _w_pres        = _iter_weight(len(U_reg_list))   # smooth ramp, not a hard cutover
    LAMBDA_P_EFF   = LAMBDA_P * _w_pres

    print(f"  Fourier regression: {len(U_reg_list)} iteration files, {n_reg_pts} pts, "
          f"{A_reg.shape[1]} features  R²={r2:.3f}  N_FOURIER={N_FOURIER}  "
          f"confidence={_w_pres:.2f}  λ_eff={LAMBDA_P_EFF:.3f}")
    for _phi, _lbl in [(0.0,'start'), (0.25,'systole'), (0.5,'mid'), (0.75,'diastole')]:
        _j = J_pres_at_phase(_phi, fourier_coeffs, N_FOURIER)[0]
        print(f"    φ={_phi:.2f} ({_lbl}):  "
              f"∂P/∂epi={_j[0]:+.4f}  ∂P/∂trans={_j[1]:+.4f}  ∂P/∂endo={_j[2]:+.4f}")
else:
    fourier_coeffs = None
    LAMBDA_P_EFF   = 0.0
    r2, n_reg_pts  = 0.0, 0
    U_reg_all = P_reg_all = PHI_reg_all = U_reg_n = P_pred_reg = np.array([])
    print(f"  Only {len(U_reg_list)} iteration file(s) found — need ≥{REGRESSION_FLOOR} — "
          f"pressure correction disabled (λ=0, geometry only).")

# Geometry Jacobian source: SINDy per-point (smoothed) only.
# Fourier/linear regression from session data was removed — regression from
# closed-loop ILC data risks fitting ILC-history artefacts rather than the
# true physical actuator→geometry sensitivity, and SINDy per-point (smoothed)
# empirically achieved geometry convergence on 6_18 without any data-driven
# geometry regression.

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

act_n_avg = norm_act(act_phys_avg)

print(f"\n  Tracking error BEFORE correction (iteration {ITER_NUMBER}):")
rmse_before = {}
for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    rmse = np.sqrt(np.mean(e_phys[:, ci]**2))
    rmse_before[col] = rmse
    print(f"    {col:<14}: RMSE = {rmse:.4f} {unit}")
p_rmse_before = np.sqrt(np.mean(e_p**2))
print(f"    {'Pressure':<14}: RMSE = {p_rmse_before:.2f} mmHg")

# ── Geometry-convergence gate for pressure correction ─────────────────────────
# Pressure correction only unlocks once geometry has converged (all outputs
# below their RMSE thresholds).  Primary goal is geometry accuracy; pressure
# has a much wider acceptable error margin and its correction row can conflict
# with geometry correction when geometry is still far off target.
# Phase-specific gating (IVC + IVR only) is applied per-phase-point in the
# ILC loop below — this gate is the session-level prerequisite.
_geom_converged = all(rmse_before[col] < GEOMETRY_RMSE_THRESHOLD[col]
                      for col in OUTPUT_NAMES)

if not _geom_converged and LAMBDA_P_EFF > 0:
    print(f"\n  Pressure correction HELD OFF — geometry not yet converged:")
    for col in OUTPUT_NAMES:
        status = 'PASS' if rmse_before[col] < GEOMETRY_RMSE_THRESHOLD[col] else 'FAIL'
        print(f"    [{status}] {col}: {rmse_before[col]:.3f} (threshold {GEOMETRY_RMSE_THRESHOLD[col]})")
    LAMBDA_P_EFF = 0.0
elif _geom_converged and LAMBDA_P_EFF > 0:
    print(f"\n  Geometry converged — pressure correction ACTIVE  "
          f"λ={LAMBDA_P_EFF:.3f}  (IVC {IVC_PHASE} + IVR {IVR_PHASE} phases only)")
else:
    print(f"\n  Pressure correction inactive  (λ=0)")

# ══════════════════════════════════════════════════════════════════════════════
# ILC UPDATE
# ══════════════════════════════════════════════════════════════════════════════
pressure_active = LAMBDA_P_EFF > 0 and _COL_PRES is not None
if pressure_active:
    mode_str = (f"geometry + pressure — geometry converged, "
                f"λ={LAMBDA_P_EFF:.3f} at IVC/IVR only")
elif _geom_converged:
    mode_str = "geometry only — pressure inactive (no history data)"
else:
    mode_str = "geometry only — pressure held off (geometry not yet converged)"
print(f"\nComputing ILC correction  [{mode_str}] …")

delta_u_n = np.zeros_like(act_n_avg)

# Static FK Jacobian — phase-varying, Gaussian-smoothed:
# The SINDy model is nonlinear, so its gradient changes significantly across
# the cardiac cycle (e.g. volume sensitivity via trans collapses to near-zero
# at peak compression, then rises to 0.5 during filling). A single constant
# evaluated at the mean operating point loses this structure and specifically
# misdirects the volume correction. Per-point evaluation restores local
# accuracy; Gaussian smoothing (sigma=SINDY_SMOOTH_SIGMA phase samples)
# removes the finite-difference noise that caused oscillation without it.
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

# Geometry Jacobian: smoothed SINDy per-point (no data-driven regression blend)
print(f"\n  Geometry Jacobian: SINDy per-point, smoothed σ={SINDY_SMOOTH_SIGMA}"
      + ("  (adaptive weights ON)" if USE_ADAPTIVE_WEIGHTS else ""))

# Pre-compute adaptive weight maps
if USE_ADAPTIVE_WEIGHTS:
    _row_norms = np.linalg.norm(_J_static_all, axis=2)          # (n_traj, 3)
    _row_max   = _row_norms.max(axis=0, keepdims=True) + 1e-8
    _adaptive_w = (_row_norms / _row_max) * (1 - SENSITIVITY_FLOOR) + SENSITIVITY_FLOOR
    _adaptive_w = _adaptive_w * GEOM_WEIGHTS[None, :]

_J_geom_all = _J_static_all   # no blending — SINDy smoothed is the geometry Jacobian

# ── Smooth phase-varying lambda envelope for pressure ─────────────────────────
# A hard binary gate (0 outside IVC/IVR, 1 inside) creates step changes in
# lambda_i → step changes in delta_u_n → kinks in the corrected trajectory.
# Fix: Gaussian-filter the hard gate with mode='wrap' so all boundaries fade
# smoothly and the cycle wraps continuously (IVC at phi=0 connects to phi≈1).
_lambda_hard = np.array([
    1.0 if (IVC_PHASE[0] <= p <= IVC_PHASE[1] or IVR_PHASE[0] <= p <= IVR_PHASE[1])
    else 0.0
    for p in traj_phase
])
_lambda_envelope = gaussian_filter1d(_lambda_hard, sigma=PRESSURE_FADE_SIGMA, mode='wrap')
# _lambda_envelope is now in [0,1] with smooth fades at every IVC/IVR boundary

peak_lambda = _lambda_envelope.max()
n_nonzero = int(np.sum(_lambda_envelope > 0.01))
print(f"  Pressure lambda envelope: peak={peak_lambda:.3f}  "
      f"non-trivial at {n_nonzero}/{n_traj} pts  "
      f"(IVC {IVC_PHASE} + IVR {IVR_PHASE}, fade σ={PRESSURE_FADE_SIGMA} samples)")

for i in range(n_traj):
    J_geom = _J_geom_all[i]
    geom_w_i = _adaptive_w[i] if USE_ADAPTIVE_WEIGHTS else GEOM_WEIGHTS

    # Smoothly-varying lambda — zero outside isovolumic phases, fades in/out
    # at boundaries so there are no discontinuities in the correction signal.
    lambda_i = LAMBDA_P_EFF * _lambda_envelope[i] if pressure_active else 0.0

    if lambda_i > 1e-6:
        J_pres = J_pres_at_phase(traj_phase[i], fourier_coeffs, N_FOURIER)

        # Phase-specific geometry weights for the pressure-active phases.
        # IVC and IVR are ISOVOLUMIC — volume is near-constant BY PHYSIOLOGICAL
        # DEFINITION.  Keeping the volume weight non-zero during those phases
        # forces the pseudoinverse to simultaneously protect something that isn't
        # changing anyway, creating unnecessary conflict with pressure correction.
        # Zero-weighting the physiologically-stationary outputs frees the
        # corresponding degrees of freedom for pressure correction.
        #
        # IVC (0–4%): volume strictly constant → release volume constraint
        #   geom_w: [twist, height, volume=0]
        # IVR (42–44%): volume + height near-constant → release both
        #   geom_w: [twist, height=0, volume=0]
        #
        # Note: no extra smoothing needed — the lambda_envelope Gaussian fade
        # already smooths the pressure correction magnitude near the boundaries,
        # so the hard weight switch at the window edges doesn't cause trajectory kinks.
        phi_i = traj_phase[i]
        if IVC_PHASE[0] <= phi_i <= IVC_PHASE[1]:
            geom_w_p = geom_w_i * np.array([1.0, 1.0, 0.0])  # twist+height only
        elif IVR_PHASE[0] <= phi_i <= IVR_PHASE[1]:
            geom_w_p = geom_w_i * np.array([1.0, 0.0, 0.0])  # twist only
        else:
            geom_w_p = geom_w_i

        J_aug  = np.vstack([J_geom, lambda_i * J_pres])           # (4, 3)
        e_aug  = np.append(e_norm[i], e_p_norm[i])
        w_full = np.append(geom_w_p, 1.0)
    else:
        J_aug  = J_geom
        e_aug  = e_norm[i]
        w_full = geom_w_i

    J_w = J_aug * w_full[:, None]
    e_w = e_aug * w_full
    delta_u_n[i] = ILC_ALPHA * np.linalg.pinv(J_w) @ e_w

# ── Hard cap + cross-actuator redistribution ──────────────────────────────────
# When an actuator's correction is clamped (hits ±MAX_DELTA_U_MM), the output
# error that actuator was meant to address remains uncorrected.  Redistribute
# that residual to the unclamped actuators at each phase point using a
# reduced-dimension solve on the same blended Jacobian.
delta_u_max_n  = MAX_DELTA_U_MM / x_den[:3]
delta_u_n_raw  = delta_u_n.copy()
delta_u_n      = np.clip(delta_u_n_raw, -delta_u_max_n, delta_u_max_n)

n_clamped = int(np.sum(np.abs(delta_u_n_raw) > delta_u_max_n))
n_redistributed = 0
for i in range(n_traj):
    clipped = np.abs(delta_u_n_raw[i]) > delta_u_max_n
    if not clipped.any():
        continue
    free = ~clipped
    if not free.any():
        continue
    J_i = _J_geom_all[i]                              # (3, 3)
    e_residual = J_i @ delta_u_n_raw[i] - J_i @ delta_u_n[i]  # output error lost to clamping
    J_free = J_i[:, free]                             # (3, n_free)
    extra, _, _, _ = np.linalg.lstsq(J_free, e_residual, rcond=None)
    delta_u_n[i, free] = np.clip(
        delta_u_n[i, free] + extra,
        -delta_u_max_n[free], delta_u_max_n[free]
    )
    n_redistributed += 1

if n_clamped > 0:
    print(f"  Correction cap: {n_clamped} actuator×phase entries clamped to ±{MAX_DELTA_U_MM} mm  "
          f"({n_redistributed} phase pts had residual redistributed to free actuators)")

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

# ── Geometry prediction ───────────────────────────────────────────────────────
# Phase-varying linearised prediction using the SAME per-phase blended Jacobian
# that drove the correction: y_norm_new[i] ≈ y_norm_current[i] + J[i] · Δu_norm[i]
_J_blend_all = _J_geom_all   # already computed before the ILC loop
delta_u_norm_pred = act_n_new - act_n_avg                   # (n, 3), normalised
y_norm_corrected  = norm_y(y_phys_avg) + np.einsum('noc,nc->no', _J_blend_all, delta_u_norm_pred)
y_phys_corrected  = denorm_y(y_norm_corrected)
_pred_method      = f"SINDy per-point smoothed (σ={SINDY_SMOOTH_SIGMA})"
e_corrected = traj_phys - y_phys_corrected

print(f"\n  Predicted geometry after correction [{_pred_method}]:")
for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    r_before = rmse_before[col]
    r_after  = np.sqrt(np.mean(e_corrected[:, ci]**2))
    pct = (r_before - r_after) / r_before * 100 if r_before > 0 else 0.0
    print(f"    {col:<14}: {r_before:.4f} → {r_after:.4f} {unit}  ({pct:+.1f}%)")

# Predict pressure after correction using regression model
if pressure_active:
    A_new_reg      = _fourier_design(traj_phase, norm_act(act_phys_new), N_FOURIER)
    p_pred_new     = (A_new_reg @ fourier_coeffs) * P_NORM_REF
    p_pred_rmse    = np.sqrt(np.mean((traj_p - p_pred_new)**2))
    print(f"    {'Pressure':<14}: {p_rmse_before:.2f} → {p_pred_rmse:.2f} mmHg  "
          f"(regression estimate)")
else:
    p_pred_new = None

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
pres_str = f"{p_rmse_before:.2f} mmHg  (threshold {PRESSURE_RMSE_THRESHOLD})"
if pressure_active:
    print(f"    [{'PASS' if p_ok else 'FAIL'}] {'Pressure':<14}: {pres_str}")
else:
    print(f"    [----] {'Pressure':<14}: {pres_str}  (passive — λ=0)")

if geo_converged and (p_ok or not pressure_active):
    print("\n" + "═"*62)
    print("  CONVERGED" + (" — geometry + pressure." if pressure_active
                           else " — geometry only."))
    if not pressure_active:
        print("  Pressure correction will activate next iteration.")
    print("═"*62)
else:
    print(f"\n  Not yet converged — run another iteration.")
    if not pressure_active:
        print("  Pressure correction will activate next iteration "
              "(history data now available).")

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

# Share x within left column (tracking) and within right actuator rows only
for row in range(1, n_rows):
    axes[row, 0].sharex(axes[0, 0])
for row in range(1, min(3, n_rows)):
    axes[row, 1].sharex(axes[0, 1])

# ── Left: tracking ────────────────────────────────────────────────────────────
for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
    ax = axes[ci, 0]
    ax.plot(traj_phase, traj_phys[:, ci],        'k-',  lw=2.5, label='Desired')
    ax.plot(traj_phase, y_phys_avg[:, ci],        'b--', lw=1.8, label='Measured')
    ax.plot(traj_phase, y_phys_corrected[:, ci],  'r-',  lw=1.5, label='FK predicted after ILC', alpha=0.8)
    rmse   = rmse_before[col]
    thresh = GEOMETRY_RMSE_THRESHOLD[col]
    ok     = rmse < thresh
    ax.set_ylabel(f'{col} ({unit})', fontsize=10)
    ax.set_title(f'RMSE = {rmse:.3f} {unit}  [{"PASS" if ok else "FAIL"} < {thresh}]',
                 fontsize=9, color='green' if ok else 'red')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

if _COL_PRES:
    ax_p = axes[3, 0]
    ax_p.plot(traj_phase, traj_p,   'k-',  lw=2.5, label='Desired')
    ax_p.plot(traj_phase, actual_p, 'b--', lw=1.8, label='Measured')
    if p_pred_new is not None:
        ax_p.plot(traj_phase, p_pred_new, 'r-', lw=1.5,
                  label=f'Regression predicted after ILC  (RMSE→{p_pred_rmse:.1f} mmHg)',
                  alpha=0.85)
    ax_p.set_ylabel('Pressure (mmHg)', fontsize=10)
    ax_p.set_xlabel('Cycle phase', fontsize=10)
    pres_label = (f'RMSE = {p_rmse_before:.2f} mmHg  '
                  f'[{"PASS" if p_ok else "FAIL"} < {PRESSURE_RMSE_THRESHOLD}]  '
                  f'(λ={LAMBDA_P_EFF})')
    ax_p.set_title(pres_label, fontsize=9, color='green' if p_ok else 'red')
    ax_p.legend(fontsize=8); ax_p.grid(True, alpha=0.3)

# ── Right: actuator signals (rows 0-2) + regression info (row 3) ──────────────
motor_names = ['Epi', 'Trans', 'Endo']
for ci in range(3):
    ax2 = axes[ci, 1]
    ax2.plot(traj_phase, act_phys_avg[:, ci], 'b--', lw=1.8, label='Current')
    ax2.plot(traj_phase, act_phys_new[:, ci], 'r-',  lw=1.8, label='Corrected')
    ax2.axhline(ACT_MIN[ci], color='grey', lw=1, ls=':')
    ax2.axhline(ACT_MAX[ci], color='grey', lw=1, ls=':')
    delta = np.abs(act_phys_new[:, ci] - act_phys_avg[:, ci]).max()
    ax2.set_title(f'Max |Δ| = {delta:.2f} mm', fontsize=9, color='grey')
    ax2.set_ylabel(f'{motor_names[ci]} (mm)', fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

if _COL_PRES and n_rows == 4:
    ax_r = axes[3, 1]
    if pressure_active:
        # Plot phase-varying J_reg(φ) for each actuator
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
        ax_r.text(0.5, 0.5, 'Pressure correction inactive\n(no history data)',
                  ha='center', va='center', transform=ax_r.transAxes, fontsize=11)
        ax_r.set_axis_off()
axes[n_rows-1, 0].set_xlabel('Cycle phase', fontsize=10)
axes[min(2, n_rows-1), 1].set_xlabel('Cycle phase', fontsize=10)

axes[0, 0].set_title('Deformation tracking', fontsize=10)
axes[0, 1].set_title('Actuator signals', fontsize=10)

plt.suptitle(
    f'ILC Correction — Iteration {ITER_NUMBER}  '
    f'(α={ILC_ALPHA},  λ={LAMBDA_P_EFF},  Q={Q_CUTOFF},  '
    f'{"R²_P="+f"{r2:.2f}" if pressure_active else "geometry only"})',
    fontsize=12)
plt.tight_layout()

fig_path = os.path.join(BASE, f'ilcCorrection_iter{ITER_NUMBER:03d}.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Figure → ilcCorrection_iter{ITER_NUMBER:03d}.png")
plt.show()
