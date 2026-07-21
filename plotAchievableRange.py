"""
plotAchievableRange.py
──────────────────────────────────────────────────────────────────────────────
Answers: "What twist / height / volume is physically achievable given the
actuator limits, and does the desired trajectory fit within that range?"

At each cycle phase point, given the CURRENT measured actuator positions:
  achievable_min[oi] = y_current[oi] + Σ_j  min(J[oi,j]·ΔuMin[j],
                                                  J[oi,j]·ΔuMax[j])
  achievable_max[oi] = y_current[oi] + Σ_j  max(J[oi,j]·ΔuMin[j],
                                                  J[oi,j]·ΔuMax[j])

where ΔuMin[j] = ACT_MIN[j] - u_current[j]   (how far down actuator j can go)
      ΔuMax[j] = ACT_MAX[j] - u_current[j]   (how far up   actuator j can go)

This is the MARGINAL achievable range for each output independently.
It answers: "If I could move all actuators freely to their limits in the most
favourable direction for this one output, what range could I reach?"

Green shaded envelope = physically achievable.
Black line   = desired trajectory (with HEIGHT_OFFSET / VOLUME_OFFSET applied).
Blue dashed  = current measured.
Red segments = where the desired trajectory exits the achievable zone.
"""

import sys, os, pathlib, re, pickle
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig_config import HEIGHT_OFFSET, VOLUME_OFFSET

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
SESSION_DIR      = os.environ.get('ILC_SESSION_DIR', '7_3')
SIM_CASE         = os.environ.get('ILC_CASE', 'healthy')
SINDY_SIGMA      = 4
ACT_MIN          = np.array([200.0, 202.0, 200.0])
ACT_MAX          = np.array([248.0, 248.0, 248.0])

# Prospective volume measurement offset — simulates the effect of running
# markerProcessing.py with a higher VOLUME_MEASURED_OFFSET without needing
# a new recording.  Set to 0 to show current state, or to the ADDITIONAL
# mL being added (e.g. +20 if going from old 10→new 30).
PROSPECTIVE_VOLUME_EXTRA_ML = 0.0   # 0 = show current actual state; set >0 to simulate offset change

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL + NORMALISATION
# ══════════════════════════════════════════════════════════════════════════════
BASE     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = pathlib.Path(BASE) / 'saved_models'
SHARED   = os.path.normpath(os.path.join(BASE, '..', '..', 'sharedCSVs'))
EXP_DATA = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Exp_data' / SESSION_DIR
ENGD_DIR = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Engineered_trajs'
ENG_CSV  = {'healthy':   ENGD_DIR / 'engineered_data_healthy.csv',
            'diastolic': ENGD_DIR / 'engineered_data_diastolic_dysfunction.csv',
            'systolic':  ENGD_DIR / 'engineered_data_systolic_dysfunction.csv'
            }.get(SIM_CASE, ENGD_DIR / 'engineered_data_healthy.csv')

norm = np.load(SAVE_DIR / 'norm_constants.npz')
x_min, x_den = norm['x_min'], norm['x_den']
y_min, y_den = norm['y_min'], norm['y_den']

def norm_act(a): return (np.asarray(a) - x_min[:3]) / x_den[:3]
def denorm_act(a): return np.asarray(a) * x_den[:3] + x_min[:3]
def norm_y(y): return (np.asarray(y) - y_min) / y_den
def denorm_y(n): return np.asarray(n) * y_den + y_min

with open(SAVE_DIR / 'sindy_data.pkl', 'rb') as f:
    sd = pickle.load(f)
OUT_SINDY = ['Twist_deg', 'Height_mm', 'Volume_mL']
def sindy_pred(x):
    xs = sd['sc'].transform(np.asarray(x).reshape(-1, 4))
    t  = np.asarray(sd['poly_lib'].transform(xs))
    return np.stack([sd['results'][c]['sy'].inverse_transform(
        (t @ sd['results'][c]['coef']).reshape(-1, 1)).ravel()
        for c in OUT_SINDY], axis=1)

FD_EPS = 1e-4
def jacobian_at(act_n, p_n):
    pp = p_n * x_den[3] + x_min[3]
    def fwd(a): return norm_y(sindy_pred(np.append(denorm_act(a), pp).reshape(1,-1))[0])
    f0 = fwd(act_n); J = np.zeros((3, 3))
    for j in range(3):
        ap = act_n.copy(); ap[j] += FD_EPS
        J[:, j] = (fwd(ap) - f0) / FD_EPS
    return J

# ══════════════════════════════════════════════════════════════════════════════
# LOAD CURRENT MEASURED TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════
def _fc(df, cands):
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    return None

# Try latest session file first, fall back to sharedCSVs
_source = None
if EXP_DATA.exists():
    def _sk(p):
        itr = p.parent
        dm  = re.match(r'(\d+)_(\d+)$', itr.parent.name)
        dk  = (int(dm.group(1)), int(dm.group(2))) if dm else (0,0)
        m   = re.search(r'(\d+)$', itr.name)
        return (dk, bool(re.match(r'p_itr', itr.name, re.I)), int(m.group(1)) if m else -1)
    _files = sorted(EXP_DATA.rglob('ILCReadyData.csv'), key=_sk)
    if _files:
        _source = str(_files[-1])
        print(f"Using latest session file: {_files[-1].parent.name}")
if _source is None:
    _source = os.path.join(SHARED, 'ILCReadyData.csv')
    print(f"Using sharedCSVs: {_source}")

meas_df = pd.read_csv(_source)
meas_df.columns = meas_df.columns.str.strip()
ce  = _fc(meas_df, ['epi_mm','epi']);  ct = _fc(meas_df, ['trans_mm','trans'])
cn  = _fc(meas_df, ['endo_mm','endo']); cp = _fc(meas_df, ['pressure','pressure_mmhg'])
cw  = _fc(meas_df, ['twist','twist_deg']); ch = _fc(meas_df, ['height','height_mm'])
cv  = _fc(meas_df, ['volume','volume_mL']); cph = _fc(meas_df, ['phase','time','time_s'])

ph = meas_df[cph].values if cph else np.linspace(0, 1, len(meas_df))
if ph.max() > 1.5: ph = (ph - ph[0]) / (ph[-1] - ph[0])

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DESIRED TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════
eng = pd.read_csv(ENG_CSV)
t_eng = eng['time'].values
traj_phase = (t_eng - t_eng[0]) / (t_eng[-1] - t_eng[0])
n_traj = len(traj_phase)

rs = lambda v: interp1d(ph, v, 'linear', bounds_error=False, fill_value='extrapolate')(traj_phase)

act_phys   = np.column_stack([rs(meas_df[ce].values),
                               rs(meas_df[ct].values),
                               rs(meas_df[cn].values)])
geom_phys  = np.column_stack([rs(meas_df[cw].values),
                               rs(meas_df[ch].values),
                               rs(meas_df[cv].values)]) if None not in [cw,ch,cv] else None
p_meas     = rs(meas_df[cp].values) if cp else np.full(n_traj, 50.)

# Apply prospective volume offset (simulates the effect of the new
# VOLUME_MEASURED_OFFSET in markerProcessing.py on future recordings)
if geom_phys is not None and PROSPECTIVE_VOLUME_EXTRA_ML != 0:
    geom_phys = geom_phys.copy()
    geom_phys[:, 2] += PROSPECTIVE_VOLUME_EXTRA_ML
    print(f"Prospective volume shift applied: +{PROSPECTIVE_VOLUME_EXTRA_ML:.0f} mL "
          f"(simulating new VOLUME_MEASURED_OFFSET)")

traj_des = np.column_stack([
    eng['twist'].values,
    eng['height'].values + HEIGHT_OFFSET,
    eng['volume'].values + VOLUME_OFFSET,
])

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE ACHIEVABLE RANGE AT EACH PHASE POINT
# ══════════════════════════════════════════════════════════════════════════════
print("Computing SINDy Jacobians per phase point …")
act_n     = norm_act(act_phys)           # (n, 3) normalised
p_n       = (p_meas - x_min[3]) / x_den[3]
act_n_min = norm_act(ACT_MIN)            # (3,)
act_n_max = norm_act(ACT_MAX)            # (3,)

J_raw = np.zeros((n_traj, 3, 3))
for i in range(n_traj):
    J_raw[i] = jacobian_at(act_n[i], p_n[i])

# Gaussian smooth (same as ilcCorrection.py)
J_smooth = np.zeros_like(J_raw)
for oi in range(3):
    for ci in range(3):
        J_smooth[:, oi, ci] = gaussian_filter1d(J_raw[:, oi, ci],
                                                  sigma=SINDY_SIGMA, mode='wrap')

# For each phase i and each geometry output oi:
#   achievable range = y_current[oi] + sum_j max/min over Δu in [ΔuMin, ΔuMax]
# where ΔuMin = act_n_min - act_n[i],  ΔuMax = act_n_max - act_n[i]
ach_min = np.zeros((n_traj, 3))   # normalised output space
ach_max = np.zeros((n_traj, 3))

y_current_n = norm_y(geom_phys) if geom_phys is not None else np.zeros((n_traj, 3))

for i in range(n_traj):
    du_min = act_n_min - act_n[i]   # (3,)  most negative Δu possible
    du_max = act_n_max - act_n[i]   # (3,)  most positive Δu possible
    for oi in range(3):
        contrib_lo = np.minimum(J_smooth[i, oi, :] * du_min,
                                J_smooth[i, oi, :] * du_max)
        contrib_hi = np.maximum(J_smooth[i, oi, :] * du_min,
                                J_smooth[i, oi, :] * du_max)
        ach_min[i, oi] = y_current_n[i, oi] + contrib_lo.sum()
        ach_max[i, oi] = y_current_n[i, oi] + contrib_hi.sum()

# Convert achievable range to physical units
ach_min_phys = denorm_y(ach_min)   # (n, 3)
ach_max_phys = denorm_y(ach_max)   # (n, 3)

# ══════════════════════════════════════════════════════════════════════════════
# DETECT WHERE DESIRED EXITS ACHIEVABLE ZONE
# ══════════════════════════════════════════════════════════════════════════════
above = traj_des > ach_max_phys    # desired exceeds max achievable
below = traj_des < ach_min_phys    # desired below min achievable
infeasible = above | below         # (n, 3)

OUT_NAMES  = ['Twist (deg)', 'Height (mm)', 'Volume (mL)']
OUT_KEYS   = ['twist',        'height',      'volume']
OUT_THRESH = [5.0, 2.0, 10.0]   # RMSE thresholds from ilcCorrection.py

print("\nFeasibility summary at each phase point:")
for oi, nm in enumerate(OUT_NAMES):
    n_inf = int(infeasible[:, oi].sum())
    n_above = int(above[:, oi].sum())
    n_below = int(below[:, oi].sum())
    gap_above = max(0, (traj_des[:, oi] - ach_max_phys[:, oi]).max())
    gap_below = max(0, (ach_min_phys[:, oi] - traj_des[:, oi]).max())
    print(f"  {nm:15s}: {n_inf:3d}/{n_traj} infeasible pts  "
          f"(above max: {n_above} pts, max gap {gap_above:.2f};  "
          f"below min: {n_below} pts, max gap {gap_below:.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
fig.suptitle(f'Achievable output range vs desired trajectory\n'
             f'Session {SESSION_DIR} · {SIM_CASE} · '
             f'HEIGHT_OFFSET={HEIGHT_OFFSET} mm · VOLUME_OFFSET={VOLUME_OFFSET} mL\n'
             f'Green = physically achievable  ·  Red = infeasible (desired outside range)',
             fontsize=11, fontweight='bold')

for oi, (ax, nm) in enumerate(zip(axes, OUT_NAMES)):
    # Green shaded achievable envelope
    ax.fill_between(traj_phase, ach_min_phys[:, oi], ach_max_phys[:, oi],
                    alpha=0.30, color='limegreen', label='Achievable range')
    ax.plot(traj_phase, ach_min_phys[:, oi], color='darkgreen', lw=1.0, alpha=0.6)
    ax.plot(traj_phase, ach_max_phys[:, oi], color='darkgreen', lw=1.0, alpha=0.6)

    # Current measured
    if geom_phys is not None:
        ax.plot(traj_phase, geom_phys[:, oi], 'b--', lw=1.8, label='Current measured')

    # Desired trajectory — green where feasible, red where infeasible
    des = traj_des[:, oi]
    infs = infeasible[:, oi]
    ax.plot(traj_phase, des, 'k-', lw=2.5, alpha=0.4, label='Desired (full)')
    # Overlay red segments where infeasible
    for i in range(n_traj - 1):
        if infs[i] or infs[i+1]:
            ax.plot(traj_phase[i:i+2], des[i:i+2], 'r-', lw=3.0)
    if infs.any():
        ax.plot([], [], 'r-', lw=3.0, label='Desired — INFEASIBLE')

    # Annotate worst gap
    gap_above = (des - ach_max_phys[:, oi]).clip(min=0)
    gap_below = (ach_min_phys[:, oi] - des).clip(min=0)
    worst = max(gap_above.max(), gap_below.max())
    n_inf_oi = int(infs.sum())
    col = 'red' if n_inf_oi > 0 else 'darkgreen'
    title = f'{nm}  ·  {n_inf_oi}/{n_traj} phase pts infeasible'
    if worst > 0:
        title += f'  ·  max gap = {worst:.2f}'
    ax.set_title(title, fontsize=10, color=col)
    ax.set_ylabel(nm, fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Cycle phase', fontsize=10)
plt.tight_layout()

out_path = os.path.join(BASE, 'achievableRange.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")
plt.show()
