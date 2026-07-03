"""
plotConflictActuatorSpace.py
──────────────────────────────────────────────────────────────────────────────
Maps the pressure-geometry conflict ratio directly in ACTUATOR POSITION SPACE
rather than against cycle phase.

The SINDy model has no concept of time or phase — it maps
(epi, trans, endo, pressure) → (twist, height, volume). Conflict ratio is
purely a function of where the actuators are in physical space. Plotting it
against phase was a convenient shorthand but fundamentally mislabelled: what
we were seeing was the variation of conflict along the PATH the actuators
trace through their operating space, not a time-based phenomenon.

This script sweeps the full actuator operating space (200–248mm per axis)
and computes the conflict ratio at each point, independent of any trajectory.
The CURRENT MEASURED trajectory is then overlaid as a path through this map —
showing which regions of the space the device currently visits and whether
those regions are low or high conflict.

Figure layout (2 columns × 3 rows):
  Left column:  2D heatmaps — conflict ratio across pairs of actuators
                (epi-trans, epi-endo, trans-endo) with third held at mean.
                Colour: low conflict = green (pressure-safe), high = red (risky).
                White path = current measured trajectory projected onto that plane.

  Right column: 1D sweeps — conflict ratio as each actuator is varied from
                200→248mm while the other two are held at their mean values.
                Shows the marginal effect of each actuator's position.

J_pres used: phase-averaged Fourier regression sensitivity, so it represents
             the average direction the pressure regression points across the
             full cardiac cycle, independent of phase.
"""

import sys, os, pathlib, re, pickle
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
SESSION_DIR    = os.environ.get('ILC_SESSION_DIR', '6_18')
N_FOURIER      = 1
P_NORM_REF     = 125.0
ACT_MIN        = np.array([200.0, 202.0, 200.0])
ACT_MAX        = np.array([248.0, 248.0, 248.0])
GRID_RES       = 40   # points per axis in 2D heatmaps
SINDY_SIGMA    = 4

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
BASE     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = pathlib.Path(BASE) / 'saved_models'
EXP_DATA = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Exp_data' / SESSION_DIR

norm = np.load(SAVE_DIR / 'norm_constants.npz')
x_min, x_den = norm['x_min'], norm['x_den']
y_min, y_den = norm['y_min'], norm['y_den']

def norm_act(a): return (np.asarray(a) - x_min[:3]) / x_den[:3]
def denorm_act(a): return np.asarray(a) * x_den[:3] + x_min[:3]
def norm_y(y): return (np.asarray(y) - y_min) / y_den

with open(SAVE_DIR / 'sindy_data.pkl', 'rb') as f:
    sd = pickle.load(f)
sindy_sc, poly_lib, sindy_res = sd['sc'], sd['poly_lib'], sd['results']
OUT_NAMES = ['Twist_deg', 'Height_mm', 'Volume_mL']

def sindy_pred(x):
    xs = sindy_sc.transform(np.asarray(x).reshape(-1, 4))
    t  = np.asarray(poly_lib.transform(xs))
    return np.stack([sindy_res[c]['sy'].inverse_transform((t @ sindy_res[c]['coef']).reshape(-1,1)).ravel()
                     for c in OUT_NAMES], axis=1)

FD_EPS = 1e-4
def J_geom_at(act_n_3, p_n_scalar):
    """Geometry Jacobian (3×3 normalised) at a given normalised actuator position."""
    p_phys = p_n_scalar * x_den[3] + x_min[3]
    def fwd(a): return norm_y(sindy_pred(np.append(denorm_act(a), p_phys).reshape(1,-1))[0])
    f0 = fwd(act_n_3); J = np.zeros((3,3))
    for j in range(3):
        ap = act_n_3.copy(); ap[j] += FD_EPS
        J[:,j] = (fwd(ap) - f0) / FD_EPS
    return J

# ══════════════════════════════════════════════════════════════════════════════
# LOAD HISTORY + FIT PRESSURE REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
def _find_col(df, c):
    low = {x.lower(): x for x in df.columns}
    for n in c:
        if n.lower() in low: return low[n.lower()]
    return None

def _sort_key(p):
    itr = p.parent
    dm = re.match(r'(\d+)_(\d+)$', itr.parent.name)
    dk = (int(dm.group(1)), int(dm.group(2))) if dm else (0, 0)
    m = re.search(r'(\d+)$', itr.name); num = int(m.group(1)) if m else -1
    return (dk, bool(re.match(r'p_itr', itr.name, re.I)), num)

GRID_PHASE = np.linspace(0, 1, 51)
U_list, P_list, PHI_list = [], [], []
traj_act_phys = None  # will store latest iteration's actuator trajectory

for fpath in sorted(EXP_DATA.rglob('ILCReadyData.csv'), key=_sort_key):
    df = pd.read_csv(fpath)
    ce = _find_col(df,['epi_mm','epi']); ct = _find_col(df,['trans_mm','trans'])
    cn = _find_col(df,['endo_mm','endo']); cp = _find_col(df,['pressure','pressure_mmhg'])
    cph= _find_col(df,['phase','time','time_s'])
    if any(c is None for c in [ce,ct,cn,cp]): continue
    ph = df[cph].values if cph else np.linspace(0,1,len(df))
    if ph.max()>1.5: ph=(ph-ph[0])/(ph[-1]-ph[0])
    rs = lambda v: interp1d(ph,v,'linear',bounds_error=False,fill_value='extrapolate')(GRID_PHASE)
    act = np.column_stack([rs(df[ce].values), rs(df[ct].values), rs(df[cn].values)])
    U_list.append(act); P_list.append(rs(df[cp].values)); PHI_list.append(GRID_PHASE)
    traj_act_phys = act   # keep updating — last file = most recent iteration

assert traj_act_phys is not None, f"No data found in {EXP_DATA}"
n_iters = len(U_list)
print(f"Session {SESSION_DIR}: {n_iters} iteration(s) loaded")

U_all   = np.vstack(U_list)
P_all   = np.concatenate(P_list) / P_NORM_REF
PHI_all = np.concatenate(PHI_list)
U_n_all = norm_act(U_all)

def fourier_design(phi, u_n, n_harm):
    cols = []
    for j in range(3):
        cols.append(u_n[:,j])
        for k in range(1, n_harm+1):
            cols.append(u_n[:,j]*np.sin(2*np.pi*k*phi))
            cols.append(u_n[:,j]*np.cos(2*np.pi*k*phi))
    cols.append(np.ones(len(phi)))
    return np.column_stack(cols)

A_p = fourier_design(PHI_all, U_n_all, N_FOURIER)
pres_coeffs, _, _, _ = np.linalg.lstsq(A_p, P_all, rcond=None)

def J_pres_at_phase(phi_s):
    """Phase-varying J_pres (1×3 normalised) from Fourier regression."""
    J = np.zeros(3); idx = 0
    for j in range(3):
        J[j] = pres_coeffs[idx]; idx += 1
        for k in range(1, N_FOURIER+1):
            J[j] += pres_coeffs[idx]*np.sin(2*np.pi*k*phi_s); idx += 1
            J[j] += pres_coeffs[idx]*np.cos(2*np.pi*k*phi_s); idx += 1
    return J

# Phase-averaged J_pres — represents the average pressure sensitivity direction
# across the full cycle, independent of any specific phase point.
J_pres_avg = np.mean([J_pres_at_phase(phi) for phi in GRID_PHASE], axis=0)
print(f"Phase-averaged J_pres: epi={J_pres_avg[0]:+.3f}  "
      f"trans={J_pres_avg[1]:+.3f}  endo={J_pres_avg[2]:+.3f}")

# Mean pressure and mean actuator positions (for holding fixed in sweeps)
mean_p_n   = (U_list[-1][:,0]*0 + P_list[-1]).mean() / P_NORM_REF  # use latest
mean_act_n = norm_act(traj_act_phys).mean(axis=0)
mean_p_phys= mean_p_n * x_den[3] + x_min[3]
print(f"Mean actuator (phys): {denorm_act(mean_act_n)}")
print(f"Mean pressure (phys): {mean_p_phys:.1f} mmHg")

# ══════════════════════════════════════════════════════════════════════════════
# CONFLICT RATIO COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def conflict_ratio(J_geom_3x3, J_pres_row):
    """Geometry disturbance per unit pressure correction."""
    norm_jp = np.linalg.norm(J_pres_row)
    if norm_jp < 1e-8: return 0.0
    du_pres = J_pres_row / (J_pres_row @ J_pres_row)   # min-norm actuator change for ΔP=1
    return np.linalg.norm(J_geom_3x3 @ du_pres)

# ── 2D GRIDS (one per actuator pair, third held at mean)
act_names   = ['Epi (mm)', 'Trans (mm)', 'Endo (mm)']
act_keys    = ['epi', 'trans', 'endo']
act_idx     = {'epi': 0, 'trans': 1, 'endo': 2}
pair_combos = [('epi','trans',2), ('epi','endo',1), ('trans','endo',0)]
# (x-axis, y-axis, fixed-idx)

grids_2d   = {}
act_ranges = {}
for ai, (lo, hi) in enumerate(zip(ACT_MIN, ACT_MAX)):
    act_ranges[act_keys[ai]] = np.linspace(lo, hi, GRID_RES)

for x_key, y_key, fixed_idx in pair_combos:
    xi = act_idx[x_key]; yi = act_idx[y_key]
    x_vals = act_ranges[x_key]; y_vals = act_ranges[y_key]
    CR = np.zeros((GRID_RES, GRID_RES))
    for ii, xv in enumerate(x_vals):
        for jj, yv in enumerate(y_vals):
            act_phys = denorm_act(mean_act_n).copy()
            act_phys[xi] = xv; act_phys[yi] = yv
            act_n = norm_act(act_phys)
            Jg = J_geom_at(act_n, mean_p_n)
            CR[jj, ii] = conflict_ratio(Jg, J_pres_avg)
    grids_2d[(x_key, y_key)] = (x_vals, y_vals, CR)
    print(f"  Grid {x_key}-{y_key}: conflict range [{CR.min():.3f}, {CR.max():.3f}]")

# ── 1D SWEEPS
sweeps_1d = {}
for ai, key in enumerate(act_keys):
    vals = act_ranges[key]; CR = np.zeros(len(vals))
    for ii, v in enumerate(vals):
        act_phys = denorm_act(mean_act_n).copy(); act_phys[ai] = v
        Jg = J_geom_at(norm_act(act_phys), mean_p_n)
        CR[ii] = conflict_ratio(Jg, J_pres_avg)
    sweeps_1d[key] = (vals, CR)

# ── CURRENT TRAJECTORY in actuator space (physical mm)
traj_epi   = traj_act_phys[:, 0]
traj_trans = traj_act_phys[:, 1]
traj_endo  = traj_act_phys[:, 2]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle(f'Pressure–Geometry Conflict in Actuator Position Space\n'
             f'session {SESSION_DIR}  ({n_iters} iterations) — '
             f'J_pres = phase-averaged Fourier regression',
             fontsize=12, fontweight='bold')

pair_titles = ['Epi vs Trans', 'Epi vs Endo', 'Trans vs Endo']
sweep_titles= ['Epi sweep (trans,endo at mean)', 'Trans sweep (epi,endo at mean)', 'Endo sweep (epi,trans at mean)']

# Shared colour scale across all 2D heatmaps
all_cr = np.concatenate([g[2].ravel() for g in grids_2d.values()])
cr_min, cr_max = all_cr.min(), all_cr.max()
cmap = plt.cm.RdYlGn_r   # red=high conflict, green=low conflict

for row, ((x_key, y_key, fixed_idx), title) in enumerate(zip(pair_combos, pair_titles)):
    # ── Left: 2D heatmap
    ax = axes[row, 0]
    x_vals, y_vals, CR = grids_2d[(x_key, y_key)]
    im = ax.pcolormesh(x_vals, y_vals, CR, cmap=cmap, vmin=cr_min, vmax=cr_max, shading='auto')

    # Overlay current trajectory
    xi = act_idx[x_key]; yi = act_idx[y_key]
    traj_x = traj_act_phys[:, xi]; traj_y = traj_act_phys[:, yi]
    ax.plot(traj_x, traj_y, 'w-', lw=1.5, alpha=0.7, label='Current trajectory')
    ax.plot(traj_x[0], traj_y[0], 'wo', ms=7, label='Start (φ=0)')
    ax.plot(traj_x[len(traj_x)//2], traj_y[len(traj_y)//2], 'w^', ms=7, label='φ=0.5')

    # Mark mean actuator position
    ax.plot(denorm_act(mean_act_n)[xi], denorm_act(mean_act_n)[yi], 'w+', ms=12, mew=2,
            label='Mean position')

    plt.colorbar(im, ax=ax, label='Conflict ratio')
    ax.set_xlabel(f'{act_names[xi]}'); ax.set_ylabel(f'{act_names[yi]}')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc='lower right', framealpha=0.7)

    # ── Right: 1D sweep for corresponding primary axis
    ax2 = axes[row, 1]
    sweep_key = x_key   # primary actuator for this row
    sv, scr = sweeps_1d[sweep_key]
    ax2.plot(sv, scr, color='crimson', lw=2.5)
    ax2.fill_between(sv, 0, scr, alpha=0.2, color='crimson')
    ax2.axhline(np.mean(all_cr), color='grey', ls='--', lw=1.5, label=f'Mean = {np.mean(all_cr):.3f}')
    # Mark where the current trajectory spans on this axis
    traj_v = traj_act_phys[:, xi]
    ax2.axvspan(traj_v.min(), traj_v.max(), alpha=0.15, color='steelblue',
                label=f'Trajectory range [{traj_v.min():.0f}–{traj_v.max():.0f}]mm')
    # Annotate min
    min_idx = np.argmin(scr)
    ax2.annotate(f'Min={scr[min_idx]:.3f}\n@ {sv[min_idx]:.0f}mm',
                 xy=(sv[min_idx], scr[min_idx]),
                 xytext=(sv[min_idx]+3, scr[min_idx]+0.05),
                 fontsize=8, color='darkgreen', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.2))
    ax2.set_xlabel(f'{act_names[xi]}'); ax2.set_ylabel('Conflict ratio')
    ax2.set_title(f'{sweep_titles[xi]}', fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(BASE, 'conflictActuatorSpace.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")

# ── Console summary
print(f"\n{'═'*60}")
print("  Global minimum conflict region:")
for x_key, y_key, _ in pair_combos:
    x_vals, y_vals, CR = grids_2d[(x_key, y_key)]
    min_idx = np.unravel_index(CR.argmin(), CR.shape)
    print(f"  {x_key}-{y_key}: min={CR.min():.3f} "
          f"@ {x_key}={x_vals[min_idx[1]]:.0f}mm, {y_key}={y_vals[min_idx[0]]:.0f}mm")
print(f"\n  Trajectory range (latest iteration):")
for ai, key in enumerate(act_keys):
    tv = traj_act_phys[:, ai]
    print(f"  {key}: [{tv.min():.0f}, {tv.max():.0f}] mm")
print(f"{'═'*60}")
plt.show()
