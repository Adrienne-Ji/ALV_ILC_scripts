"""
plotPressureConflict.py
──────────────────────────────────────────────────────────────────────────────
Explores whether pressure correction can be achieved WITHOUT compromising
geometry alignment — and which phases of the cycle allow it most cleanly.

With 3 actuators controlling 3 geometry outputs + 1 pressure output (4
targets, 3 inputs), you cannot independently control all 4 simultaneously.
Some trade-off is always necessary.  The question is WHERE in the cycle that
trade-off is cheapest.

Key metric: CONFLICT RATIO
  If we apply the minimum actuator change needed to achieve a unit pressure
  correction (using the pseudoinverse of J_pres), how much does geometry
  inevitably move as a side-effect?

  conflict_ratio[i] = ||J_geom[i] @ Δu_pres[i]|| / |target_ΔP|

  Low  → pressure correction barely disturbs geometry at that phase
  High → pressure and geometry corrections strongly interfere

IVC (0%–4%) and IVR (42%–44%) are the isovolumic phases where the desired
geometry barely changes anyway — so even if pressure correction causes some
geometry movement there, it matters less.

Row 1: Pressure Jacobian J_pres(φ) from Fourier regression — which actuators
        drive pressure and when.

Row 2: Cosine similarity between J_pres(φ) and each geometry Jacobian row —
        how parallel the pressure and geometry correction directions are.
        Near ±1 = pushing in the same direction → strong coupling.
        Near 0  = orthogonal → pressure correction doesn't look like geometry.

Row 3: Conflict ratio across the full cycle.
        IVC + IVR windows highlighted — confirm these are low-conflict regions.
        Also shows the geometry disturbance split by output.

Row 4: At IVC/IVR specifically, shows what a pure pressure correction does to
        each geometry output — the "cost" of pressure correction there.
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

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS  — keep in sync with ilcCorrection.py
# ══════════════════════════════════════════════════════════════════════════════
SESSION_DIR         = os.environ.get('ILC_SESSION_DIR', '6_28')
IVC_PHASE           = (0.00, 0.04)
IVR_PHASE           = (0.42, 0.44)
SINDY_SMOOTH_SIGMA  = 4
N_FOURIER           = 1
N_FOURIER_GEOM      = 1
P_NORM_REF          = 125.0
MIN_HISTORY_ITERS   = 1
FULL_TRUST_ITERS    = 10
GOOD_COND           = 1e2
MAX_COND            = 1e4

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL + HISTORY
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
    t = np.asarray(poly_lib.transform(xs))
    return np.stack([sindy_res[c]['sy'].inverse_transform((t @ sindy_res[c]['coef']).reshape(-1,1)).ravel()
                     for c in OUT_NAMES], axis=1)

FD_EPS = 1e-4
def jacobian_at(act_n, p_n):
    p_phys = p_n * x_den[3] + x_min[3]
    def fwd(a): return norm_y(sindy_pred(np.append(denorm_act(a), p_phys).reshape(1,-1))[0])
    f0 = fwd(act_n); J = np.zeros((3,3))
    for j in range(3):
        ap = act_n.copy(); ap[j] += FD_EPS
        J[:,j] = (fwd(ap) - f0) / FD_EPS
    return J

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

GRID = np.linspace(0, 1, 51)

def _fourier_design(phi, u_n, n_harm):
    cols = []
    for j in range(3):
        cols.append(u_n[:,j])
        for k in range(1, n_harm+1):
            cols.append(u_n[:,j]*np.sin(2*np.pi*k*phi))
            cols.append(u_n[:,j]*np.cos(2*np.pi*k*phi))
    cols.append(np.ones(len(phi)))
    return np.column_stack(cols)

records = []
for fpath in sorted(EXP_DATA.rglob('ILCReadyData.csv'), key=_sort_key):
    df = pd.read_csv(fpath)
    ce,ct,cn = _find_col(df,['epi_mm','epi']),_find_col(df,['trans_mm','trans']),_find_col(df,['endo_mm','endo'])
    cp = _find_col(df,['pressure','pressure_mmhg'])
    cw,ch,cv = _find_col(df,['twist','twist_deg']),_find_col(df,['height','height_mm']),_find_col(df,['volume','volume_mL'])
    cph = _find_col(df,['phase','time','time_s'])
    if any(c is None for c in [ce,ct,cn,cp]): continue
    ph = df[cph].values if cph else np.linspace(0,1,len(df))
    if ph.max()>1.5: ph=(ph-ph[0])/(ph[-1]-ph[0])
    rs = lambda v: interp1d(ph,v,'linear',bounds_error=False,fill_value='extrapolate')(GRID)
    records.append({'epi':rs(df[ce].values),'trans':rs(df[ct].values),'endo':rs(df[cn].values),
                    'pressure':rs(df[cp].values),
                    'twist':rs(df[cw].values) if cw else None,
                    'height':rs(df[ch].values) if ch else None,
                    'volume':rs(df[cv].values) if cv else None})
n_iters = len(records)
print(f"Session {SESSION_DIR}: {n_iters} iteration(s)")

# ── Fit Fourier pressure regression
fourier_pres_coeffs = None
if n_iters >= 1:
    U_all   = np.vstack([norm_act(np.column_stack([r['epi'],r['trans'],r['endo']])) for r in records])
    PHI_all = np.tile(GRID, n_iters)
    P_all   = np.concatenate([r['pressure'] for r in records]) / P_NORM_REF
    A_p = _fourier_design(PHI_all, U_all, N_FOURIER)
    fourier_pres_coeffs, _, _, _ = np.linalg.lstsq(A_p, P_all, rcond=None)
    print(f"Pressure regression fitted (N_FOURIER={N_FOURIER})")

# ── Fit Fourier geometry regression
geom_fourier_coeffs = None
if n_iters >= 1 and all(r['twist'] is not None for r in records):
    Y_all = norm_y(np.vstack([np.column_stack([r['twist'],r['height'],r['volume']]) for r in records]))
    A_g = _fourier_design(PHI_all, U_all, N_FOURIER_GEOM)
    geom_fourier_coeffs, _, _, _ = np.linalg.lstsq(A_g, Y_all, rcond=None)
    print(f"Geometry regression fitted (N_FOURIER_GEOM={N_FOURIER_GEOM})")

def J_pres_at(phi_s):
    """Phase-varying pressure Jacobian row (1×3) from Fourier regression."""
    J = np.zeros(3); idx = 0
    for j in range(3):
        J[j] = fourier_pres_coeffs[idx]; idx += 1
        for k in range(1, N_FOURIER+1):
            J[j] += fourier_pres_coeffs[idx]*np.sin(2*np.pi*k*phi_s); idx += 1
            J[j] += fourier_pres_coeffs[idx]*np.cos(2*np.pi*k*phi_s); idx += 1
    return J   # (3,)

def J_geom_reg_at(phi_s):
    """Phase-varying geometry Jacobian (3×3) from Fourier regression."""
    if geom_fourier_coeffs is None: return None
    J = np.zeros((3,3)); npa = 1+2*N_FOURIER_GEOM
    for ci in range(3):
        b = ci*npa
        for oi in range(3):
            J[oi,ci] = geom_fourier_coeffs[b,oi]
            for k in range(1,N_FOURIER_GEOM+1):
                J[oi,ci] += geom_fourier_coeffs[b+2*k-1,oi]*np.sin(2*np.pi*k*phi_s)
                J[oi,ci] += geom_fourier_coeffs[b+2*k,  oi]*np.cos(2*np.pi*k*phi_s)
    return J

# ── Compute per-phase SINDy geometry Jacobian (smoothed) using latest iteration
latest = records[-1]
act_n_latest = norm_act(np.column_stack([latest['epi'],latest['trans'],latest['endo']]))
p_n_latest   = (latest['pressure'] - x_min[3])/x_den[3]
J_sindy_raw  = np.array([jacobian_at(act_n_latest[i], p_n_latest[i]) for i in range(51)])
J_sindy      = np.zeros_like(J_sindy_raw)
for oi in range(3):
    for ci in range(3):
        J_sindy[:,oi,ci] = gaussian_filter1d(J_sindy_raw[:,oi,ci], sigma=SINDY_SMOOTH_SIGMA, mode='wrap')

# ── Per-phase conflict analysis
OUT_LABELS = ['Twist', 'Height', 'Volume']
J_pres_all  = np.array([J_pres_at(p) for p in GRID])   # (51, 3)  normalized units
J_geom_all  = J_sindy   # (51, 3, 3)  — SINDy smoothed as primary geometry estimate

# For each phase point: minimum-norm actuator change to achieve unit pressure correction
# Δu_pres = J_pres.T / (J_pres @ J_pres.T) = J_pres.T / ||J_pres||²
# Geometry disturbance = J_geom @ Δu_pres
conflict_ratio    = np.zeros(51)
geom_disturbance  = np.zeros((51, 3))   # (n, 3 outputs)
cosine_sim        = np.zeros((51, 3))   # (n, geometry rows)

for i in range(51):
    jp  = J_pres_all[i]      # (3,)
    jg  = J_geom_all[i]      # (3, 3) rows=outputs, cols=actuators
    norm_jp = np.linalg.norm(jp)
    if norm_jp < 1e-8:
        continue
    # Minimum-norm actuator change for unit pressure correction
    du_pres  = jp / (jp @ jp)   # (3,)  right pseudoinverse of the 1×3 row
    # How much each geometry output moves
    geom_disturbance[i] = jg @ du_pres   # (3,) normalized geometry change per unit pressure change
    conflict_ratio[i] = np.linalg.norm(geom_disturbance[i])
    # Cosine similarity between pressure direction and each geometry row
    for oi in range(3):
        jg_row = jg[oi]
        denom  = norm_jp * np.linalg.norm(jg_row) + 1e-8
        cosine_sim[i, oi] = (jp @ jg_row) / denom

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(14, 16))
fig.suptitle(f'Pressure–Geometry Conflict Analysis — session {SESSION_DIR}  ({n_iters} iteration(s))',
             fontsize=13, fontweight='bold')

ivc_lo, ivc_hi = IVC_PHASE
ivr_lo, ivr_hi = IVR_PHASE
act_colors = ['steelblue', 'darkorange', 'forestgreen']
act_names  = ['epi', 'trans', 'endo']

def _shade_windows(ax):
    ax.axvspan(ivc_lo, ivc_hi, alpha=0.15, color='deepskyblue', label='IVC' if ax==axes[0] else None)
    ax.axvspan(ivr_lo, ivr_hi, alpha=0.15, color='tomato',      label='IVR' if ax==axes[0] else None)

# ── Row 1: Pressure Jacobian J_pres(φ)
ax = axes[0]
for ci, (nm, clr) in enumerate(zip(act_names, act_colors)):
    ax.plot(GRID, J_pres_all[:, ci], color=clr, lw=2.5, label=f'∂P/∂{nm}')
ax.axhline(0, color='grey', lw=0.8, ls=':', alpha=0.5)
_shade_windows(ax)
ax.set_title('Pressure Jacobian J_pres(φ) — which actuators drive pressure and when', fontsize=11)
ax.set_ylabel('Normalised ∂P/∂u'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Row 2: Cosine similarity
ax = axes[1]
out_colors = ['tab:blue', 'tab:orange', 'tab:green']
for oi, (nm, clr) in enumerate(zip(OUT_LABELS, out_colors)):
    ax.plot(GRID, cosine_sim[:, oi], color=clr, lw=2.0, label=f'vs {nm}')
ax.axhline(0, color='grey', lw=1.0, ls='--', alpha=0.6)
ax.fill_between(GRID, -0.3, 0.3, alpha=0.08, color='green')
ax.text(0.5, 0.25, '< 0.3: low coupling — pressure ⊥ geometry here', fontsize=8,
        ha='center', color='green', transform=ax.transAxes)
_shade_windows(ax)
ax.set_title('Cosine similarity between J_pres and each geometry Jacobian row\n'
             '  Near ±1 = same direction (strong coupling)   Near 0 = orthogonal (low coupling)',
             fontsize=10)
ax.set_ylabel('Cosine similarity'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_ylim(-1.1, 1.1)

# ── Row 3: Conflict ratio
ax = axes[2]
ax.plot(GRID, conflict_ratio, color='crimson', lw=2.5,
        label='Conflict ratio: geometry disturbance per unit pressure correction')
ax.fill_between(GRID, 0, conflict_ratio, alpha=0.2, color='crimson')
_shade_windows(ax)
ax.set_title('Conflict ratio — geometry disturbance per unit pressure correction\n'
             '  Low = pressure can be corrected cheaply   '
             'High = pressure correction strongly disturbs geometry',
             fontsize=10)
ax.set_ylabel('||J_geom @ Δu_pres||'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Row 4: Geometry disturbance breakdown at IVC/IVR
ax = axes[3]
for oi, (nm, clr) in enumerate(zip(OUT_LABELS, out_colors)):
    ax.plot(GRID, geom_disturbance[:, oi], color=clr, lw=2.0, label=f'{nm} disturbance')
ax.axhline(0, color='grey', lw=0.8, ls=':', alpha=0.5)
_shade_windows(ax)
ax.set_title('Geometry disturbance breakdown — which outputs are disturbed most\n'
             '  by a unit pressure correction at each phase',
             fontsize=10)
ax.set_ylabel('Δoutput per unit ΔP\n(normalised)')
ax.set_xlabel('Cycle phase')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Annotate IVC and IVR on the bottom plot
for lo, hi, lbl, clr in [(ivc_lo, ivc_hi, 'IVC', 'deepskyblue'),
                          (ivr_lo, ivr_hi, 'IVR', 'tomato')]:
    mid = (lo + hi) / 2
    for a in axes:
        a.axvline(lo, color=clr, lw=1.0, ls='--', alpha=0.7)
        a.axvline(hi, color=clr, lw=1.0, ls='--', alpha=0.7)
    axes[3].text(mid, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 0,
                 lbl, ha='center', fontsize=9, color=clr, fontweight='bold')

plt.tight_layout()
out_path = os.path.join(BASE, 'pressureConflict.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")

# ── Console summary
print(f"\n{'═'*60}")
print("  Conflict ratio at IVC and IVR:")
print(f"{'─'*60}")
for name, lo, hi in [('IVC', ivc_lo, ivc_hi), ('IVR', ivr_lo, ivr_hi)]:
    mask = (GRID >= lo) & (GRID <= hi)
    if mask.any():
        cr = conflict_ratio[mask].mean()
        gd = geom_disturbance[mask].mean(axis=0)
        print(f"  {name} (φ {lo:.2f}–{hi:.2f}): conflict={cr:.3f}  "
              f"Δtwist={gd[0]:+.3f}  Δheight={gd[1]:+.3f}  Δvolume={gd[2]:+.3f}")
rest = (GRID < ivc_lo) | ((GRID > ivc_hi) & (GRID < ivr_lo)) | (GRID > ivr_hi)
cr_rest = conflict_ratio[rest].mean()
print(f"  Ejection/Filling (rest):       conflict={cr_rest:.3f}  (baseline)")
print(f"{'═'*60}")
plt.show()
