"""
plotILCChanges.py
──────────────────────────────────────────────────────────────────────────────
Shows the two recent ILC changes in plain visual terms.

CHANGE 1 — Adaptive correction weights
  When a particular output's sensitivity collapses (e.g. Volume barely
  responds to any actuator at peak compression), the old architecture kept
  pushing equally hard there anyway — wasting effort and potentially
  misdirecting the correction.  The new approach scales back that output's
  weight at those phases and redirects effort to outputs whose actuators
  still have real leverage.

CHANGE 2 — Phase-varying data-driven geometry Jacobian
  Old regression: Y = J·U + b  (one constant slope for the whole cycle)
  New regression: Y = J(φ)·U + b(φ)  (slope varies with cycle phase, fitted
    from session data using the same Fourier structure as the pressure model)
  This lets the geometry correction know "at compression, trans barely moves
  volume — but at filling, it moves it a lot" — and act accordingly.
"""

import sys, os, pathlib, re, pickle
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS — keep in sync with ilcCorrection.py
# ══════════════════════════════════════════════════════════════════════════════
SESSION_DIR         = os.environ.get('ILC_SESSION_DIR', '6_28')
GEOM_WEIGHTS        = np.array([1.0, 0.5, 1.0])
SENSITIVITY_FLOOR   = 0.25
SINDY_SMOOTH_SIGMA  = 4
N_FOURIER_GEOM      = 1
MIN_HISTORY_ITERS   = 1
FULL_TRUST_ITERS    = 10
GOOD_COND           = 1e2
IVC_PHASE           = (0.00, 0.04)
IVR_PHASE           = (0.42, 0.44)
PRESSURE_FADE_SIGMA = 1.5
MAX_COND           = 1e4

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

def sindy_predict_phys(x):
    x_s = sindy_sc.transform(np.asarray(x).reshape(-1, 4))
    theta = np.asarray(poly_lib.transform(x_s))
    return np.stack([sindy_res[c]['sy'].inverse_transform((theta @ sindy_res[c]['coef']).reshape(-1,1)).ravel()
                     for c in OUT_NAMES], axis=1)

FD_EPS = 1e-4
def jacobian_at(act_n, p_n):
    p_phys = p_n * x_den[3] + x_min[3]
    def fwd(a): return norm_y(sindy_predict_phys(np.append(denorm_act(a), p_phys).reshape(1,-1))[0])
    f0 = fwd(act_n)
    J = np.zeros((3,3))
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
records = []
for fpath in sorted(EXP_DATA.rglob('ILCReadyData.csv'), key=_sort_key):
    df = pd.read_csv(fpath)
    ce,ct,cn = _find_col(df,['epi_mm','epi']), _find_col(df,['trans_mm','trans']), _find_col(df,['endo_mm','endo'])
    cp = _find_col(df,['pressure','pressure_mmhg'])
    cw,ch,cv = _find_col(df,['twist','twist_deg']), _find_col(df,['height','height_mm']), _find_col(df,['volume','volume_mL'])
    cph = _find_col(df,['phase','time','time_s'])
    if any(c is None for c in [ce,ct,cn]): continue
    ph = df[cph].values if cph else np.linspace(0,1,len(df))
    if ph.max() > 1.5: ph = (ph-ph[0])/(ph[-1]-ph[0])
    rs = lambda v: interp1d(ph, v, 'linear', bounds_error=False, fill_value='extrapolate')(GRID)
    records.append({'epi':rs(df[ce].values), 'trans':rs(df[ct].values), 'endo':rs(df[cn].values),
                    'pressure':rs(df[cp].values) if cp else np.full(51,50.),
                    'twist':rs(df[cw].values) if cw else None,
                    'height':rs(df[ch].values) if ch else None,
                    'volume':rs(df[cv].values) if cv else None})

n_iters = len(records)
print(f"Session {SESSION_DIR}: {n_iters} iteration(s)")

# Compute per-point SINDy Jacobians (latest iteration, as proxy for current)
latest = records[-1]
act_n_latest = norm_act(np.column_stack([latest['epi'], latest['trans'], latest['endo']]))
p_n_latest   = (latest['pressure'] - x_min[3]) / x_den[3]
J_raw = np.array([jacobian_at(act_n_latest[i], p_n_latest[i]) for i in range(51)])
J_smooth = np.zeros_like(J_raw)
for oi in range(3):
    for ci in range(3):
        J_smooth[:,oi,ci] = gaussian_filter1d(J_raw[:,oi,ci], sigma=SINDY_SMOOTH_SIGMA, mode='wrap')

# Adaptive weights
row_norms    = np.linalg.norm(J_smooth, axis=2)          # (51, 3)
row_max      = row_norms.max(axis=0, keepdims=True) + 1e-8
adaptive_w   = (row_norms / row_max) * (1 - SENSITIVITY_FLOOR) + SENSITIVITY_FLOOR
adaptive_w   = adaptive_w * GEOM_WEIGHTS[None, :]

# Fourier geometry regression
geom_fourier_coeffs = None
_w_geom_reg = 0.0
cond_geom = np.inf
if all(r['twist'] is not None for r in records) and n_iters >= 1:
    U_all  = np.vstack([norm_act(np.column_stack([r['epi'],r['trans'],r['endo']])) for r in records])
    PHI_all= np.tile(GRID, n_iters)
    Y_all  = norm_y(np.vstack([np.column_stack([r['twist'],r['height'],r['volume']]) for r in records]))

    def fourier_design(phi, u_n, n_harm):
        cols = []
        for j in range(3):
            cols.append(u_n[:,j])
            for k in range(1, n_harm+1):
                cols.append(u_n[:,j]*np.sin(2*np.pi*k*phi))
                cols.append(u_n[:,j]*np.cos(2*np.pi*k*phi))
        cols.append(np.ones(len(phi)))
        return np.column_stack(cols)

    A = fourier_design(PHI_all, U_all, N_FOURIER_GEOM)
    cond_geom = np.linalg.cond(A)
    geom_fourier_coeffs, _, _, _ = np.linalg.lstsq(A, Y_all, rcond=None)
    w_n = float(np.clip((n_iters - MIN_HISTORY_ITERS)/(FULL_TRUST_ITERS - MIN_HISTORY_ITERS), 0, 1))
    if cond_geom <= GOOD_COND:   w_c = 1.0
    elif cond_geom >= MAX_COND:  w_c = 0.0
    else:
        lc,lg,lm = np.log10(cond_geom), np.log10(GOOD_COND), np.log10(MAX_COND)
        w_c = float(np.clip(1 - (lc-lg)/(lm-lg), 0, 1))
    _w_geom_reg = w_n * w_c
    print(f"Fourier geom regression: cond={cond_geom:.1e}  blend_weight={_w_geom_reg:.2f}")

    # Phase-varying J from regression (Volume row, for visualisation)
    def J_reg_at_phase(phi_s):
        J = np.zeros((3, 3))
        npa = 1 + 2 * N_FOURIER_GEOM
        for ci in range(3):
            b = ci * npa
            for oi in range(3):
                J[oi,ci] = geom_fourier_coeffs[b, oi]
                for k in range(1, N_FOURIER_GEOM+1):
                    J[oi,ci] += geom_fourier_coeffs[b+2*k-1,oi]*np.sin(2*np.pi*k*phi_s)
                    J[oi,ci] += geom_fourier_coeffs[b+2*k,  oi]*np.cos(2*np.pi*k*phi_s)
        return J
    J_reg_all = np.array([J_reg_at_phase(p) for p in GRID])

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 11))
fig.suptitle(f'Two changes to the ILC correction — session {SESSION_DIR}  ({n_iters} iteration(s))',
             fontsize=13, fontweight='bold', y=0.98)

cmap = cm.get_cmap('viridis', max(n_iters, 2))
OUT_LABELS = ['Twist (deg)', 'Height (mm)', 'Volume (mL)']
OUT_COLOURS = ['tab:blue', 'tab:orange', 'tab:green']

# ── CHANGE 1: Adaptive correction weights ─────────────────────────────────────
ax_title1 = fig.add_axes([0.02, 0.55, 0.96, 0.03])
ax_title1.axis('off')
ax_title1.text(0.5, 0.5,
    'CHANGE 1 — Adaptive correction weights  '
    '(base weight × local sensitivity — backs off when actuators have near-zero leverage)',
    ha='center', va='center', fontsize=11, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', fc='#e8f4e8', ec='green', lw=1.5))

for oi, (nm, clr) in enumerate(zip(OUT_LABELS, OUT_COLOURS)):
    ax = fig.add_axes([0.05 + oi*0.32, 0.34, 0.27, 0.19])
    ax.plot(GRID, adaptive_w[:, oi], color=clr, lw=2.5, label='New — adaptive')
    ax.axhline(GEOM_WEIGHTS[oi], color='grey', lw=2.0, ls='--', label=f'Old — constant ({GEOM_WEIGHTS[oi]})')
    ax.fill_between(GRID, GEOM_WEIGHTS[oi] * SENSITIVITY_FLOOR, adaptive_w[:, oi],
                    alpha=0.15, color=clr)
    ax.set_title(f'{nm}', fontsize=10)
    ax.set_xlabel('Cycle phase', fontsize=9)
    ax.set_ylabel('Correction weight', fontsize=9)
    ax.set_ylim(0, GEOM_WEIGHTS[oi] * 1.15 + 0.05)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    if oi == 1:
        ax.annotate('Weight drops here\n→ actuators have low\n   leverage for this output\n'
                    '→ effort redirected elsewhere',
                    xy=(GRID[np.argmin(adaptive_w[:,oi])], adaptive_w[:,oi].min()),
                    xytext=(0.55, GEOM_WEIGHTS[oi]*0.55),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=8, color='red')

# ── CHANGE 2: Phase-varying Fourier regression ────────────────────────────────
ax_title2 = fig.add_axes([0.02, 0.28, 0.96, 0.03])
ax_title2.axis('off')
status_txt = (f'CURRENTLY ACTIVE (blend weight = {_w_geom_reg:.2f})' if _w_geom_reg > 0
              else f'Not yet trusted (cond={cond_geom:.1e} > {MAX_COND:.0e}) — shown for reference')
status_clr = 'green' if _w_geom_reg > 0 else '#b05000'
ax_title2.text(0.5, 0.5,
    f'CHANGE 2 — Phase-varying data-driven geometry Jacobian  J(φ)·U + b(φ)   [{status_txt}]',
    ha='center', va='center', fontsize=11, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', fc='#e8eef8', ec='steelblue', lw=1.5, alpha=0.9))

# Show Volume row (most impactful) across all 3 actuators
ACT_NAMES = ['Epi', 'Trans', 'Endo']
vol_idx = 2  # Volume row
for ci, act_nm in enumerate(ACT_NAMES):
    ax = fig.add_axes([0.05 + ci*0.32, 0.07, 0.27, 0.19])

    # SINDy smoothed (current fallback)
    ax.plot(GRID, J_smooth[:, vol_idx, ci], color='steelblue', lw=2.5,
            label=f'SINDy smoothed (σ={SINDY_SMOOTH_SIGMA}) — current')

    # Fourier regression from data
    if geom_fourier_coeffs is not None:
        reg_line = J_reg_all[:, vol_idx, ci]
        reg_lbl = f'Fourier regression (data-driven, w={_w_geom_reg:.2f})'
        reg_clr = 'crimson' if _w_geom_reg > 0.05 else 'grey'
        ax.plot(GRID, reg_line, color=reg_clr, lw=2.5, ls='--', label=reg_lbl)

    ax.axhline(0, color='black', lw=0.8, ls=':', alpha=0.5)
    ax.set_title(f'∂Volume/∂{act_nm}', fontsize=10)
    ax.set_xlabel('Cycle phase\n(= actuator position proxy)', fontsize=9)
    ax.set_ylabel('Sensitivity (normalised)', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    if ci == 1:  # Trans is most dramatic
        ax.annotate('Near-zero here\n→ pushing trans does\n   nothing for volume\n'
                    '→ adaptive weight\n   reduces effort here',
                    xy=(GRID[np.argmin(J_smooth[:,vol_idx,ci])], J_smooth[:,vol_idx,ci].min()),
                    xytext=(0.6, J_smooth[:,vol_idx,ci].max()*0.6),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                    fontsize=8, color='darkred')

# ── CHANGE 3: Smooth pressure lambda envelope ──────────────────────────────────
ax_title3 = fig.add_axes([0.02, 0.01, 0.96, 0.03])
ax_title3.axis('off')
ax_title3.text(0.5, 0.5,
    'CHANGE 3 — Smooth phase-varying pressure lambda  '
    '(Gaussian fade at IVC/IVR boundaries — no trajectory kinks)',
    ha='center', va='center', fontsize=11, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.4', fc='#f8eef8', ec='purple', lw=1.5))

# Compute the smooth envelope
_lhard = np.array([1.0 if (IVC_PHASE[0] <= p <= IVC_PHASE[1] or IVR_PHASE[0] <= p <= IVR_PHASE[1])
                   else 0.0 for p in GRID])
_lsmooth = gaussian_filter1d(_lhard, sigma=PRESSURE_FADE_SIGMA, mode='wrap')

ax_lam = fig.add_axes([0.08, -0.16, 0.84, 0.14])
ax_lam.fill_between(GRID, 0, _lhard, alpha=0.2, color='grey', label='Hard gate (old — creates kinks)')
ax_lam.fill_between(GRID, 0, _lsmooth, alpha=0.35, color='purple')
ax_lam.plot(GRID, _lsmooth, color='purple', lw=2.5, label=f'Smooth envelope (σ={PRESSURE_FADE_SIGMA} samples)')
ax_lam.axvline(IVC_PHASE[0], color='steelblue', ls='--', lw=1.2, alpha=0.7, label=f'IVC {IVC_PHASE}')
ax_lam.axvline(IVC_PHASE[1], color='steelblue', ls='--', lw=1.2, alpha=0.7)
ax_lam.axvline(IVR_PHASE[0], color='darkorange', ls='--', lw=1.2, alpha=0.7, label=f'IVR {IVR_PHASE}')
ax_lam.axvline(IVR_PHASE[1], color='darkorange', ls='--', lw=1.2, alpha=0.7)
ax_lam.set_xlabel('Cycle phase', fontsize=10)
ax_lam.set_ylabel('λ multiplier (0–1)', fontsize=10)
ax_lam.set_title('Effective pressure weight = LAMBDA_P × this envelope  '
                 '(zero outside IVC/IVR → no conflict with geometry correction during ejection/filling)',
                 fontsize=9)
ax_lam.set_ylim(-0.05, 1.15)
ax_lam.legend(fontsize=9, loc='center right')
ax_lam.grid(True, alpha=0.3)

out_path = os.path.join(BASE, 'ilcChanges.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")
plt.show()
