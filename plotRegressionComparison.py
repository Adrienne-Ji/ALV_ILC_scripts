"""
plotRegressionComparison.py
──────────────────────────────────────────────────────────────────────────────
Three-way Jacobian comparison across the cardiac cycle:

  RED dashed    = Linear regression  Y = J·U + b  (constant slope)
  BLUE solid    = Fourier regression Y = J(φ)·U + b(φ)  (data-driven, varies)
  GREY band     = SINDy per-point (model-based, varies — range across iterations)

Bottom panel: How does Fourier regression improve with more iterations?
  Left:  R² vs number of iterations (both Twist/Height/Volume)
  Right: Condition number vs number of iterations
  → Answers: do we still need the blend weighting with SINDy?
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
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
SESSION_DIR    = os.environ.get('ILC_SESSION_DIR', '6_18')
N_FOURIER_GEOM = 1
SINDY_SIGMA    = 4

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL + HISTORY
# ══════════════════════════════════════════════════════════════════════════════
BASE     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = pathlib.Path(BASE) / 'saved_models'
EXP_DATA = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Exp_data' / SESSION_DIR

norm = np.load(SAVE_DIR / 'norm_constants.npz')
x_min, x_den = norm['x_min'], norm['x_den']
y_min, y_den = norm['y_min'], norm['y_den']
norm_act = lambda a: (np.asarray(a) - x_min[:3]) / x_den[:3]
denorm_act = lambda a: np.asarray(a) * x_den[:3] + x_min[:3]
norm_y = lambda y: (np.asarray(y) - y_min) / y_den

with open(SAVE_DIR / 'sindy_data.pkl', 'rb') as f:
    sd = pickle.load(f)
OUT_NAMES_SINDY = ['Twist_deg', 'Height_mm', 'Volume_mL']
def sindy_pred(x):
    xs = sd['sc'].transform(np.asarray(x).reshape(-1, 4))
    t  = np.asarray(sd['poly_lib'].transform(xs))
    return np.stack([sd['results'][c]['sy'].inverse_transform(
        (t @ sd['results'][c]['coef']).reshape(-1,1)).ravel()
        for c in OUT_NAMES_SINDY], axis=1)

FD_EPS = 1e-4
def sindy_jacobian(act_n, p_n):
    pp = p_n * x_den[3] + x_min[3]
    fwd = lambda a: norm_y(sindy_pred(np.append(denorm_act(a), pp).reshape(1,-1))[0])
    f0 = fwd(act_n); J = np.zeros((3, 3))
    for j in range(3):
        ap = act_n.copy(); ap[j] += FD_EPS
        J[:, j] = (fwd(ap) - f0) / FD_EPS
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
    ce = _find_col(df,['epi_mm','epi']); ct = _find_col(df,['trans_mm','trans'])
    cn = _find_col(df,['endo_mm','endo'])
    cw = _find_col(df,['twist','twist_deg']); ch = _find_col(df,['height','height_mm'])
    cv = _find_col(df,['volume','volume_mL'])
    cph = _find_col(df,['phase','time','time_s'])
    if any(c is None for c in [ce,ct,cn,cw,ch,cv]): continue
    ph = df[cph].values if cph else np.linspace(0,1,len(df))
    if ph.max()>1.5: ph=(ph-ph[0])/(ph[-1]-ph[0])
    rs = lambda v: interp1d(ph,v,'linear',bounds_error=False,fill_value='extrapolate')(GRID)
    records.append({
        'act':  np.column_stack([rs(df[ce].values), rs(df[ct].values), rs(df[cn].values)]),
        'geom': np.column_stack([rs(df[cw].values), rs(df[ch].values), rs(df[cv].values)]),
        'pres': rs(df[_find_col(df,['pressure','pressure_mmhg'])].values)
                if _find_col(df,['pressure','pressure_mmhg']) else np.zeros(51),
        'label': f"{fpath.parent.parent.name}/{fpath.parent.name}",
    })

n = len(records)
print(f"Session {SESSION_DIR}: {n} iteration(s)")

OUT_NAMES = ['Twist', 'Height', 'Volume']
ACT_NAMES = ['Epi',   'Trans',  'Endo']

# ══════════════════════════════════════════════════════════════════════════════
# REGRESSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fourier_design(phi, u_n, n_harm):
    cols = []
    for j in range(3):
        cols.append(u_n[:,j])
        for k in range(1, n_harm+1):
            cols.append(u_n[:,j]*np.sin(2*np.pi*k*phi))
            cols.append(u_n[:,j]*np.cos(2*np.pi*k*phi))
    cols.append(np.ones(len(phi)))
    return np.column_stack(cols)

def fit_regressions(recs):
    U_n   = norm_act(np.vstack([r['act']  for r in recs]))
    Y_n   = norm_y(np.vstack([r['geom'] for r in recs]))
    PHI   = np.tile(GRID, len(recs))
    # Linear
    A_lin = np.column_stack([U_n, np.ones(len(U_n))])
    c_lin, _, _, _ = np.linalg.lstsq(A_lin, Y_n, rcond=None)
    J_lin = c_lin[:3,:].T   # (3 outputs, 3 acts)
    pred_lin = A_lin @ c_lin
    cond_lin = np.linalg.cond(A_lin)
    # Fourier
    A_f = fourier_design(PHI, U_n, N_FOURIER_GEOM)
    c_f, _, _, _ = np.linalg.lstsq(A_f, Y_n, rcond=None)
    pred_f = A_f @ c_f
    cond_f = np.linalg.cond(A_f)
    # R²
    def r2(pred, actual):
        ss_res = np.sum((actual - pred)**2, axis=0)
        ss_tot = np.sum((actual - actual.mean(axis=0))**2, axis=0)
        return 1 - ss_res / np.where(ss_tot>0, ss_tot, 1)
    return {'J_lin': J_lin, 'c_f': c_f, 'pred_lin': pred_lin, 'pred_f': pred_f,
            'Y_n': Y_n, 'cond_lin': cond_lin, 'cond_f': cond_f,
            'r2_lin': r2(pred_lin, Y_n), 'r2_f': r2(pred_f, Y_n)}

def J_fourier_at(phi_s, c_f):
    J = np.zeros((3, 3)); npa = 1+2*N_FOURIER_GEOM
    for ci in range(3):
        b = ci*npa
        for oi in range(3):
            J[oi,ci] = c_f[b,oi]
            for k in range(1,N_FOURIER_GEOM+1):
                J[oi,ci] += c_f[b+2*k-1,oi]*np.sin(2*np.pi*k*phi_s)
                J[oi,ci] += c_f[b+2*k,  oi]*np.cos(2*np.pi*k*phi_s)
    return J

# Full fit on all iterations
fit = fit_regressions(records)
phi_plot = np.linspace(0, 1, 200)
J_fourier_curve = np.array([J_fourier_at(p, fit['c_f']) for p in phi_plot])  # (200,3,3)

# ── SINDy per-point Jacobians (smoothed, ALL iterations → band)
print("Computing SINDy Jacobians…")
sindy_all = []
for r in records:
    act_n_iter = norm_act(r['act'])
    p_n_iter   = (r['pres'] - x_min[3]) / x_den[3]
    Js = np.zeros((51,3,3))
    for i in range(51):
        Js[i] = sindy_jacobian(act_n_iter[i], p_n_iter[i])
    # Gaussian smooth
    for oi in range(3):
        for ci in range(3):
            Js[:,oi,ci] = gaussian_filter1d(Js[:,oi,ci], sigma=SINDY_SIGMA, mode='wrap')
    sindy_all.append(Js)
sindy_stack = np.stack(sindy_all, axis=0)   # (n_iters, 51, 3, 3)
sindy_mean  = sindy_stack.mean(axis=0)       # (51, 3, 3)
sindy_lo    = sindy_stack.min(axis=0)
sindy_hi    = sindy_stack.max(axis=0)
print("  Done.")

# ── Iteration-progression: R² and cond vs n_iters
itr_counts = list(range(1, n+1))
r2_lin_prog  = {nm: [] for nm in OUT_NAMES}
r2_f_prog    = {nm: [] for nm in OUT_NAMES}
cond_lin_prog, cond_f_prog = [], []

for k in itr_counts:
    res = fit_regressions(records[:k])
    cond_lin_prog.append(res['cond_lin'])
    cond_f_prog.append(res['cond_f'])
    for oi, nm in enumerate(OUT_NAMES):
        r2_lin_prog[nm].append(res['r2_lin'][oi])
        r2_f_prog[nm].append(res['r2_f'][oi])

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 18))
r2_improvement = {nm: fit['r2_f'][oi]-fit['r2_lin'][oi] for oi,nm in enumerate(OUT_NAMES)}
fig.suptitle(
    f'Geometry Jacobian: Linear vs Fourier vs SINDy — session {SESSION_DIR}  ({n} iters)\n'
    f'R² improvement (Fourier over Linear):  '
    + '  '.join(f'{nm} {r2_improvement[nm]:+.3f}' for nm in OUT_NAMES),
    fontsize=12, fontweight='bold', y=0.99
)

gs = fig.add_gridspec(4, 3, hspace=0.45, wspace=0.35,
                       top=0.94, bottom=0.06, left=0.07, right=0.97,
                       height_ratios=[1,1,1,1.2])

# ── Rows 0-2: Jacobian comparison
for oi, out_nm in enumerate(OUT_NAMES):
    for ci, act_nm in enumerate(ACT_NAMES):
        ax = fig.add_subplot(gs[oi, ci])

        # SINDy band (range across all iterations)
        ax.fill_between(GRID, sindy_lo[:,oi,ci], sindy_hi[:,oi,ci],
                        alpha=0.18, color='grey', label='SINDy range (all iters)')
        ax.plot(GRID, sindy_mean[:,oi,ci], color='dimgrey', lw=1.5, ls='-.',
                label='SINDy mean (smoothed)')

        # Fourier regression
        ax.plot(phi_plot, J_fourier_curve[:,oi,ci], color='steelblue', lw=2.5,
                label=f'Fourier data  R²={fit["r2_f"][oi]:.3f}')

        # Linear regression (constant)
        ax.axhline(fit['J_lin'][oi,ci], color='crimson', lw=2.0, ls='--',
                   label=f'Linear const  R²={fit["r2_lin"][oi]:.3f}')

        ax.axhline(0, color='grey', lw=0.6, ls=':', alpha=0.4)
        ax.set_title(f'∂{out_nm}/∂{act_nm}', fontsize=10)
        ax.set_xlabel('Phase (actuator position proxy)', fontsize=8)
        ax.set_ylabel('Normalised sensitivity', fontsize=8)
        ax.grid(True, alpha=0.25)
        if oi == 0 and ci == 2:
            ax.legend(fontsize=7.5, loc='lower right')

# ── Row 3: Iteration-progression (R² and cond)
ax_r2   = fig.add_subplot(gs[3, :2])
ax_cond = fig.add_subplot(gs[3, 2])

out_colors = ['tab:blue','tab:orange','tab:green']
for oi, (nm, clr) in enumerate(zip(OUT_NAMES, out_colors)):
    ax_r2.plot(itr_counts, r2_lin_prog[nm], color=clr, lw=1.5, ls='--', alpha=0.6,
               label=f'Linear {nm}')
    ax_r2.plot(itr_counts, r2_f_prog[nm],  color=clr, lw=2.0, ls='-',
               label=f'Fourier {nm}')

ax_r2.set_xlabel('Number of session iterations used in regression fit', fontsize=10)
ax_r2.set_ylabel('R²', fontsize=10)
ax_r2.set_title('How Fourier regression quality improves with more iterations\n'
                '(dashed = linear, solid = Fourier — same colour per output)',
                fontsize=10)
ax_r2.legend(fontsize=8, ncol=2); ax_r2.grid(True, alpha=0.3)
ax_r2.set_xticks(itr_counts); ax_r2.set_xlim(0.5, n+0.5)

ax_cond.semilogy(itr_counts, cond_lin_prog, 'crimson', lw=2.0, ls='--', label='Linear')
ax_cond.semilogy(itr_counts, cond_f_prog,   'steelblue', lw=2.0,       label='Fourier')
ax_cond.axhline(1e4, color='red', lw=1.2, ls=':', alpha=0.7, label='Trust threshold (1e4)')
ax_cond.axhline(1e2, color='green', lw=1.2, ls=':', alpha=0.7, label='Good threshold (1e2)')
ax_cond.set_xlabel('Number of session iterations', fontsize=10)
ax_cond.set_ylabel('Condition number (log scale)', fontsize=10)
ax_cond.set_title('Conditioning vs iterations\n(below 1e2 = trusted fully)',
                  fontsize=10)
ax_cond.legend(fontsize=8); ax_cond.grid(True, alpha=0.3)
ax_cond.set_xticks(itr_counts); ax_cond.set_xlim(0.5, n+0.5)

out_path = os.path.join(BASE, 'regressionComparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")

# Console summary
print(f"\n{'═'*60}")
print("  Conditioning at each iteration count:")
for k, cl, cf in zip(itr_counts, cond_lin_prog, cond_f_prog):
    trusted = 'TRUSTED' if cf < 1e4 else 'rejected'
    print(f"  {k:2d} iter: linear={cl:.0f}  Fourier={cf:.0f}  [{trusted}]")
print('='*60)
plt.show()
