"""
plotSINDyLocalGradient.py
──────────────────────────────────────────────────────────────────────────────
Three-way comparison of how each approach estimates actuator-to-output
sensitivity (∂output/∂actuator), to understand why they behave differently
and which best captures the true physics.

Row 1: Actuator trajectories (epi/trans/endo vs phase) — the x-axis is cycle
       phase, but what's actually driving the sensitivity variation is the
       ACTUATOR POSITION at that phase.  The Jacobian doesn't change with
       time; it changes because SINDy is nonlinear and the actuators sweep
       a wide range of positions through the cycle.

Rows 2-4: For each output (Twist/Height/Volume), the estimated sensitivity
          to each actuator (epi, trans, endo), shown three ways:

  ── Coloured lines  ──  SINDy evaluated per-phase-point (CURRENT arch.)
     Computed at each of the 51 actual actuator positions along the measured
     trajectory.  Gaussian-smoothed (σ=SINDY_SMOOTH_SIGMA) to remove
     finite-difference noise while preserving the genuine cycle variation.
     THIS is what ilcCorrection.py now uses.

  -- Black dashed  --  Constant mean-point SINDy (DISCARDED workaround)
     Evaluated once at the session's mean actuator position.  Introduced to
     eliminate oscillation, but throws away the real local structure.
     Specifically wrong for Volume where sensitivity collapses at compression.

  ·· Red dotted  ··  Empirical regression Jacobian (Y = J·U + b)
     Fit directly from the real measured data across all session iterations.
     A single GLOBAL slope — what the data says the AVERAGE sensitivity is
     across the whole operating range.  No phase variation (constant line),
     but reflects the actual physical device not the SINDy model.
     Shown even if currently rejected by the conditioning check (labelled).
"""

import sys, os, pathlib, re, pickle
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
SESSION_DIR       = os.environ.get('ILC_SESSION_DIR', '6_28')
SINDY_SMOOTH_SIGMA = 4   # must match ilcCorrection.py's SINDY_SMOOTH_SIGMA

# ══════════════════════════════════════════════════════════════════════════════
# PATHS + MODEL
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
    sindy_data = pickle.load(f)
sindy_sc = sindy_data['sc']
poly_lib  = sindy_data['poly_lib']
sindy_res = sindy_data['results']
OUT_NAMES = ['Twist_deg', 'Height_mm', 'Volume_mL']

def sindy_predict_phys(x):
    x_s = sindy_sc.transform(np.asarray(x).reshape(-1, 4))
    theta = np.asarray(poly_lib.transform(x_s))
    outs = []
    for col in OUT_NAMES:
        r = sindy_res[col]
        outs.append(r['sy'].inverse_transform((theta @ r['coef']).reshape(-1, 1)).ravel())
    return np.stack(outs, axis=1)

FD_EPS = 1e-4
def jacobian_at(act_n, p_n):
    p_phys = p_n * x_den[3] + x_min[3]
    def fwd(a_n):
        return norm_y(sindy_predict_phys(np.append(denorm_act(a_n), p_phys).reshape(1,-1))[0])
    f0 = fwd(act_n)
    J = np.zeros((3, 3))
    for j in range(3):
        ap = act_n.copy(); ap[j] += FD_EPS
        J[:, j] = (fwd(ap) - f0) / FD_EPS
    return J

# ══════════════════════════════════════════════════════════════════════════════
# LOAD SESSION ITERATION FILES
# ══════════════════════════════════════════════════════════════════════════════
def _find_col(df, cands):
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    return None

def _sort_key(p):
    itr = p.parent
    dm = re.match(r'(\d+)_(\d+)$', itr.parent.name)
    date_k = (int(dm.group(1)), int(dm.group(2))) if dm else (0, 0)
    m = re.search(r'(\d+)$', itr.name)
    num = int(m.group(1)) if m else -1
    return (date_k, bool(re.match(r'p_itr', itr.name, re.I)), num)

GRID = np.linspace(0, 1, 51)
records = []
for fpath in sorted(EXP_DATA.rglob('ILCReadyData.csv'), key=_sort_key):
    df = pd.read_csv(fpath)
    ce = _find_col(df, ['epi_mm','epi']); ct = _find_col(df, ['trans_mm','trans'])
    cn = _find_col(df, ['endo_mm','endo']); cp = _find_col(df, ['pressure','pressure_mmhg'])
    cph= _find_col(df, ['phase','time','time_s'])
    if any(c is None for c in [ce,ct,cn]):
        continue
    ph = df[cph].values if cph else np.linspace(0,1,len(df))
    if ph.max() > 1.5: ph = (ph-ph[0])/(ph[-1]-ph[0])
    rs = lambda v: interp1d(ph, v, kind='linear', bounds_error=False, fill_value='extrapolate')(GRID)
    records.append({
        'label': f"{fpath.parent.parent.name}/{fpath.parent.name}",
        'epi': rs(df[ce].values), 'trans': rs(df[ct].values), 'endo': rs(df[cn].values),
        'pressure': rs(df[cp].values) if cp else np.full(51, 50.0),
    })

assert records, f"No iteration files found under:\n  {EXP_DATA}"
n_iters = len(records)
print(f"Loaded {n_iters} iteration(s) from session {SESSION_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE PER-POINT JACOBIANS FOR EACH ITERATION
# ══════════════════════════════════════════════════════════════════════════════
# Also compute the CONSTANT mean-point Jacobian (what ilcCorrection.py uses)
iter_jacobians = []  # list of (51, 3, 3) arrays — one 3×3 matrix per phase point per iteration
for r in records:
    act_n_iter = norm_act(np.column_stack([r['epi'], r['trans'], r['endo']]))
    p_n_iter   = (r['pressure'] - x_min[3]) / x_den[3]
    Js = np.zeros((51, 3, 3))
    for i in range(51):
        Js[i] = jacobian_at(act_n_iter[i], p_n_iter[i])
    iter_jacobians.append(Js)
    print(f"  Computed per-point Jacobians for {r['label']}")

# ══════════════════════════════════════════════════════════════════════════════
# EMPIRICAL REGRESSION JACOBIAN  (Y = J·U + b fit from history data)
# In physical space (mm → deg/mm/mL), then scaled to normalised space to
# overlay on the same axes as the SINDy plots.
# ══════════════════════════════════════════════════════════════════════════════
geom_all = np.vstack([np.column_stack([r.get('twist', np.zeros(51)),
                                        r.get('height', np.zeros(51)),
                                        r.get('volume', np.zeros(51))]) for r in records])

# Reload geometry columns (they weren't stored in records above)
geom_records = []
for fpath in sorted(EXP_DATA.rglob('ILCReadyData.csv'), key=_sort_key):
    df = pd.read_csv(fpath)
    ce = _find_col(df, ['epi_mm','epi']); ct = _find_col(df, ['trans_mm','trans'])
    cn = _find_col(df, ['endo_mm','endo'])
    cw = _find_col(df, ['twist','twist_deg'])
    ch = _find_col(df, ['height','height_mm']); cv = _find_col(df, ['volume','volume_mL'])
    cph= _find_col(df, ['phase','time','time_s'])
    if any(c is None for c in [ce, ct, cn, cw, ch, cv]):
        continue
    ph = df[cph].values if cph else np.linspace(0, 1, len(df))
    if ph.max() > 1.5: ph = (ph-ph[0])/(ph[-1]-ph[0])
    rs = lambda v: interp1d(ph, v, kind='linear', bounds_error=False, fill_value='extrapolate')(GRID)
    geom_records.append({
        'act':  np.column_stack([rs(df[ce].values), rs(df[ct].values), rs(df[cn].values)]),
        'geom': np.column_stack([rs(df[cw].values), rs(df[ch].values), rs(df[cv].values)]),
    })

J_empirical_norm = None
cond_reg = np.inf
if len(geom_records) >= 2:
    act_all_phys  = np.vstack([g['act']  for g in geom_records])
    geom_all_phys = np.vstack([g['geom'] for g in geom_records])
    A_phys = np.column_stack([act_all_phys, np.ones(len(act_all_phys))])
    cond_reg = np.linalg.cond(A_phys)
    coeffs_phys, _, _, _ = np.linalg.lstsq(A_phys, geom_all_phys, rcond=None)
    J_phys = coeffs_phys[:3, :].T    # (3 outputs, 3 actuators) in physical/physical units

    # Scale to normalised/normalised space (same as SINDy jacobian_at output):
    # J_norm[oi, ci] = J_phys[oi, ci] * x_den[ci] / y_den[oi]
    J_empirical_norm = J_phys * x_den[:3][None, :] / y_den[:, None]

    print(f"\nEmpirical regression Jacobian (normalised, cond={cond_reg:.2e}):")
    for i, nm in enumerate(['Twist ', 'Height', 'Volume']):
        print(f"  d{nm}/du  epi={J_empirical_norm[i,0]:+.4f}  "
              f"trans={J_empirical_norm[i,1]:+.4f}  endo={J_empirical_norm[i,2]:+.4f}")
    if cond_reg > 1e4:
        print("  NOTE: conditioning too poor to trust (would be rejected by ilcCorrection.py)")
else:
    print("\nNot enough history files for empirical regression.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
cmap = cm.get_cmap('viridis', max(n_iters, 2))

fig, axes = plt.subplots(4, 3, figsize=(16, 14))
act_names = ['Epi (mm)', 'Trans (mm)', 'Endo (mm)']
out_names = ['Twist_deg', 'Height_mm', 'Volume_mL']
out_labels= ['∂Twist/∂u (normalised)', '∂Height/∂u (normalised)', '∂Volume/∂u (normalised)']
act_keys  = ['epi', 'trans', 'endo']

# Row 0: actuator trajectories
for ci, (key, nm) in enumerate(zip(act_keys, act_names)):
    ax = axes[0, ci]
    for ii, r in enumerate(records):
        ax.plot(GRID, r[key], color=cmap(ii/max(n_iters-1,1)), lw=1.8, label=r['label'], alpha=0.85)
    ax.set_title(f'{nm} — measured trajectory each iteration', fontsize=10)
    ax.set_xlabel('Cycle phase'); ax.grid(True, alpha=0.3)
    if ci == 0:
        ax.legend(fontsize=7, loc='upper right')
    ax.set_ylabel('mm')

# Rows 1-3: three-way comparison per gradient subplot
out_row_names = ['Twist ', 'Height', 'Volume']
for oi in range(3):
    for ci in range(3):
        ax = axes[oi+1, ci]

        # 1. Per-phase-point SINDy smoothed — CURRENT architecture
        #    Varies because sensitivity genuinely changes with actuator position.
        for ii, Js in enumerate(iter_jacobians):
            lbl = records[ii]['label'] if (oi == 0 and ci == 0) else None
            ax.plot(GRID, Js[:, oi, ci], color=cmap(ii/max(n_iters-1,1)),
                    lw=1.8, alpha=0.85, label=lbl)

        # 2. Empirical linear regression — Y = J·U + b fitted to session history.
        #    Always a constant (horizontal line) — a LINEAR model has one slope
        #    everywhere, it cannot represent sensitivity that varies with phase or
        #    actuator position.  Flagged if poorly conditioned.
        if J_empirical_norm is not None:
            cond_lbl = f'⚠ cond={cond_reg:.1e} — poor, rejected' if cond_reg > 1e4 else f'cond={cond_reg:.1e}'
            ax.axhline(J_empirical_norm[oi, ci], color='crimson', lw=2.0, ls=':',
                       label=(f'Linear regression Y=J·U+b ({cond_lbl})\n'
                               '  — constant slope, no phase awareness') if oi==0 and ci==2 else None)

        ax.set_title(f'∂{out_row_names[oi]}/∂{act_keys[ci]}', fontsize=10)
        ax.set_xlabel('Actuator position varies along this axis (cycle phase = proxy)')
        ax.set_ylabel('Normalised sensitivity')
        ax.grid(True, alpha=0.3)
        if oi == 0 and ci == 2:
            ax.legend(fontsize=7, loc='upper right')
        if oi == 0 and ci == 0:
            ax.legend(fontsize=6, loc='upper right', title='Iterations')

plt.suptitle(
    f'Jacobian comparison — session {SESSION_DIR}  ({n_iters} iterations)\n'
    f'Coloured = per-point SINDy smoothed (current arch, σ={SINDY_SMOOTH_SIGMA}) · '
    f'Red ·· = linear regression Y=J·U+b (constant — no phase awareness)',
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
out_path = os.path.join(BASE, 'sindyLocalGradient.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")
plt.show()
