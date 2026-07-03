"""
plotRegressionDiagnostics.py
──────────────────────────────────────────────────────────────────────────────
Visualises exactly what data feeds the empirical Jacobian / pressure
regression in ilcCorrection.py, and whether that data is diverse enough to
trust — read-only, no side effects (no history files written, no
sharedCSVs/ilc_corrected_actuators.csv touched).

Mirrors ilcCorrection.py's SESSION_DIR scoping and fitting logic exactly, so
what you see here is what the next ilcCorrection.py run will actually use.

Outputs: regressionDiagnostics.png with 4 rows:
  1. Actuator traces (epi/trans/endo vs phase) — one line per iteration file,
     colour-coded. Nearly-identical lines = thin actuator-space coverage.
  2. Pairwise actuator scatter (epi-trans, epi-endo, trans-endo) — points
     falling on a tight line/curve = collinearity (the regression can't
     reliably separate that pair's individual effects).
  3. Geometry fit quality: predicted vs actual (twist/height/volume), R²
     annotated, y=x reference line.
  4. Pressure: raw data vs phase, fit quality scatter, and the resulting
     phase-varying pressure Jacobian.

Also prints a summary table (files used, actuator ranges, condition number,
blend weight that would result) to the console.
"""

import os, sys, pathlib, re
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS — must match ilcCorrection.py for this diagnostic to be meaningful
# ══════════════════════════════════════════════════════════════════════════════
SESSION_DIR = os.environ.get('ILC_SESSION_DIR', '6_28')

MIN_HISTORY_ITERS = 1
FULL_TRUST_ITERS  = 10
GOOD_COND_NUMBER  = 1e2
MAX_COND_NUMBER   = 1e4
N_FOURIER         = 1
P_NORM_REF        = 125.0

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE     = os.path.dirname(os.path.abspath(__file__))
EXP_DATA = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Exp_data' / SESSION_DIR
assert EXP_DATA.exists(), f"Session folder not found:\n  {EXP_DATA}"
print(f"Session history scope: {EXP_DATA}")

def _find_col(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def _itr_sort_key(csv_path):
    itr_folder = csv_path.parent
    _dm = re.match(r'(\d+)_(\d+)\s*$', itr_folder.parent.name)
    date_key = (int(_dm.group(1)), int(_dm.group(2))) if _dm else (0, 0)
    m = re.search(r'(\d+)\s*$', itr_folder.name)
    num = int(m.group(1)) if m else -1
    is_p = bool(re.match(r'p_itr', itr_folder.name, re.IGNORECASE))
    return (date_key, 1 if is_p else 0, num)

def _iter_weight(n):
    if FULL_TRUST_ITERS <= MIN_HISTORY_ITERS:
        return 1.0 if n >= MIN_HISTORY_ITERS else 0.0
    return float(np.clip((n - MIN_HISTORY_ITERS) / (FULL_TRUST_ITERS - MIN_HISTORY_ITERS), 0.0, 1.0))

def _cond_weight(cond):
    if cond <= GOOD_COND_NUMBER:
        return 1.0
    if cond >= MAX_COND_NUMBER:
        return 0.0
    lc, lg, lm = np.log10(cond), np.log10(GOOD_COND_NUMBER), np.log10(MAX_COND_NUMBER)
    return float(np.clip(1.0 - (lc - lg) / (lm - lg), 0.0, 1.0))

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL ITERATION FILES IN THIS SESSION
# ══════════════════════════════════════════════════════════════════════════════
_GRID = np.linspace(0, 1, 51)
records = []   # one dict per iteration file

for fpath in sorted(EXP_DATA.rglob('ILCReadyData.csv'), key=_itr_sort_key):
    df = pd.read_csv(fpath)
    ce  = _find_col(df, ['epi_mm',  'epi'])
    ct  = _find_col(df, ['trans_mm','trans'])
    cn  = _find_col(df, ['endo_mm', 'endo'])
    cw  = _find_col(df, ['twist',   'twist_deg'])
    ch  = _find_col(df, ['height',  'height_mm'])
    cv  = _find_col(df, ['volume',  'volume_mL'])
    cp  = _find_col(df, ['pressure','pressure_mmhg'])
    cph = _find_col(df, ['phase','cycle_phase','time','time_s'])
    if any(c is None for c in [ce, ct, cn, cw, ch, cv]):
        print(f"  SKIPPED (missing columns): {fpath.parent.name}")
        continue
    ph = df[cph].values if cph else np.linspace(0, 1, len(df))
    if ph.max() > 1.5:
        ph = (ph - ph[0]) / (ph[-1] - ph[0])
    def rs(v):
        return interp1d(ph, v, kind='linear', bounds_error=False, fill_value='extrapolate')(_GRID)
    rec = {
        'label':  f"{fpath.parent.parent.name}/{fpath.parent.name}",
        'epi':    rs(df[ce].values), 'trans': rs(df[ct].values), 'endo': rs(df[cn].values),
        'twist':  rs(df[cw].values), 'height': rs(df[ch].values), 'volume': rs(df[cv].values),
        'pressure': rs(df[cp].values) if cp else None,
        'phase':  _GRID,
    }
    records.append(rec)
    print(f"  + {rec['label']}  epi=[{rec['epi'].min():.1f},{rec['epi'].max():.1f}]  "
          f"trans=[{rec['trans'].min():.1f},{rec['trans'].max():.1f}]  "
          f"endo=[{rec['endo'].min():.1f},{rec['endo'].max():.1f}] mm")

assert records, f"No usable ILCReadyData.csv found under:\n  {EXP_DATA}"
n_sources = len(records)
print(f"\n{n_sources} iteration file(s) loaded for this session.")

# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY FIT  (Y = J·U + b)  — same as ilcCorrection.py
# ══════════════════════════════════════════════════════════════════════════════
act_all  = np.vstack([np.column_stack([r['epi'], r['trans'], r['endo']]) for r in records])
geom_all = np.vstack([np.column_stack([r['twist'], r['height'], r['volume']]) for r in records])
A_act    = np.column_stack([act_all, np.ones(len(act_all))])
cond     = np.linalg.cond(A_act)
coeffs, _, _, _ = np.linalg.lstsq(A_act, geom_all, rcond=None)
J_geom, J_bias  = coeffs[:3, :].T, coeffs[3, :]
geom_pred_all   = A_act @ coeffs

w_iter = _iter_weight(n_sources)
w_cond = _cond_weight(cond)
w_blend = w_iter * w_cond

print(f"\nGeometry fit: cond={cond:.2e}  iter-confidence={w_iter:.2f}  "
      f"cond-confidence={w_cond:.2f}  blend weight={w_blend:.2f}")
geom_r2 = {}
for ci, nm in enumerate(['Twist', 'Height', 'Volume']):
    ss_res = np.sum((geom_all[:, ci] - geom_pred_all[:, ci])**2)
    ss_tot = np.sum((geom_all[:, ci] - geom_all[:, ci].mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    geom_r2[nm] = r2
    print(f"  {nm:<8} R²={r2:.3f}  epi={J_geom[ci,0]:+.3f}  trans={J_geom[ci,1]:+.3f}  endo={J_geom[ci,2]:+.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# PRESSURE FIT  (Fourier-augmented regression)  — same as ilcCorrection.py
# ══════════════════════════════════════════════════════════════════════════════
has_pressure = all(r['pressure'] is not None for r in records)
if has_pressure:
    x_act = act_all  # reuse, same stacking order
    u_min, u_max = x_act.min(axis=0), x_act.max(axis=0)
    u_den = np.where(u_max > u_min, u_max - u_min, 1.0)
    U_n   = (x_act - u_min) / u_den
    P_all = np.concatenate([r['pressure'] for r in records])
    PHI_all = np.concatenate([r['phase'] for r in records])
    P_n   = P_all / P_NORM_REF

    def fourier_design(phi, u_norm, n_harm):
        cols = []
        for j in range(3):
            cols.append(u_norm[:, j])
            for k in range(1, n_harm + 1):
                cols.append(u_norm[:, j] * np.sin(2*np.pi*k*phi))
                cols.append(u_norm[:, j] * np.cos(2*np.pi*k*phi))
        cols.append(np.ones(len(phi)))
        return np.column_stack(cols)

    def J_pres_at_phase(phi_s, coeffs_, n_harm):
        J = np.zeros(3); idx = 0
        for j in range(3):
            J[j] = coeffs_[idx]; idx += 1
            for k in range(1, n_harm + 1):
                J[j] += coeffs_[idx]*np.sin(2*np.pi*k*phi_s); idx += 1
                J[j] += coeffs_[idx]*np.cos(2*np.pi*k*phi_s); idx += 1
        return J

    A_reg = fourier_design(PHI_all, U_n, N_FOURIER)
    fourier_coeffs, _, _, _ = np.linalg.lstsq(A_reg, P_n, rcond=None)
    P_pred_n = A_reg @ fourier_coeffs
    p_r2 = 1 - np.sum((P_n-P_pred_n)**2)/np.sum((P_n-P_n.mean())**2)
    lambda_eff_would_be = w_iter
    print(f"\nPressure fit: R²={p_r2:.3f}  N_FOURIER={N_FOURIER}  "
          f"iter-confidence (λ_eff scale)={lambda_eff_would_be:.2f}")
else:
    print("\nPressure column missing in at least one file — pressure fit skipped.")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
n_rows = 4 if has_pressure else 3
fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.2*n_rows))
cmap = cm.get_cmap('viridis', max(n_sources, 2))

# Row 1 — actuator traces overlaid
for ci, (key, nm) in enumerate(zip(['epi','trans','endo'], ['Epi','Trans','Endo'])):
    ax = axes[0, ci]
    for i, r in enumerate(records):
        ax.plot(r['phase'], r[key], color=cmap(i/max(n_sources-1,1)), lw=1.5,
                label=r['label'], alpha=0.85)
    ax.set_title(f'{nm} (mm) — all iterations', fontsize=10)
    ax.set_xlabel('Phase'); ax.grid(True, alpha=0.3)
    if ci == 0:
        ax.legend(fontsize=6, loc='upper right')

# Row 2 — pairwise actuator scatter (collinearity check)
pairs = [('epi','trans','Epi vs Trans'), ('epi','endo','Epi vs Endo'), ('trans','endo','Trans vs Endo')]
for ci, (kx, ky, title) in enumerate(pairs):
    ax = axes[1, ci]
    for i, r in enumerate(records):
        ax.scatter(r[kx], r[ky], s=6, color=cmap(i/max(n_sources-1,1)), alpha=0.6)
    ax.set_title(f'{title}  (cond={cond:.1e})', fontsize=10)
    ax.set_xlabel(f'{kx} (mm)'); ax.set_ylabel(f'{ky} (mm)')
    ax.grid(True, alpha=0.3)

# Row 3 — geometry fit quality: predicted vs actual
for ci, nm in enumerate(['twist', 'height', 'volume']):
    ax = axes[2, ci]
    actual = geom_all[:, ci]
    pred   = geom_pred_all[:, ci]
    ax.scatter(actual, pred, s=6, color='steelblue', alpha=0.4)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, 'r--', lw=1.5, label='y = x')
    ax.set_title(f'{nm.capitalize()} fit  R²={geom_r2[nm.capitalize()]:.3f}', fontsize=10)
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Row 4 — pressure
if has_pressure:
    ax = axes[3, 0]
    for i, r in enumerate(records):
        ax.plot(r['phase'], r['pressure'], color=cmap(i/max(n_sources-1,1)), lw=1.2, alpha=0.8)
    ax.set_title('Pressure (mmHg) — all iterations', fontsize=10)
    ax.set_xlabel('Phase'); ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.scatter(P_all, P_pred_n*P_NORM_REF, s=6, color='steelblue', alpha=0.4)
    lims = [P_all.min(), P_all.max()]
    ax.plot(lims, lims, 'r--', lw=1.5, label='y = x')
    ax.set_title(f'Pressure fit  R²={p_r2:.3f}', fontsize=10)
    ax.set_xlabel('Actual (mmHg)'); ax.set_ylabel('Predicted (mmHg)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[3, 2]
    phi_plot = np.linspace(0, 1, 200)
    Jp = np.array([J_pres_at_phase(p, fourier_coeffs, N_FOURIER) for p in phi_plot])
    ax.plot(phi_plot, Jp[:,0], label='∂P/∂epi')
    ax.plot(phi_plot, Jp[:,1], label='∂P/∂trans')
    ax.plot(phi_plot, Jp[:,2], label='∂P/∂endo')
    ax.axhline(0, color='grey', lw=1, ls='--', alpha=0.5)
    ax.set_title('Phase-varying pressure Jacobian', fontsize=10)
    ax.set_xlabel('Phase'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.suptitle(f'Regression Diagnostics — session {SESSION_DIR}  '
             f'({n_sources} iterations, blend weight={w_blend:.2f})', fontsize=13, fontweight='bold')
plt.tight_layout()
out_path = os.path.join(BASE, 'regressionDiagnostics.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")
plt.show()
