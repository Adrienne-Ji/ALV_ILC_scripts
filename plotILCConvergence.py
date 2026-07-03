"""
plotILCConvergence.py
──────────────────────────────────────────────────────────────────────────────
Plots all experimental deformations across ILC iterations to visualise
convergence toward the desired trajectory.

Reads:  ILCFiles/Exp_data/**/ILCReadyData.csv  (all sessions/dates, filtered to CASE)
        the desired trajectory matching CASE (see _SIM_CASE_FILES below)

Outputs:
  ilcConvergence_tracking.png   — twist / height / volume / pressure
                                   all iterations overlaid, colour-coded by
                                   iteration number (light → dark)
  ilcConvergence_rmse.png       — RMSE vs iteration number per output
"""

import os, pathlib, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
# Which trajectory case to compare against — must match the CASE you actually
# ran. Overridable via env var ILC_CASE (set automatically if launched from
# run_ilc_pipeline.py-style tooling); otherwise edit the default below.
CASE = os.environ.get('ILC_CASE', 'healthy')

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE             = os.path.dirname(os.path.abspath(__file__))
PYTHONCODES      = os.path.join(BASE, '..')
ENGINEERED_TRAJS = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Engineered_trajs'

_SIM_CASE_FILES = {
    'healthy':   ENGINEERED_TRAJS / 'engineered_data_healthy.csv',
    'diastolic': ENGINEERED_TRAJS / 'engineered_data_diastolic_dysfunction.csv',
    'systolic':  ENGINEERED_TRAJS / 'engineered_data_systolic_dysfunction.csv',
}
assert CASE in _SIM_CASE_FILES, f"CASE must be one of {list(_SIM_CASE_FILES)}, got '{CASE}'"
ENG_CSV = _SIM_CASE_FILES[CASE]
print(f"Comparing against case: {CASE}  →  {ENG_CSV}")

EXP_DATA    = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Exp_data'
assert EXP_DATA.exists(), f"Exp_data folder not found:\n  {EXP_DATA}"

# Find all itrX / p_itrX subfolders that contain ILCReadyData.csv (possibly
# nested under a date folder, e.g. Exp_data/6_18/itr 0/ILCReadyData.csv)
# Sort: by date folder, then geometry iters (itr X) before pressure iters (p_itr X), by number
def _itr_sort_key(csv_path):
    itr_folder = csv_path.parent
    _dm = re.match(r'(\d+)_(\d+)\s*$', itr_folder.parent.name)
    date_key = (int(_dm.group(1)), int(_dm.group(2))) if _dm else (0, 0)
    m = re.search(r'(\d+)\s*$', itr_folder.name)
    num = int(m.group(1)) if m else -1
    is_p = bool(re.match(r'p_itr', itr_folder.name, re.IGNORECASE))
    return (date_key, 1 if is_p else 0, num)

def _matches_case(csv_path, case):
    """Keep folders whose path names this case explicitly. Folders with no
    case label at all (legacy data from before disease cases existed) are
    treated as 'healthy' only — never silently pooled into diastolic/systolic."""
    path_str   = str(csv_path).lower()
    case_l     = case.lower()
    other_cases = [c.lower() for c in _SIM_CASE_FILES if c.lower() != case_l]
    if case_l in path_str:
        return True
    if any(oc in path_str for oc in other_cases):
        return False
    return case_l == 'healthy'

_all_files = sorted(EXP_DATA.rglob('ILCReadyData.csv'), key=_itr_sort_key)
iter_files = [f for f in _all_files if _matches_case(f, CASE)]
assert iter_files, f"No iteration files found for case '{CASE}' under:\n  {EXP_DATA}"

iter_labels = [p.parent.name for p in iter_files]   # e.g. ['itr0','itr1',...]
print(f"Found {len(iter_files)} iteration(s) for case '{CASE}': {iter_labels}")

# ══════════════════════════════════════════════════════════════════════════════
# DESIRED TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════
eng        = pd.read_csv(ENG_CSV)
traj_time  = eng['time'].values
traj_phase = (traj_time - traj_time[0]) / (traj_time[-1] - traj_time[0])

from rig_config import HEIGHT_OFFSET, VOLUME_OFFSET   # single source of truth

desired = {
    'twist':    eng['twist'].values,
    'height':   eng['height'].values + HEIGHT_OFFSET,
    'volume':   eng['volume'].values + VOLUME_OFFSET,
    'pressure': eng['pressure'].values,
}

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL ITERATIONS
# ══════════════════════════════════════════════════════════════════════════════
OUTPUTS = ['twist', 'height', 'volume', 'pressure']
UNITS   = ['deg',   'mm',     'mL',     'mmHg']
LABELS  = ['Twist', 'Height', 'Volume', 'Pressure']

iters   = []   # list of dicts, one per iteration
rmse    = {k: [] for k in OUTPUTS}

for fpath, lbl in zip(iter_files, iter_labels):
    df = pd.read_csv(fpath)

    # Phase: use existing column or generate uniformly
    _phase_col = next((c for c in ['phase', 'time', 'time_s', 'abs_time_s']
                       if c in df.columns), None)
    if _phase_col:
        phase_raw = df[_phase_col].values
        if phase_raw.max() > 1.5:
            phase_raw = (phase_raw - phase_raw[0]) / (phase_raw[-1] - phase_raw[0])
    else:
        phase_raw = np.linspace(0.0, 1.0, len(df))

    # Resample onto desired trajectory phase grid
    def _resamp(vals):
        return interp1d(phase_raw, vals, kind='linear',
                        bounds_error=False, fill_value='extrapolate')(traj_phase)

    entry = {'label': lbl}
    for col in OUTPUTS:
        if col in df.columns:
            entry[col] = _resamp(df[col].values)
            rmse[col].append(float(np.sqrt(np.mean((desired[col] - entry[col])**2))))
        else:
            entry[col] = np.full_like(traj_phase, np.nan)
            rmse[col].append(np.nan)

    iters.append(entry)

n_iters = len(iters)
print(f"Loaded {n_iters} iteration(s)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Tracking across iterations
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)

# Split colour maps: geometry iters (blues), pressure iters (oranges)
n_geom = sum(1 for f in iter_files if not re.match(r'p_itr', f.parent.name, re.IGNORECASE))
n_pres = n_iters - n_geom
cmap   = cm.get_cmap('viridis', max(n_iters, 2))
sm     = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=1, vmax=n_iters))
sm.set_array([])

for ax, key, label, unit in zip(axes, OUTPUTS, LABELS, UNITS):
    # Desired
    ax.plot(traj_phase, desired[key], 'k-', lw=2.5, label='Desired', zorder=10)

    # Each iteration — colour gradient light (early) → dark (recent)
    for i, entry in enumerate(iters):
        color   = cmap(i / max(n_iters - 1, 1))
        is_last = (i == n_iters - 1)
        ax.plot(traj_phase, entry[key],
                color=color, lw=1.8 if is_last else 1.2,
                alpha=1.0  if is_last else 0.6,
                label=entry['label'] if is_last else None,
                zorder=5 if is_last else 3)

    ax.set_ylabel(f'{label} ({unit})', fontsize=10)
    ax.grid(True, alpha=0.3)

    # RMSE annotation for latest iteration
    if not np.isnan(rmse[key][-1]):
        ax.text(0.98, 0.05, f'Latest RMSE: {rmse[key][-1]:.2f} {unit}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=9, color=cmap(1.0),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

axes[0].legend(loc='upper right', fontsize=9)
axes[-1].set_xlabel('Cycle phase', fontsize=10)

# Colourbar
cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
cbar.set_label('Iteration number', fontsize=10)
cbar.set_ticks(np.arange(1, n_iters + 1))

plt.suptitle(f'ILC Convergence — Experimental Deformations  ({n_iters} iterations)',
             fontsize=13, fontweight='bold')

fig.savefig(os.path.join(BASE, 'ilcConvergence_tracking.png'),
            dpi=150, bbox_inches='tight')
print("  Saved → ilcConvergence_tracking.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — RMSE vs iteration
# ══════════════════════════════════════════════════════════════════════════════
iter_nums = np.arange(1, n_iters + 1)
colors    = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

# Left: all outputs on one axis (normalised to itr1 for relative comparison)
ax_norm = axes2[0]
for key, label, unit, clr in zip(OUTPUTS, LABELS, UNITS, colors):
    vals = np.array(rmse[key], dtype=float)
    if np.all(np.isnan(vals)):
        continue
    base = vals[0] if not np.isnan(vals[0]) else 1.0
    ax_norm.plot(iter_nums, vals / base, 'o-', color=clr, lw=1.8,
                 markersize=6, label=label)
ax_norm.axhline(1.0, color='grey', lw=1, ls='--', alpha=0.5)
ax_norm.set_xlabel('Iteration', fontsize=10)
ax_norm.set_ylabel('RMSE / RMSE$_{itr1}$  (normalised)', fontsize=10)
ax_norm.set_title('Relative convergence', fontsize=11)
ax_norm.legend(fontsize=9)
ax_norm.grid(True, alpha=0.3)
ax_norm.set_xticks(iter_nums)

# Right: physical RMSE per output (separate y-axes stacked)
ax_phys = axes2[1]
for key, label, unit, clr in zip(OUTPUTS, LABELS, UNITS, colors):
    vals = np.array(rmse[key], dtype=float)
    if np.all(np.isnan(vals)):
        continue
    ax_phys.plot(iter_nums, vals, 'o-', color=clr, lw=1.8,
                 markersize=6, label=f'{label} ({unit})')
ax_phys.set_xlabel('Iteration', fontsize=10)
ax_phys.set_ylabel('RMSE  (physical units)', fontsize=10)
ax_phys.set_title('Absolute RMSE per output', fontsize=11)
ax_phys.legend(fontsize=9)
ax_phys.grid(True, alpha=0.3)
ax_phys.set_xticks(iter_nums)

if n_geom > 0 and n_pres > 0:
    for ax_ in axes2:
        ax_.axvline(n_geom + 0.5, color='grey', lw=1.5, ls='--', alpha=0.6)
        ax_.text(n_geom + 0.6, ax_.get_ylim()[1] * 0.95, 'P-ILC →',
                 fontsize=8, color='grey', va='top')

plt.suptitle('ILC Convergence — RMSE vs Iteration', fontsize=13, fontweight='bold')
plt.tight_layout()

fig2.savefig(os.path.join(BASE, 'ilcConvergence_rmse.png'),
             dpi=150, bbox_inches='tight')
print("  Saved → ilcConvergence_rmse.png")

# ══════════════════════════════════════════════════════════════════════════════
# PRINT RMSE TABLE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  {'Iter':<8}", end='')
for label, unit in zip(LABELS, UNITS):
    print(f"  {label+' ('+unit+')':<18}", end='')
print()
print("  " + "-" * (8 + 20 * len(OUTPUTS)))
for i, lbl in enumerate(iter_labels):
    print(f"  {lbl:<8}", end='')
    for key in OUTPUTS:
        v = rmse[key][i]
        print(f"  {v:<18.3f}" if not np.isnan(v) else f"  {'—':<18}", end='')
    print()

plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FINAL ITERATION RMS vs DESIRED
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*48}")
print(f"  Final iteration ({iter_labels[-1]}) RMS vs desired:")
print(f"{'─'*48}")
for key, label, unit in zip(OUTPUTS, LABELS, UNITS):
    v = rmse[key][-1]
    print(f"  {label:<12}: {v:.3f} {unit}")
print(f"{'═'*48}")
