"""
run_ilc_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Orchestrates one full ILC iteration. All outputs land under
ILCFiles/ILC_traj/<SESSION_DIR>/<CASE>/, mirroring the input history layout
at ILCFiles/Exp_data/<SESSION_DIR>/<CASE>/.

  1. Runs ilcCorrection.py  →  sharedCSVs/ilc_corrected_actuators.csv
       First iteration of a session : geometry only (no history yet, λ=0)
       Every iteration after        : geometry + pressure (fitted on session history)
  2. Saves 1-cycle output  →  ILC_traj/<SESSION_DIR>/<CASE>/ILC_<tag>.csv
  3. Plots corrected vs previous actuator signal
  4. Runs PVTwrite.py  →  ILC_traj/<SESSION_DIR>/<CASE>/PVT_ILC_<tag>_20s_15cycles.csv
"""

import os, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS — this script is the single source of truth for SESSION_DIR/CASE.
# It passes them down to ilcCorrection.py and PVTwrite.py via env vars, so all
# three stay consistent. (ilcCorrection.py's own SESSION_DIR/SIM_CASE settings
# are only used as fallback when it's run standalone, without this wrapper.)
# ══════════════════════════════════════════════════════════════════════════════

SESSION_DIR = '6_28'       # date folder — must match ilcCorrection.py's SESSION_DIR
CASE        = 'healthy'    # 'healthy' | 'diastolic' | 'systolic' — must match SIM_CASE

# Tag appended to output filename — increment each iteration
OUTPUT_TAG = 'P_iter2'    # e.g. 'iter1', 'iter2', '2026-06-02'

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE        = os.path.dirname(os.path.abspath(__file__))
PYTHONCODES = os.path.normpath(os.path.join(BASE, '..'))
SHARED      = os.path.normpath(os.path.join(BASE, '..', '..', 'sharedCSVs'))
ILC_TRAJ    = os.path.normpath(os.path.join(BASE, '..', '..', 'ILCFiles', 'ILC_traj', SESSION_DIR, CASE))
os.makedirs(ILC_TRAJ, exist_ok=True)
print(f"Output folder for this experiment: {ILC_TRAJ}")

ILC_CSV     = os.path.join(SHARED, 'ilc_corrected_actuators.csv')   # transient handoff
PREV_CSV    = os.path.join(SHARED, 'ILCReadyData.csv')

ILC_SCRIPT  = os.path.join(BASE, 'ilcCorrection.py')
PVT_SCRIPT  = os.path.join(PYTHONCODES, 'PVTwrite.py')
OUT_CSV     = os.path.join(ILC_TRAJ, f'ILC_{OUTPUT_TAG}.csv')
OUT_FIG     = os.path.join(ILC_TRAJ, f'ILC_{OUTPUT_TAG}_comparison.png')

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Run ilcCorrection.py
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 64)
print(f"  STEP 1 — Running ilcCorrection.py …")
print("=" * 64)

result = subprocess.run(
    [sys.executable, ILC_SCRIPT], cwd=BASE,
    env={**os.environ, 'ILC_SESSION_DIR': SESSION_DIR, 'ILC_CASE': CASE,
         'ILC_OUTPUT_TAG': OUTPUT_TAG},
)
if result.returncode != 0:
    sys.exit(f"\n  ilcCorrection.py failed (exit code {result.returncode}). Aborting.")

assert os.path.exists(ILC_CSV), \
    f"Expected output not found:\n  {ILC_CSV}\nCheck ilcCorrection.py for errors."
print(f"\n  ILC correction ready → {ILC_CSV}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Save 1-cycle output with normalised phase
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  STEP 2 — Writing single-cycle output …")
print("=" * 64)

ilc_df = pd.read_csv(ILC_CSV)

# Ensure phase is normalised [0, 1]
phase = ilc_df['phase'].values
phase = (phase - phase[0]) / (phase[-1] - phase[0])

df_out = pd.DataFrame({
    'phase': phase,
    'epi':   ilc_df['epi'].values,
    'trans': ilc_df['trans'].values,
    'endo':  ilc_df['endo'].values,
})
df_out.to_csv(OUT_CSV, index=False, float_format='%.4f')
print(f"  Single-cycle file written → {OUT_CSV}  ({len(df_out)} pts)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Plot: corrected vs previous actuator signal
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  STEP 3 — Generating comparison plot …")
print("=" * 64)

# Load previous signal from ILCReadyData.csv
assert os.path.exists(PREV_CSV), \
    f"Previous signal not found:\n  {PREV_CSV}"

prev_df = pd.read_csv(PREV_CSV)

# Auto-detect actuator columns in previous file (epi_mm or epi)
def _col(df, candidates):
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

prev_epi   = prev_df[_col(prev_df, ['epi_mm',   'epi'])].values
prev_trans = prev_df[_col(prev_df, ['trans_mm', 'trans'])].values
prev_endo  = prev_df[_col(prev_df, ['endo_mm',  'endo'])].values
prev_phase = np.linspace(0, 1, len(prev_epi))

new_phase = df_out['phase'].values
new_epi   = df_out['epi'].values
new_trans = df_out['trans'].values
new_endo  = df_out['endo'].values

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
labels    = ['Epi (mm)', 'Trans (mm)', 'Endo (mm)']
prev_sigs = [prev_epi,   prev_trans,   prev_endo]
new_sigs  = [new_epi,    new_trans,    new_endo]

for ax, label, prev, new in zip(axes, labels, prev_sigs, new_sigs):
    ax.plot(prev_phase, prev, color='steelblue',  lw=2.0, ls='--', label='Previous (ILCReadyData)')
    ax.plot(new_phase,  new,  color='darkorange', lw=2.0,          label=f'Corrected ({OUTPUT_TAG})')
    delta_max = np.abs(new - np.interp(new_phase, prev_phase, prev)).max()
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(f'Max |Δ| = {delta_max:.2f} mm', fontsize=9, color='grey')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Cycle phase (normalised)', fontsize=10)
plt.suptitle(f'ILC Actuator Correction — {OUTPUT_TAG}\n'
             f'Previous signal vs corrected signal', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
print(f"  Comparison plot saved → {OUT_FIG}")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Run PVTwrite.py (tiling, ramps, velocity → Zaber-ready PVT CSV)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("  STEP 4 — Running PVTwrite.py …")
print("=" * 64)

assert os.path.exists(PVT_SCRIPT), \
    f"PVTwrite.py not found at:\n  {PVT_SCRIPT}"

result = subprocess.run(
    [sys.executable, PVT_SCRIPT],
    cwd=PYTHONCODES,
    env={**os.environ, 'ILC_ITER_TAG': OUTPUT_TAG,
         'ILC_SESSION_DIR': SESSION_DIR, 'ILC_CASE': CASE},
)
if result.returncode != 0:
    sys.exit(f"\n  PVTwrite.py failed (exit code {result.returncode}). Aborting.")

PVT_CSV = os.path.join(ILC_TRAJ, f'PVT_ILC_{OUTPUT_TAG}_20s_15cycles.csv')
print(f"\n  PVT file written → {PVT_CSV}")

print("\n  Pipeline complete.")
print(f"  ILC CSV : {OUT_CSV}")
print(f"  Plot    : {OUT_FIG}")
print(f"  PVT     : {PVT_CSV}")
