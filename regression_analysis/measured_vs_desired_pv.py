"""
Unfiltered measured vs desired (phase) + PV loop.
Uses ILCReadyData.csv directly — already phase-aligned, no interpolation.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pathlib, re, sys as _sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_sys.path.insert(0, r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\PythonCodes\MPC_depthCam')
from rig_config import HEIGHT_OFFSET, VOLUME_OFFSET

SESSION  = '7_17'
EXP_ROOT = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\ILCFiles\Exp_data')
ENG_CSV  = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\ILCFiles\Engineered_trajs\engineered_data_healthy.csv')
OUT_DIR  = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\PythonCodes\MPC_depthCam\regression_analysis')

def sort_key(p):
    m = re.search(r'(\d+)', p.name)
    return int(m.group(1)) if m else -1

# ── Desired trajectory ────────────────────────────────────────────────────────
eng  = pd.read_csv(ENG_CSV)
t    = eng['time'].values
phi  = (t - t[0]) / (t[-1] - t[0])
N_DES = 100
phi_des = np.linspace(0, 1, N_DES)
des = {
    'twist':    np.interp(phi_des, phi, eng['twist'].values),
    'height':   np.interp(phi_des, phi, eng['height'].values) + HEIGHT_OFFSET,
    'volume':   np.interp(phi_des, phi, eng['volume'].values) + VOLUME_OFFSET,
    'pressure': np.interp(phi_des, phi, eng['pressure'].values),
}

# ── Iterations ────────────────────────────────────────────────────────────────
sess_dir = EXP_ROOT / SESSION
itr_dirs = sorted(
    [d for d in sess_dir.iterdir() if d.is_dir() and re.search(r'itr\d+', d.name, re.I)],
    key=sort_key
)

itrs = []
for d in itr_dirs:
    csv = d / 'ILCReadyData.csv'
    if not csv.exists():
        continue
    df = pd.read_csv(csv)
    itrs.append({'name': d.name, 'df': df})
    print(f"  {d.name}: {len(df)} pts  cols={df.columns.tolist()}")

N       = len(itrs)
COLOURS = cm.plasma(np.linspace(0.1, 0.85, N))
phi_m   = np.linspace(0, 1, len(itrs[0]['df']))   # match actual point count

SIGNALS = [
    ('twist',    'Twist (°)'),
    ('height',   'Height (mm)'),
    ('volume',   'Volume (mL)'),
    ('pressure', 'Pressure (mmHg)'),
]

# ── Figure 1: measured vs desired phase ──────────────────────────────────────
fig1, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

for ax, (col, ylabel) in zip(axes, SIGNALS):
    ax.plot(phi_des, des[col], 'k--', lw=2.0, label='Desired', zorder=5)
    for ci, d in enumerate(itrs):
        meas = d['df'][col].values if col in d['df'].columns else None
        if meas is None:
            # pressure may be named differently
            alt = [c for c in d['df'].columns if 'pres' in c.lower()]
            meas = d['df'][alt[0]].values if alt else None
        if meas is not None:
            ax.plot(phi_m, meas, color=COLOURS[ci], lw=1.2,
                    alpha=0.85, label=d['name'])
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

axes[0].legend(fontsize=8, loc='upper right', ncol=min(N+1, 4))
axes[-1].set_xlabel('Cycle phase φ', fontsize=9)
fig1.suptitle(f'Measured vs Desired — {SESSION} — unfiltered ILCReadyData (100 phase pts)',
              fontsize=11)
plt.tight_layout()
fig1.savefig(OUT_DIR / f'meas_vs_des_unfiltered_{SESSION}.png', dpi=150, bbox_inches='tight')
print(f"\nFig 1 → meas_vs_des_unfiltered_{SESSION}.png")

# ── Figure 2: PV loop ─────────────────────────────────────────────────────────
fig2, (ax_pv, ax_leg) = plt.subplots(1, 2, figsize=(13, 6),
                                      gridspec_kw={'width_ratios': [3, 1]})

# desired loop
ax_pv.plot(des['volume'], des['pressure'], 'k--', lw=2.5,
           label='Desired', zorder=6)
# close the desired loop
ax_pv.plot([des['volume'][-1], des['volume'][0]],
           [des['pressure'][-1], des['pressure'][0]],
           'k--', lw=2.5, zorder=6)

for ci, d in enumerate(itrs):
    df  = d['df']
    vol = df['volume'].values
    col_p = 'pressure' if 'pressure' in df.columns else [c for c in df.columns if 'pres' in c.lower()][0]
    pres = df[col_p].values
    c    = COLOURS[ci]
    ax_pv.plot(vol, pres, color=c, lw=1.5, alpha=0.85, label=d['name'])
    # close the loop
    ax_pv.plot([vol[-1], vol[0]], [pres[-1], pres[0]], color=c, lw=1.5, alpha=0.5)
    # direction arrow at mid-loop
    mid = len(vol) // 2
    ax_pv.annotate('', xy=(vol[mid+1], pres[mid+1]), xytext=(vol[mid], pres[mid]),
                   arrowprops=dict(arrowstyle='->', color=c, lw=1.5))

ax_pv.set_xlabel('Volume (mL)', fontsize=10)
ax_pv.set_ylabel('Pressure (mmHg)', fontsize=10)
ax_pv.grid(True, alpha=0.3)
ax_pv.set_title(f'PV loop — {SESSION} — unfiltered', fontsize=11)

# legend panel
ax_leg.axis('off')
handles = [plt.Line2D([0],[0], color='k', ls='--', lw=2, label='Desired')]
for ci, d in enumerate(itrs):
    handles.append(plt.Line2D([0],[0], color=COLOURS[ci], lw=2, label=d['name']))
ax_leg.legend(handles=handles, loc='center', fontsize=10, frameon=False)

plt.tight_layout()
fig2.savefig(OUT_DIR / f'pv_loop_unfiltered_{SESSION}.png', dpi=150, bbox_inches='tight')
print(f"Fig 2 → pv_loop_unfiltered_{SESSION}.png")
