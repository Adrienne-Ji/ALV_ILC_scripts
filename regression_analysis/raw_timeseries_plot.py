"""
Raw unfiltered time-series plot — pressure + geometry for all iterations.
No phase conversion, no averaging. Just the raw signals vs time (seconds).
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pathlib, re
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ── Config ────────────────────────────────────────────────────────────────────
SESSION   = '7_17'
EXP_ROOT  = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\ILCFiles\Exp_data')
OUT_DIR   = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\PythonCodes\MPC_depthCam\regression_analysis')

def sort_key(p):
    m = re.search(r'(\d+)', p.name)
    return int(m.group(1)) if m else -1

# ── Gather iteration folders ──────────────────────────────────────────────────
sess_dir = EXP_ROOT / SESSION
itr_dirs = sorted(
    [d for d in sess_dir.iterdir() if d.is_dir() and re.search(r'itr\d+', d.name, re.I)],
    key=sort_key
)
print(f"Found {len(itr_dirs)} iterations: {[d.name for d in itr_dirs]}")

# ── Layout: 4 rows (Twist / Height / Volume / Pressure) × N_itr cols ─────────
N = len(itr_dirs)
ROWS = 4
fig, axes = plt.subplots(ROWS, N, figsize=(5 * N, 10), sharey='row')
if N == 1: axes = axes[:, np.newaxis]   # keep 2-D indexing

ROW_LABELS = ['Twist (°)', 'Height (mm)', 'Volume (mL)', 'Pressure (mmHg)']
GEOM_COLS  = ['twist_deg', 'height_mm', 'volume_mL']
COLOURS    = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']

for col_i, itr_dir in enumerate(itr_dirs):
    colour = COLOURS[col_i % len(COLOURS)]
    itr_label = itr_dir.name

    # ── Geometry (processed_markers_full.csv) ─────────────────────────────────
    geom_path = itr_dir / 'processed_markers_full.csv'
    if geom_path.exists():
        gdf = pd.read_csv(geom_path)
        t_col = 'abs_time_s' if 'abs_time_s' in gdf.columns else 'Time_s'
        tg = gdf[t_col].values

        for row_i, gcol in enumerate(GEOM_COLS):
            ax = axes[row_i, col_i]
            if gcol in gdf.columns:
                ax.plot(tg, gdf[gcol].values, color=colour, lw=0.6, alpha=0.85)
            ax.set_title(itr_label, fontsize=8)
            if col_i == 0:
                ax.set_ylabel(ROW_LABELS[row_i], fontsize=9)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)
    else:
        for row_i in range(3):
            axes[row_i, col_i].text(0.5, 0.5, 'no geom file',
                                    ha='center', va='center', transform=axes[row_i, col_i].transAxes)

    # ── LVP raw from .mat ─────────────────────────────────────────────────────
    ax_p = axes[3, col_i]
    lvp_path = itr_dir / 'LVP.mat'
    aop_path = itr_dir / 'AOP.mat'
    p_csv    = itr_dir / 'pressure_aligned.csv'

    plotted_mat = False
    for mat_path, label, lc in [(lvp_path, 'LVP', 'tab:red'), (aop_path, 'AOP', 'steelblue')]:
        if mat_path.exists():
            try:
                mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
                raw = mat['OutPressure']
                t_raw = raw[0, :].astype(float)
                p_raw = raw[1, :].astype(float)
                # shift to same abs_time reference using pressure_aligned.csv anchor
                if p_csv.exists():
                    pref = pd.read_csv(p_csv)
                    offset = pref['elapsed_s'].values[0] - t_raw[0]
                    t_raw  = t_raw + offset
                ax_p.plot(t_raw, p_raw, lw=0.4, alpha=0.7, color=lc, label=label)
                plotted_mat = True
            except Exception as e:
                print(f"  {mat_path.name} load failed: {e}")

    if not plotted_mat and p_csv.exists():
        pref = pd.read_csv(p_csv)
        ax_p.plot(pref['elapsed_s'].values, pref['pressure_mmhg'].values,
                  lw=0.6, alpha=0.85, color='tab:red', label='LVP (csv)')

    if col_i == 0:
        ax_p.set_ylabel(ROW_LABELS[3], fontsize=9)
    ax_p.legend(fontsize=6, loc='upper right')
    ax_p.grid(True, alpha=0.25)
    ax_p.tick_params(labelsize=7)
    ax_p.set_xlabel('Time (s)', fontsize=8)

fig.suptitle(f'Raw unfiltered time-series — {SESSION}  (no phase conversion, no averaging)',
             fontsize=11, y=1.01)
plt.tight_layout()

out_path = OUT_DIR / f'raw_timeseries_{SESSION}.png'
fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
print(f"\nSaved → {out_path}")
