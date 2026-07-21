"""
Raw combined plot: pressure + geometry overlaid on same time axis,
plus PV loop — all iterations, no phase conversion, no averaging.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pathlib, re
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
from scipy.interpolate import interp1d

SESSION  = '7_17'
EXP_ROOT = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\ILCFiles\Exp_data')
OUT_DIR  = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\PythonCodes\MPC_depthCam\regression_analysis')

def sort_key(p):
    m = re.search(r'(\d+)', p.name)
    return int(m.group(1)) if m else -1

sess_dir = EXP_ROOT / SESSION
itr_dirs = sorted(
    [d for d in sess_dir.iterdir() if d.is_dir() and re.search(r'itr\d+', d.name, re.I)],
    key=sort_key
)
N = len(itr_dirs)
print(f"Iterations: {[d.name for d in itr_dirs]}")

COLOURS = cm.plasma(np.linspace(0.1, 0.85, N))

# ── Load all iterations ───────────────────────────────────────────────────────
itrs = []
for itr_dir in itr_dirs:
    d = {'name': itr_dir.name}

    # geometry: clip to the ILC window using pressure_aligned.csv time range
    p_csv = itr_dir / 'pressure_aligned.csv'
    geom_csv = itr_dir / 'processed_markers_full.csv'

    if not p_csv.exists() or not geom_csv.exists():
        print(f"  {itr_dir.name}: missing files, skipping")
        continue

    pref = pd.read_csv(p_csv)
    t_win_start = pref['elapsed_s'].values[0]
    t_win_end   = pref['elapsed_s'].values[-1]

    gdf = pd.read_csv(geom_csv)
    t_col = 'abs_time_s' if 'abs_time_s' in gdf.columns else 'Time_s'
    tg = gdf[t_col].values

    # clip geometry to ILC window
    mask = (tg >= t_win_start) & (tg <= t_win_end)
    tg_clip   = tg[mask]
    twist_clip = gdf['twist_deg'].values[mask]
    height_clip= gdf['height_mm'].values[mask]
    vol_clip   = gdf['volume_mL'].values[mask]

    # relative time (0 = start of ILC window)
    t_rel = tg_clip - t_win_start

    # LVP: load from .mat, align, interpolate onto geometry time grid
    lvp_path = itr_dir / 'LVP.mat'
    if lvp_path.exists():
        try:
            mat   = loadmat(str(lvp_path), squeeze_me=True, struct_as_record=False)
            raw   = mat['OutPressure']
            t_mat = raw[0, :].astype(float)
            p_mat = raw[1, :].astype(float)
            # align mat to recording time base
            offset = t_win_start - t_mat[0]
            t_mat  = t_mat + offset
            # clip mat to window
            m2    = (t_mat >= t_win_start) & (t_mat <= t_win_end)
            t_mat_clip = t_mat[m2] - t_win_start
            p_mat_clip = p_mat[m2]
            # resample to geometry time grid for PV loop
            f_p   = interp1d(t_mat_clip, p_mat_clip,
                             bounds_error=False, fill_value=np.nan)
            p_on_geom = f_p(t_rel)
        except Exception as e:
            print(f"  LVP load failed for {itr_dir.name}: {e}")
            t_mat_clip = pref['elapsed_s'].values - t_win_start
            p_mat_clip = pref['pressure_mmhg'].values
            p_on_geom  = np.interp(t_rel, t_mat_clip, p_mat_clip)
    else:
        t_mat_clip = pref['elapsed_s'].values - t_win_start
        p_mat_clip = pref['pressure_mmhg'].values
        p_on_geom  = np.interp(t_rel, t_mat_clip, p_mat_clip)

    d.update(dict(
        t_rel=t_rel,
        twist=twist_clip, height=height_clip, volume=vol_clip,
        t_p=t_mat_clip, pressure_raw=p_mat_clip,
        pressure_on_geom=p_on_geom,
    ))
    itrs.append(d)
    print(f"  {itr_dir.name}: geom pts={len(tg_clip)}, pres pts={len(p_mat_clip)}, t=[{t_rel[0]:.1f}..{t_rel[-1]:.1f}]s")

# ── Figure 1: Combined time-series ────────────────────────────────────────────
fig1, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
labels = ['Twist (°)', 'Height (mm)', 'Volume (mL)', 'Pressure (mmHg)']

for ci, d in enumerate(itrs):
    c   = COLOURS[ci]
    lbl = d['name']
    axes[0].plot(d['t_rel'], d['twist'],        color=c, lw=0.8, label=lbl)
    axes[1].plot(d['t_rel'], d['height'],       color=c, lw=0.8, label=lbl)
    axes[2].plot(d['t_rel'], d['volume'],       color=c, lw=0.8, label=lbl)
    axes[3].plot(d['t_p'],   d['pressure_raw'], color=c, lw=0.5, alpha=0.75, label=lbl)

for ax, lbl in zip(axes, labels):
    ax.set_ylabel(lbl, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)

axes[0].legend(fontsize=8, loc='upper right', ncol=N)
axes[3].set_xlabel('Time within ILC window (s)', fontsize=9)
fig1.suptitle(f'Raw unfiltered — {SESSION} — ILC window only  (all iterations overlaid)',
              fontsize=11)
plt.tight_layout()
fig1.savefig(OUT_DIR / f'raw_combined_{SESSION}.png', dpi=150, bbox_inches='tight')
print(f"\nFig 1 → raw_combined_{SESSION}.png")

# ── Figure 2: PV loops ────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Left: all iterations overlaid
ax_all = axes2[0]
for ci, d in enumerate(itrs):
    c = COLOURS[ci]
    v = d['volume']
    p = d['pressure_on_geom']
    ok = np.isfinite(v) & np.isfinite(p)
    ax_all.plot(v[ok], p[ok], color=c, lw=0.7, alpha=0.8, label=d['name'])
    # arrow on last point to show direction
    idx = np.where(ok)[0]
    if len(idx) > 10:
        i0 = idx[len(idx)//2]
        ax_all.annotate('', xy=(v[i0+1], p[i0+1]), xytext=(v[i0], p[i0]),
                        arrowprops=dict(arrowstyle='->', color=c, lw=1.2))

ax_all.set_xlabel('Volume (mL)', fontsize=10)
ax_all.set_ylabel('LVP (mmHg)', fontsize=10)
ax_all.legend(fontsize=8)
ax_all.grid(True, alpha=0.3)
ax_all.set_title('PV loop — all iterations overlaid', fontsize=10)

# Right: per-iteration small multiples
ax_grid = axes2[1]
ax_grid.set_visible(False)
fig2.delaxes(ax_grid)

n_cols = min(N, 3)
n_rows = int(np.ceil(N / n_cols))
gs = fig2.add_gridspec(n_rows, n_cols,
                        left=0.54, right=0.98, top=0.92, bottom=0.1,
                        hspace=0.45, wspace=0.35)

for ci, d in enumerate(itrs):
    r, c_idx = divmod(ci, n_cols)
    ax = fig2.add_subplot(gs[r, c_idx])
    c  = COLOURS[ci]
    v  = d['volume']
    p  = d['pressure_on_geom']
    ok = np.isfinite(v) & np.isfinite(p)
    ax.plot(v[ok], p[ok], color=c, lw=0.8)
    ax.set_title(d['name'], fontsize=8)
    ax.set_xlabel('Vol (mL)', fontsize=7)
    ax.set_ylabel('LVP (mmHg)', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)

fig2.suptitle(f'PV loops — raw unfiltered — {SESSION}', fontsize=11)
fig2.savefig(OUT_DIR / f'pv_loop_raw_{SESSION}.png', dpi=150, bbox_inches='tight')
print(f"Fig 2 → pv_loop_raw_{SESSION}.png")
