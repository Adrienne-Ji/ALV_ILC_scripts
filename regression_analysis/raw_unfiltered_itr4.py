"""
Unfiltered measured vs desired (phase) + PV loop for 7_17 healthy itr4.
- Geometry: processed_markers_full.csv at 10 Hz (no filtering)
- Pressure: LVP.mat at ~1 kHz (no filtering)
- Matching: nearest-neighbour on timestamps (no interpolation)
- Phase: derived from ALV actuator troughs (no signal filtering)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pathlib, sys as _sys
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_sys.path.insert(0, r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\PythonCodes\MPC_depthCam')
from rig_config import HEIGHT_OFFSET, VOLUME_OFFSET

ITR_DIR = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\ILCFiles\Exp_data\7_17\healthy itr4')
ENG_CSV = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\ILCFiles\Engineered_trajs\engineered_data_healthy.csv')
OUT_DIR = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\PythonCodes\MPC_depthCam\regression_analysis')

# ── 1. ALV motor log → find cycle boundaries (troughs of mean actuator) ───────
alv_path = list(ITR_DIR.glob('ALV_10Hz*.csv'))[0]
with open(alv_path) as f:
    header = f.readline().strip()
rec_start_iso = header.split('=')[1]
alv = pd.read_csv(alv_path, skiprows=1)
t_alv  = alv['time_sec'].values
mean_act = (alv['Epi_mm'].values + alv['Trans_mm'].values + alv['Endo_mm'].values) / 3.0

# find peaks (cycle starts) — actuator is longest at end-diastole (max volume)
# desired trajectory starts at φ=0 = max volume = actuator at max extension
peaks, _ = find_peaks(mean_act, distance=80, prominence=1.0)
print(f"ALV: {len(t_alv)} pts, t=[{t_alv[0]:.1f}..{t_alv[-1]:.1f}]s")
print(f"Peaks found: {len(peaks)} at t={t_alv[peaks[:8]]}")

# use the last N complete cycles for analysis (steady-state)
N_CYCLES_USE = 5
if len(peaks) >= N_CYCLES_USE + 1:
    use_troughs = peaks[-(N_CYCLES_USE + 1):]
else:
    use_troughs = peaks
t_win_start = t_alv[use_troughs[0]]
t_win_end   = t_alv[use_troughs[-1]]
print(f"Analysis window: t=[{t_win_start:.1f}..{t_win_end:.1f}]s  ({N_CYCLES_USE} cycles)")

# ── 2. Geometry: processed_markers_full.csv ────────────────────────────────────
gdf  = pd.read_csv(ITR_DIR / 'processed_markers_full.csv')
tg   = gdf['abs_time_s'].values
mask = (tg >= t_win_start) & (tg <= t_win_end)
tg   = tg[mask]
twist  = gdf['twist_deg'].values[mask]
height = gdf['height_mm'].values[mask]
volume = gdf['volume_mL'].values[mask]
print(f"Geometry: {len(tg)} pts in window")

# ── 3. Pressure: LVP.mat ──────────────────────────────────────────────────────
mat   = loadmat(str(ITR_DIR / 'LVP.mat'), squeeze_me=True, struct_as_record=False)
raw   = mat['OutPressure']
t_p_raw = raw[0, :].astype(float)
p_raw   = raw[1, :].astype(float)

# align pressure time to ALV time base
# ALV recording start → abs offset between LVP.mat t=0 and ALV t=0
# Use pressure_aligned.csv which already encodes the alignment offset
p_csv = pd.read_csv(ITR_DIR / 'pressure_aligned.csv')
# pressure_aligned elapsed_s is in ALV time base
# find the corresponding t_p_raw value at the start of pressure_aligned
p_csv_t0 = p_csv['elapsed_s'].values[0]
p_raw_t0  = t_p_raw[0]
offset    = p_csv_t0 - p_raw_t0
t_p       = t_p_raw + offset
print(f"Pressure: {len(t_p)} pts, t=[{t_p[0]:.1f}..{t_p[-1]:.1f}]s  (offset={offset:.3f}s)")

# clip pressure to window
pm = (t_p >= t_win_start) & (t_p <= t_win_end)
t_p_win = t_p[pm]
p_win   = p_raw[pm]
print(f"Pressure in window: {len(t_p_win)} pts")

# ── 4. Nearest-neighbour match: for each geometry point, find closest pressure ─
# geometry is 10 Hz, pressure is ~1 kHz → nearest neighbour is safe
idx_nn   = np.searchsorted(t_p_win, tg).clip(0, len(t_p_win) - 1)
p_at_geom = p_win[idx_nn]
dt_match  = np.abs(t_p_win[idx_nn] - tg)
print(f"Nearest-neighbour matching: max dt = {dt_match.max()*1000:.2f} ms  mean dt = {dt_match.mean()*1000:.2f} ms")

# ── 5. Phase from ALV troughs ─────────────────────────────────────────────────
# find trough times in the geometry window
trough_t = t_alv[use_troughs]
# for each geometry point, compute phase within its cycle
phi_geom = np.full(len(tg), np.nan)
for i in range(len(trough_t) - 1):
    t0, t1 = trough_t[i], trough_t[i + 1]
    m = (tg >= t0) & (tg < t1)
    phi_geom[m] = (tg[m] - t0) / (t1 - t0)

# remove points not assigned (outside cycles)
valid = np.isfinite(phi_geom)
phi_v  = phi_geom[valid]
tw_v   = twist[valid]
ht_v   = height[valid]
vl_v   = volume[valid]
pr_v   = p_at_geom[valid]

print(f"Phase-assigned points: {valid.sum()} / {len(tg)}")

# sort by phase
order  = np.argsort(phi_v)
phi_s  = phi_v[order]
tw_s   = tw_v[order]
ht_s   = ht_v[order]
vl_s   = vl_v[order]
pr_s   = pr_v[order]

# ── 6. Desired trajectory ─────────────────────────────────────────────────────
eng     = pd.read_csv(ENG_CSV)
t_eng   = eng['time'].values
phi_des = (t_eng - t_eng[0]) / (t_eng[-1] - t_eng[0])
des_tw  = np.interp(phi_s, phi_des, eng['twist'].values)
des_ht  = np.interp(phi_s, phi_des, eng['height'].values) + HEIGHT_OFFSET
des_vl  = np.interp(phi_s, phi_des, eng['volume'].values) + VOLUME_OFFSET
des_pr  = np.interp(phi_s, phi_des, eng['pressure'].values)

# ── 7. Plot: measured vs desired phase ────────────────────────────────────────
fig1, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
CFG = [
    (tw_s, des_tw, 'Twist (°)',      'tab:blue'),
    (ht_s, des_ht, 'Height (mm)',    'tab:green'),
    (vl_s, des_vl, 'Volume (mL)',    'tab:orange'),
    (pr_s, des_pr, 'Pressure (mmHg)','tab:red'),
]
for ax, (meas, des, ylabel, col) in zip(axes, CFG):
    ax.scatter(phi_s, meas, s=4, color=col, alpha=0.4, label='Measured (raw)', zorder=3)
    ax.plot(phi_s, des, 'k--', lw=2, label='Desired', zorder=5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

axes[0].legend(fontsize=9, loc='upper right')
axes[-1].set_xlabel('Cycle phase φ', fontsize=9)
fig1.suptitle(
    f'7_17 healthy itr4 — RAW unfiltered, nearest-neighbour time match\n'
    f'{N_CYCLES_USE} cycles, {valid.sum()} pts, phase from ALV motor troughs',
    fontsize=10)
plt.tight_layout()
fig1.savefig(OUT_DIR / 'raw_unfiltered_meas_vs_des_itr4.png', dpi=150, bbox_inches='tight')
print(f"\nFig 1 → raw_unfiltered_meas_vs_des_itr4.png")

# ── 8. PV loop ────────────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(9, 7))

# individual cycles as faint traces
for i in range(len(trough_t) - 1):
    t0, t1 = trough_t[i], trough_t[i + 1]
    m  = (tg >= t0) & (tg < t1) & np.isfinite(phi_geom)
    vl_c = volume[m]
    pr_c = p_at_geom[m]
    ax.plot(vl_c, pr_c, color='tab:red', lw=0.8, alpha=0.35)

# overall scatter coloured by phase
sc = ax.scatter(vl_v, pr_v, c=phi_v, cmap='hsv', s=6, alpha=0.6, zorder=4)
plt.colorbar(sc, ax=ax, label='Cycle phase φ')

# desired loop
phi_fine = np.linspace(0, 1, 500)
des_vl_f = np.interp(phi_fine, phi_des, eng['volume'].values) + VOLUME_OFFSET
des_pr_f = np.interp(phi_fine, phi_des, eng['pressure'].values)
ax.plot(des_vl_f, des_pr_f, 'k--', lw=2.5, label='Desired', zorder=6)

ax.set_xlabel('Volume (mL)', fontsize=10)
ax.set_ylabel('LVP (mmHg)', fontsize=10)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
ax.set_title('PV loop — 7_17 healthy itr4 — raw unfiltered', fontsize=11)
plt.tight_layout()
fig2.savefig(OUT_DIR / 'raw_unfiltered_pv_loop_itr4.png', dpi=150, bbox_inches='tight')
print(f"Fig 2 → raw_unfiltered_pv_loop_itr4.png")
