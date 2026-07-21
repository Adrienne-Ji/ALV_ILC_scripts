"""
Shared-axis raw time series — 7_17 healthy itr4.
Geometry: recomputed from raw MDC (Motive) CSV with NO filtering at all.
Pressure: raw LVP.mat, shifted onto MDC time base.
Plot spans full recording from t=0. Vertical reference lines at cycle peaks.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pathlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ITR_DIR = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\ILCFiles\Exp_data\7_17\healthy itr4')
OUT_DIR = pathlib.Path(r'c:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator\ArtificialHeart\PythonCodes\MPC_depthCam\regression_analysis')

# ── Geometry constants (from markerProcessing.py) ─────────────────────────────
WALL_THICKNESS_MM     = 20.5
CAP_WALL_THICKNESS_MM = 20.5
CYLINDER_FRAC         = 0.9
CAP_FRAC              = 0.1
CAP_HEIGHT_OFFSET_MM  = 30.0
VOLUME_MEASURED_OFFSET = -15.0   # mL
HEIGHT_MEASURED_SCALE  =  1.1

# ── Parse raw MDC Motive CSV (no filtering) ───────────────────────────────────
mdc_path = list(ITR_DIR.glob('MDC_data*.csv'))[0]
# Row 6 is the actual column header: Timestamp, Time (Seconds), X, Y, Z, X, Y, Z ...
# Marker order: Tag0, Tag1, Tag2, Midpoint, Purple, Pink, Green
mdc_raw = pd.read_csv(mdc_path, skiprows=6, header=0)
mdc_raw.columns = [
    'Timestamp', 't_s',
    'Tag0_X','Tag0_Y','Tag0_Z',
    'Tag1_X','Tag1_Y','Tag1_Z',
    'Tag2_X','Tag2_Y','Tag2_Z',
    'Mid_X','Mid_Y','Mid_Z',
    'Pur_X','Pur_Y','Pur_Z',
    'Pnk_X','Pnk_Y','Pnk_Z',
    'Grn_X','Grn_Y','Grn_Z',
]
mdc_raw = mdc_raw.dropna(subset=['t_s'])
t_m  = mdc_raw['t_s'].values.astype(float)

def col(name): return mdc_raw[name].values.astype(float)

# per-frame centring on Midpoint (in metres)
Mid_X = col('Mid_X');  Mid_Y = col('Mid_Y');  Mid_Z = col('Mid_Z')
Pnk_cX = (col('Pnk_X') - Mid_X)*1e3;  Pnk_cY = (col('Pnk_Y') - Mid_Y)*1e3
Grn_cX = (col('Grn_X') - Mid_X)*1e3;  Grn_cY = (col('Grn_Y') - Mid_Y)*1e3
Pur_cX = (col('Pur_X') - Mid_X)*1e3;  Pur_cY = (col('Pur_Y') - Mid_Y)*1e3

# radii
pink_rBase = np.sqrt(Pnk_cX**2 + Pnk_cY**2)
grn_rBase  = np.sqrt(Grn_cX**2 + Grn_cY**2)
pur_rBase  = np.sqrt(Pur_cX**2 + Pur_cY**2)
rBase_mm   = (grn_rBase + pur_rBase) / 2.0

# base Z (raw, metres)
grn_Z  = col('Grn_Z');  pur_Z = col('Pur_Z')
base_Z = (grn_Z + pur_Z) / 2.0
pnk_Z  = col('Pnk_Z')

# height (mm), with scale correction
height_mm = (pnk_Z - base_Z) * 1e3 * HEIGHT_MEASURED_SCALE

# twist: arctan2(Pink_cx, |Pink_cy|) - base_angle, zeroed at first valid frame
grn_angle = np.degrees(np.arctan2(Grn_cX, np.abs(Grn_cY)))
pur_angle = np.degrees(np.arctan2(Pur_cX, np.abs(Pur_cY)))
both      = np.isfinite(grn_angle) & np.isfinite(pur_angle)
gp_offset = float(np.nanmean((grn_angle - pur_angle)[both])) if both.any() else 0.0
base_angle = np.where(np.isfinite(grn_angle), grn_angle,
             np.where(np.isfinite(pur_angle), pur_angle + gp_offset, np.nan))

twist_abs  = np.degrees(np.arctan2(Pnk_cX, np.abs(Pnk_cY))) - base_angle
first_ok   = np.where(np.isfinite(twist_abs))[0]
twist_deg  = twist_abs - (twist_abs[first_ok[0]] if first_ok.size else 0.0)

# volume (cylinder_cap, mm3 -> mL)
t    = WALL_THICKNESS_MM;   t_cap = CAP_WALL_THICKNESS_MM
a_en = rBase_mm - t;        r_pen = pink_rBase - t_cap
h_cy = CYLINDER_FRAC * height_mm
c_cp = CAP_FRAC * height_mm + (CAP_HEIGHT_OFFSET_MM - t_cap)
vol  = (np.pi * a_en**2 * h_cy + (2.0/3.0) * np.pi * r_pen**2 * c_cp) / 1000.0
bad  = (a_en <= 0) | (h_cy <= 0) | (r_pen <= 0) | (c_cp <= 0)
vol[bad] = np.nan
vol  = vol + VOLUME_MEASURED_OFFSET

print(f"MDC geometry: {len(t_m)} pts, t=[{t_m[0]:.1f}..{t_m[-1]:.1f}]s")
print(f"  height: [{np.nanmin(height_mm):.1f}..{np.nanmax(height_mm):.1f}] mm")
print(f"  twist:  [{np.nanmin(twist_deg):.1f}..{np.nanmax(twist_deg):.1f}] deg")
print(f"  volume: [{np.nanmin(vol):.1f}..{np.nanmax(vol):.1f}] mL")

# ── ALV: cycle reference lines (master clock = MDC/ALV, same start time) ──────
alv_path = list(ITR_DIR.glob('ALV_10Hz*.csv'))[0]
alv = pd.read_csv(alv_path, skiprows=1)
t_alv    = alv['time_sec'].values
mean_act = (alv['Epi_mm'] + alv['Trans_mm'] + alv['Endo_mm']).values / 3.0
peaks, _ = find_peaks(mean_act, distance=80, prominence=1.0)
ref_lines = t_alv[peaks]
print(f"\nALV: {len(t_alv)} pts  Peaks: {len(peaks)} at t={ref_lines[:5]}")

# ── Pressure: LVP.mat shifted to MDC clock ────────────────────────────────────
mat     = loadmat(str(ITR_DIR / 'LVP.mat'), squeeze_me=True, struct_as_record=False)
raw     = mat['OutPressure']
t_p_raw = raw[0, :].astype(float)
p_raw   = raw[1, :].astype(float)
p_ref   = pd.read_csv(ITR_DIR / 'pressure_aligned.csv')
offset  = p_ref['elapsed_s'].values[0] - t_p_raw[0]
t_p     = t_p_raw + offset          # now on MDC clock
# clip both to t >= 0
t_p_ok  = t_p >= 0
t_p_plot = t_p[t_p_ok];  p_plot = p_raw[t_p_ok]
print(f"Pressure: {t_p_ok.sum()} pts, t=[{t_p_plot[0]:.1f}..{t_p_plot[-1]:.1f}]s (MDC clock)")

# ── Plot: 4 shared-axis subplots, full recording from t=0 ─────────────────────
fig, axes = plt.subplots(4, 1, figsize=(20, 10), sharex=True)

VLINE_KW = dict(color='dimgray', lw=0.9, ls='--', alpha=0.6)

sigs = [
    (t_m, twist_deg,  'Twist (°)',   'tab:blue'),
    (t_m, height_mm,  'Height (mm)', 'tab:green'),
    (t_m, vol,        'Volume (mL)', 'tab:orange'),
    (t_p_plot, p_plot,'LVP (mmHg)',  'tab:red'),
]
for ax, (tx, sig, ylabel, c) in zip(axes, sigs):
    ax.plot(tx, sig, color=c, lw=0.6, alpha=0.85)
    for vt in ref_lines:
        ax.axvline(vt, **VLINE_KW)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

# label every other cycle peak on the top axis to avoid clutter
for i, vt in enumerate(ref_lines):
    if i % 2 == 0:
        axes[0].text(vt + 0.5, axes[0].get_ylim()[1] * 0.9,
                     f'C{i}', fontsize=6, color='dimgray', va='top')

axes[-1].set_xlabel('MDC / ALV clock — time (s)', fontsize=9)
axes[0].set_title(
    f'7_17 healthy itr4 — raw unfiltered  '
    f'(geometry from MDC Motive CSV, NO filter  |  '
    f'pressure = raw LVP.mat, offset={offset:.2f}s)\n'
    f'Full recording t=0..end  |  dashed lines = actuator peaks (cycle boundaries)',
    fontsize=9)

plt.tight_layout()
out = OUT_DIR / 'aligned_timeseries_itr4.png'
fig.savefig(str(out), dpi=150, bbox_inches='tight')
print(f"\nSaved → {out}")
