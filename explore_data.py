"""
explore_data.py
───────────────
Load all deformation_by_actuator_{p}mmhg.csv files and plot all data in 3D
deformation space (X=volume, Y=height, Z=twist) with colour representing
the nominal pressure level.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ── Paths ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
SHARED = os.path.join(BASE, "..", "..", "sharedCSVs")

PRESSURES = [0, 30, 60, 90, 120, 150]

# ── Load ─────────────────────────────────────────────────────────────────────
print("Loading data …")
frames, loaded_pressures = [], []
for p in PRESSURES:
    fpath = os.path.join(SHARED, f"deformation_by_actuator_V_adjusted_{p}mmhg.csv")
    if not os.path.exists(fpath):
        print(f"  {p:3d} mmHg → file not found, skipping")
        continue
    df = pd.read_csv(fpath).dropna()
    df['nominal_pressure'] = p
    frames.append(df)
    loaded_pressures.append(p)
    print(f"  {p:3d} mmHg → {len(df):4d} rows")

assert frames, "No data files found — check SHARED path."
all_data = pd.concat(frames, ignore_index=True)
print(f"  Total: {len(all_data)} rows\n")

# ── Colour map: one distinct colour per pressure level ───────────────────────
N_P    = len(loaded_pressures)
cmap   = cm.get_cmap('plasma', N_P)
COLORS = {p: cmap(i) for i, p in enumerate(loaded_pressures)}

# ── 3D scatter: X=volume, Y=height, Z=twist, colour=pressure ─────────────────
fig = plt.figure(figsize=(11, 8))
ax  = fig.add_subplot(111, projection='3d')

for p, df in zip(loaded_pressures, frames):
    ax.scatter(
        df['volume_endo_mL'],
        df['height_mm'],
        df['dtwist_deg'],
        c=[COLORS[p]],
        s=8,
        alpha=0.6,
        label=f'{p} mmHg',
        depthshade=True,
    )

ax.set_xlabel('Volume (mL)',  fontsize=11, labelpad=8)
ax.set_ylabel('Height (mm)', fontsize=11, labelpad=8)
ax.set_zlabel('Twist (deg)', fontsize=11, labelpad=8)
ax.set_title('deformation_by_actuator\n(colour = nominal pressure)', fontsize=12, pad=12)

ax.legend(title='Pressure (mmHg)', fontsize=9, title_fontsize=10,
          loc='upper left', bbox_to_anchor=(0.0, 1.0),
          framealpha=0.8, markerscale=2)

# ── Load desired trajectory ───────────────────────────────────────────────────
ENG_CSV = os.path.join(BASE, "..", "engineered_data_withP.csv")
eng     = pd.read_csv(ENG_CSV)
traj_twist  = eng['twist'].values                      # deg
traj_height = eng['height'].values + 70                # mm  (same offset as FK script)
# traj_volume = (eng['volume'].values-60)*0.75 +75   # mL  (same scaling as FK script) volume range (DCM: 125 -165) normal range 75 to 135ml 
traj_volume = eng['volume'].values + 15   # mL
traj_p      = eng['pressure'].values                   # mmHg (for colouring)

# Overlay trajectory as a thick black line + pressure-coloured scatter
ax.plot(traj_volume, traj_height, traj_twist,
        color='black', lw=1.5, alpha=0.7, zorder=5, label='Desired trajectory')
sc_traj = ax.scatter(traj_volume, traj_height, traj_twist,
                     c=traj_p, cmap='coolwarm', s=12, zorder=6,
                     vmin=traj_p.min(), vmax=traj_p.max(), depthshade=False)
plt.colorbar(sc_traj, ax=ax, label='Trajectory pressure (mmHg)', shrink=0.5, pad=0.08)

ax.legend(title='Pressure (mmHg)', fontsize=9, title_fontsize=10,
          loc='upper left', bbox_to_anchor=(0.0, 1.0),
          framealpha=0.8, markerscale=2)
ax.set_title('Data cloud + desired trajectory\n(colour = nominal pressure / trajectory pressure)',
             fontsize=11, pad=12)

plt.tight_layout()
out_path = os.path.join(BASE, 'data_3d_deformation.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved → data_3d_deformation.png")

# ── Side-by-side comparison with PressureSweepData ───────────────────────────
SWEEP_DIR      = os.path.join(BASE, "..", "PressureSweepData")
SWEEP_PRESSURES = [0, 20, 40, 60, 80, 100, 120]

print("\nLoading PressureSweepData …")
sweep_frames, sweep_pressures = [], []
for p in SWEEP_PRESSURES:
    fpath = os.path.join(SWEEP_DIR, f"{p}mmhg_data.csv")
    if not os.path.exists(fpath):
        print(f"  {p:3d} mmHg → file not found, skipping")
        continue
    df = pd.read_csv(fpath).dropna()
    df['nominal_pressure'] = p
    sweep_frames.append(df)
    sweep_pressures.append(p)
    print(f"  {p:3d} mmHg → {len(df):4d} rows")

# Shared colour scale across both datasets: 0 – 150 mmHg
norm  = plt.Normalize(vmin=0, vmax=150)
cmap2 = cm.get_cmap('plasma')

fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7),
                                   subplot_kw={'projection': '3d'})

# Left — deformation_by_actuator
for p, df in zip(loaded_pressures, frames):
    ax_l.scatter(df['volume_endo_mL'], df['height_mm'], df['dtwist_deg'],
                 c=[cmap2(norm(p))], s=8, alpha=0.6, label=f'{p} mmHg',
                 depthshade=True)
ax_l.set_xlabel('Volume (mL)',  fontsize=10, labelpad=8)
ax_l.set_ylabel('Height (mm)', fontsize=10, labelpad=8)
ax_l.set_zlabel('Twist (deg)', fontsize=10, labelpad=8)
ax_l.set_title('deformation_by_actuator\n(0–150 mmHg, Δ30)', fontsize=11)
ax_l.plot(traj_volume, traj_height, traj_twist,
          color='black', lw=2, zorder=5, label='Desired traj.')
ax_l.legend(title='Pressure', fontsize=8, title_fontsize=9, markerscale=2,
            loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.7)

# Right — PressureSweepData
for p, df in zip(sweep_pressures, sweep_frames):
    ax_r.scatter(df['volume'], df['height'], df['twist'],
                 c=[cmap2(norm(p))], s=8, alpha=0.6, label=f'{p} mmHg',
                 depthshade=True)
ax_r.plot(traj_volume, traj_height, traj_twist,
          color='black', lw=2, zorder=5, label='Desired traj.')
ax_r.set_xlabel('Volume (mL)',  fontsize=10, labelpad=8)
ax_r.set_ylabel('Height (mm)', fontsize=10, labelpad=8)
ax_r.set_zlabel('Twist (deg)', fontsize=10, labelpad=8)
ax_r.set_title('PressureSweepData\n(0–120 mmHg, Δ20)', fontsize=11)
ax_r.legend(title='Pressure', fontsize=8, title_fontsize=9, markerscale=2,
            loc='upper left', bbox_to_anchor=(0, 1), framealpha=0.7)

# Shared axis limits across both plots
all_vol    = pd.concat([all_data['volume_endo_mL']] +
                       [df['volume'] for df in sweep_frames] +
                       [pd.Series(traj_volume)])
all_height = pd.concat([all_data['height_mm']] +
                       [df['height'] for df in sweep_frames] +
                       [pd.Series(traj_height)])
all_twist  = pd.concat([all_data['dtwist_deg']] +
                       [df['twist'] for df in sweep_frames] +
                       [pd.Series(traj_twist)])

# Equal-length axes: find the largest range and apply it centred on each axis
mid_vol    = (all_vol.min()    + all_vol.max())    / 2
mid_height = (all_height.min() + all_height.max()) / 2
mid_twist  = (all_twist.min()  + all_twist.max())  / 2

half = max(all_vol.max()    - all_vol.min(),
           all_height.max() - all_height.min(),
           all_twist.max()  - all_twist.min()) / 2

vol_lim    = (mid_vol    - half, mid_vol    + half)
height_lim = (mid_height - half, mid_height + half)
twist_lim  = (mid_twist  - half, mid_twist  + half)

for ax in (ax_l, ax_r):
    ax.set_xlim(vol_lim)
    ax.set_ylim(height_lim)
    ax.set_zlim(twist_lim)
    ax.set_box_aspect([1, 1, 1])

# Shared colourbar
sm = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
sm.set_array([])
fig2.colorbar(sm, ax=[ax_l, ax_r], label='Nominal pressure (mmHg)',
              shrink=0.5, pad=0.05)

plt.suptitle('Deformation Space Comparison — same colour scale', fontsize=13)
plt.tight_layout()
out_path2 = os.path.join(BASE, 'data_3d_comparison.png')
fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
print(f"Saved → data_3d_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# KDE — training data density evaluated along the desired trajectory
#
# KDE is fit in 4D: (volume, height, twist, pressure).
# Because pressure correlates with geometry, this tells us whether each
# trajectory point (geometry + pressure) was well-sampled during training.
# Low density → FK model is likely extrapolating there.
# ══════════════════════════════════════════════════════════════════════════════
print("\nFitting 4-D KDE on training data …")
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

# Build 4-D training matrix: (volume, height, twist, pressure)
X_train = np.column_stack([
    all_data['volume_endo_mL'].values,
    all_data['height_mm'].values,
    all_data['dtwist_deg'].values,
    all_data['nominal_pressure'].values,
])

# Trajectory matrix (same 4 columns, same order)
X_traj = np.column_stack([traj_volume, traj_height, traj_twist, traj_p])
traj_time = eng['time'].values

# Standardise so KDE bandwidth is scale-independent across variables
scaler    = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train).T   # (4, N) for gaussian_kde
X_traj_s  = scaler.transform(X_traj).T   # (4, T)

kde           = gaussian_kde(X_train_s)
traj_density  = kde(X_traj_s)             # (T,) — higher = better sampled
train_density = kde(X_train_s)            # reference: density at training pts

# Normalise so training-set median = 1  (makes threshold intuitive)
traj_density_n  = traj_density  / np.median(train_density)
LOW_THRESH      = 0.3   # below 30% of median training density → sparse

print(f"  Trajectory density range: [{traj_density_n.min():.3f}, {traj_density_n.max():.3f}]"
      f"  (relative to training median)")
print(f"  Fraction of trajectory in low-density region (<{LOW_THRESH}): "
      f"{(traj_density_n < LOW_THRESH).mean()*100:.1f}%")

# ── Fig A: density + pressure along trajectory over time ─────────────────────
figA, (ax_d, ax_p) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

ax_d.plot(traj_time, traj_density_n, color='steelblue', lw=1.5)
ax_d.fill_between(traj_time, 0, traj_density_n, alpha=0.25, color='steelblue')
ax_d.axhline(LOW_THRESH, color='red', lw=1.2, ls='--',
             label=f'Low-density threshold ({LOW_THRESH})')
ax_d.axhline(1.0, color='grey', lw=1, ls=':', label='Training median')
ax_d.set_ylabel('Relative KDE density', fontsize=10)
ax_d.set_title('4-D KDE: training data density along desired trajectory\n'
               '(volume, height, twist, pressure)', fontsize=12)
ax_d.legend(fontsize=9); ax_d.grid(True, alpha=0.3)

ax_p.plot(traj_time, traj_p, color='darkorange', lw=1.5)
ax_p.set_ylabel('Pressure (mmHg)', fontsize=10)
ax_p.set_xlabel('Time (s)', fontsize=10)
ax_p.set_title('Trajectory pressure over time', fontsize=11)
ax_p.grid(True, alpha=0.3)

plt.tight_layout()
figA.savefig(os.path.join(BASE, 'kde_density_time.png'), dpi=150, bbox_inches='tight')
print("  Saved → kde_density_time.png")
plt.close(figA)

# ── Fig B: 3-D scatter — trajectory coloured by KDE density ──────────────────
figB = plt.figure(figsize=(11, 8))
axB  = figB.add_subplot(111, projection='3d')

# Data cloud — grey background
axB.scatter(all_data['volume_endo_mL'], all_data['height_mm'], all_data['dtwist_deg'],
            c='lightgrey', s=5, alpha=0.3, depthshade=True, label='Training data')

# Trajectory coloured by density (green=well-sampled, red=sparse)
sc = axB.scatter(traj_volume, traj_height, traj_twist,
                 c=traj_density_n, cmap='RdYlGn',
                 vmin=0, vmax=2,
                 s=18, zorder=6, depthshade=False)
plt.colorbar(sc, ax=axB, label='Relative KDE density\n(1 = training median)',
             shrink=0.55, pad=0.08)

axB.set_xlabel('Volume (mL)',  fontsize=10, labelpad=8)
axB.set_ylabel('Height (mm)', fontsize=10, labelpad=8)
axB.set_zlabel('Twist (deg)', fontsize=10, labelpad=8)
axB.set_title('Trajectory coverage: red = sparse training data at that\n'
              '(geometry, pressure) combination', fontsize=11)
axB.legend(fontsize=9)

plt.tight_layout()
figB.savefig(os.path.join(BASE, 'kde_density_3d.png'), dpi=150, bbox_inches='tight')
print("  Saved → kde_density_3d.png")
plt.close(figB)

# AUTO OFFSET SEARCH — muted (offsets are now fixed in the USER SETTINGS above)
# To re-enable: uncomment this block
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# RANGE DIAGNOSTIC — training data min/max per pressure level vs trajectory
# Shows exactly where the trajectory sits relative to sampled data range.
# Inside the blue band = covered. Outside = extrapolation.
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating range diagnostic …")

outputs    = ['volume_endo_mL', 'height_mm',  'dtwist_deg']
traj_vals  = [traj_volume,       traj_height,   traj_twist]
out_labels = ['Volume (mL)',     'Height (mm)', 'Twist (deg)']

fig_d, axes_d = plt.subplots(1, 3, figsize=(17, 5))

for ax, col, tval, label in zip(axes_d, outputs, traj_vals, out_labels):
    p_vals, medians, mins, maxs = [], [], [], []
    for p, df in zip(loaded_pressures, frames):
        p_vals.append(p)
        medians.append(df[col].median())
        mins.append(df[col].min())
        maxs.append(df[col].max())

    ax.fill_between(p_vals, mins, maxs, alpha=0.25, color='steelblue',
                    label='Training range (min–max)')
    ax.plot(p_vals, medians, 'o-', color='steelblue', lw=2, label='Training median')

    sc = ax.scatter(traj_p, tval, c=traj_time, cmap='viridis',
                    s=10, alpha=0.7, zorder=5, label='Trajectory')
    plt.colorbar(sc, ax=ax, label='Time (s)', shrink=0.7)

    ax.set_xlabel('Pressure (mmHg)', fontsize=10)
    ax.set_ylabel(label, fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Training data range vs trajectory at each pressure level\n'
             'Trajectory inside blue band = well covered by training data', fontsize=12)
plt.tight_layout()
fig_d.savefig(os.path.join(BASE, 'kde_range_diagnostic.png'), dpi=150, bbox_inches='tight')
print("  Saved → kde_range_diagnostic.png")
plt.close(fig_d)

# ══════════════════════════════════════════════════════════════════════════════
# ACTUATOR SPACE DIAGNOSTIC
# Check whether MPC-computed actuator positions fall within the training data
# actuator range. Points outside = the FK model is extrapolating in INPUT space.
# Also flags positions likely hitting the physical safety cap.
# ══════════════════════════════════════════════════════════════════════════════
print("\nActuator space diagnostic …")

MPC_FILES = {
    'DataDriven': os.path.join(BASE, 'MPC_actuators_DataDriven.csv'),
    'PINN':       os.path.join(BASE, 'MPC_actuators_PINN.csv'),
    'SINDy':      os.path.join(BASE, 'MPC_actuators_SINDy.csv'),
    'SINDy2':     os.path.join(BASE, 'MPC_actuators_SINDy2.csv'),
    'pSINDy':     os.path.join(BASE, 'MPC_actuators_pSINDy.csv'),
}
ACT_COLS   = ['epi', 'trans', 'endo']
ACT_UNITS  = ['mm', 'mm', 'mm']
ACT_COLORS = {'DataDriven': '#d62728', 'PINN': '#2ca02c',
              'SINDy': '#1f77b4', 'SINDy2': '#9467bd', 'pSINDy': '#e377c2'}

# Training data actuator ranges
act_min = all_data[ACT_COLS].min()
act_max = all_data[ACT_COLS].max()

# Load available MPC files
mpc_dfs = {}
for arch, fpath in MPC_FILES.items():
    if os.path.exists(fpath):
        mpc_dfs[arch] = pd.read_csv(fpath)
        print(f"  Loaded {arch}: {len(mpc_dfs[arch])} pts")

# ── Achievability diagnostic ──────────────────────────────────────────────────
# For each discrete pressure level, plot the range of deformation values the
# training data can produce (scatter of output vs each actuator).
# Overlay the desired trajectory values at that pressure as horizontal lines.
# If the desired line falls OUTSIDE the scatter cloud → the deformation is
# simply not achievable within the sampled actuator working range at that
# pressure — no actuator combination in training can produce it.
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating achievability diagnostic …")

out_cols   = ['dtwist_deg', 'height_mm', 'volume_endo_mL']
out_labels = ['Twist (deg)', 'Height (mm)', 'Volume (mL)']
traj_out   = np.stack([traj_twist, traj_height, traj_volume], axis=1)

fig_ach, axes_ach = plt.subplots(len(out_cols), len(loaded_pressures),
                                  figsize=(4 * len(loaded_pressures), 4 * len(out_cols)),
                                  sharey='row')

for oi, (out_col, out_lbl) in enumerate(zip(out_cols, out_labels)):
    for pi, (p, df_p) in enumerate(zip(loaded_pressures, frames)):
        ax = axes_ach[oi, pi]

        # Training data scatter: colour by epi (most influential actuator)
        sc = ax.scatter(df_p['epi'], df_p[out_col],
                        c=df_p['trans'], cmap='coolwarm',
                        s=6, alpha=0.5, label='Training (colour=trans)')

        # Desired trajectory values at this pressure level
        # Use trajectory points whose pressure is within ±15 mmHg of this level
        mask = np.abs(traj_p - p) < 15
        if mask.any():
            desired_vals = traj_out[mask, oi]
            ax.axhline(desired_vals.min(), color='red', lw=1.2, ls='--')
            ax.axhline(desired_vals.max(), color='red', lw=1.2, ls='--',
                       label=f'Desired range')
            ax.axhspan(desired_vals.min(), desired_vals.max(),
                       alpha=0.12, color='red')

            # Check if desired range overlaps with training data output range
            train_min = df_p[out_col].min()
            train_max = df_p[out_col].max()
            fully_inside  = (desired_vals.min() >= train_min and
                             desired_vals.max() <= train_max)
            partially_out = not fully_inside

            status = '✓ achievable' if fully_inside else '✗ OUT OF RANGE'
            color  = 'green' if fully_inside else 'red'
            ax.set_title(f'{p} mmHg\n{status}', fontsize=9,
                         color=color, fontweight='bold')
        else:
            ax.set_title(f'{p} mmHg\n(no traj pts)', fontsize=9)

        if pi == 0:
            ax.set_ylabel(out_lbl, fontsize=9)
        ax.set_xlabel('epi (mm)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

plt.suptitle('Achievability: can training data produce the desired deformation?\n'
             'Red band = desired range at that pressure  |  '
             '✗ = desired deformation outside actuator working range',
             fontsize=12)
plt.tight_layout()
fig_ach.savefig(os.path.join(BASE, 'achievability_diagnostic.png'),
                dpi=130, bbox_inches='tight')
print("  Saved → achievability_diagnostic.png")
plt.close(fig_ach)

plt.show()
