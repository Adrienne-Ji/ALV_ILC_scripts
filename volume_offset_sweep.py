"""
volume_offset_sweep.py
──────────────────────
Find which VOLUME_OFFSET values put ALL points of every desired trajectory
within the FK model's achievable volume range at each trajectory pressure.

Achievable range comes from training data min/max per pressure level.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pathlib

# ── FK training achievable envelope (from retrain_sindy.py output) ────────────
TRAIN_P    = np.array([  0,   30,   60,   90,  120,  150], dtype=float)
TRAIN_VMIN = np.array([ 50.2, 51.0, 58.2, 63.9, 68.5, 87.6])
TRAIN_VMAX = np.array([131.1,142.1,155.1,156.6,178.6,215.5])

vmin_fn = interp1d(TRAIN_P, TRAIN_VMIN, kind='linear', bounds_error=False,
                   fill_value=(TRAIN_VMIN[0], TRAIN_VMIN[-1]))
vmax_fn = interp1d(TRAIN_P, TRAIN_VMAX, kind='linear', bounds_error=False,
                   fill_value=(TRAIN_VMAX[0], TRAIN_VMAX[-1]))

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAJ_DIR = pathlib.Path(
    r'C:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator'
    r'\ArtificialHeart\ILCFiles\Engineered_trajs')

CASES = {
    'healthy':   TRAJ_DIR / 'engineered_data_healthy.csv',
    'diastolic': TRAJ_DIR / 'engineered_data_diastolic_dysfunction.csv',
    'systolic':  TRAJ_DIR / 'engineered_data_systolic_dysfunction.csv',
}
COLORS = {'healthy': 'tab:blue', 'diastolic': 'tab:orange', 'systolic': 'tab:green'}

offsets = np.arange(-20, 21, 1)

# ── Analysis ──────────────────────────────────────────────────────────────────
print(f"\n{'─'*62}")
print(f"  VOLUME OFFSET FEASIBILITY SWEEP")
print(f"{'─'*62}")

case_data = {}
for case_name, csv_path in CASES.items():
    df      = pd.read_csv(csv_path)
    p       = df['pressure'].values
    v       = df['volume'].values

    v_floor = vmin_fn(p)   # achievable min at each point's pressure
    v_ceil  = vmax_fn(p)   # achievable max at each point's pressure

    # Analytical feasibility bounds for the offset:
    #   offset >= (v_floor_i - v_i)  for every i  → lower_bound = max of these
    #   offset <= (v_ceil_i  - v_i)  for every i  → upper_bound = min of these
    lb = np.max(v_floor - v)    # minimum offset needed (tightest floor constraint)
    ub = np.min(v_ceil  - v)    # maximum offset allowed (tightest ceiling constraint)

    # Worst-offending points
    floor_gap_idx = np.argmax(v_floor - v)   # point furthest below floor (needs most shift up)
    ceil_gap_idx  = np.argmin(v_ceil  - v)   # point closest to ceiling  (limits shift up)

    # Feasibility fraction at each swept offset
    frac = np.array([np.mean(((v + off) >= v_floor) & ((v + off) <= v_ceil))
                     for off in offsets])

    case_data[case_name] = dict(p=p, v=v, v_floor=v_floor, v_ceil=v_ceil,
                                lb=lb, ub=ub, frac=frac,
                                floor_gap_idx=floor_gap_idx,
                                ceil_gap_idx=ceil_gap_idx)

    print(f"\n  {case_name.upper()}")
    print(f"    Tightest floor constraint:  p={p[floor_gap_idx]:.0f} mmHg  "
          f"v_desired={v[floor_gap_idx]:.1f} mL  "
          f"v_floor={v_floor[floor_gap_idx]:.1f} mL  "
          f"→ need offset ≥ {lb:+.1f} mL")
    print(f"    Tightest ceiling constraint: p={p[ceil_gap_idx]:.0f} mmHg  "
          f"v_desired={v[ceil_gap_idx]:.1f} mL  "
          f"v_ceil={v_ceil[ceil_gap_idx]:.1f} mL  "
          f"→ need offset ≤ {ub:+.1f} mL")
    if lb <= ub:
        mid = round((lb + ub) / 2)
        print(f"    ✅ FEASIBLE RANGE: [{lb:+.1f}, {ub:+.1f}] mL  →  recommend offset = {mid:+d} mL")
    else:
        best_idx = int(np.argmax(frac))
        print(f"    ❌ NO SINGLE OFFSET ACHIEVES 100% FEASIBILITY")
        print(f"       Gap = {lb - ub:.1f} mL  (floor constraint {lb:+.1f} exceeds ceiling {ub:+.1f})")
        print(f"       Best offset = {offsets[best_idx]:+d} mL → {100*frac[best_idx]:.1f}% feasible")

print(f"\n{'─'*62}\n")

# ── Figure 1: feasibility % vs offset for all 3 cases ────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 5))
for case_name, d in case_data.items():
    ax1.plot(offsets, 100 * d['frac'], 'o-', lw=2, ms=5,
             color=COLORS[case_name], label=case_name)
    lb, ub = d['lb'], d['ub']
    if lb <= ub:
        ax1.axvspan(lb, ub, alpha=0.12, color=COLORS[case_name])

ax1.axhline(100, color='k', lw=1.2, ls='--', label='100% feasible')
ax1.axvline(0,   color='grey', lw=0.8, ls=':')
ax1.set_xlabel('VOLUME_OFFSET (mL)', fontsize=11)
ax1.set_ylabel('Trajectory points within FK range (%)', fontsize=11)
ax1.set_title('Feasibility vs volume offset\n'
              '(shaded band = 100% feasible for that case)', fontsize=11)
ax1.legend(fontsize=10); ax1.grid(True, alpha=0.3)
ax1.set_xlim(-20, 20); ax1.set_ylim(0, 105)
plt.tight_layout()

# ── Figure 2: achievable envelope vs trajectory per case ─────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
p_plot = np.linspace(0, 150, 300)

for ax, (case_name, d) in zip(axes, case_data.items()):
    lb, ub = d['lb'], d['ub']
    opt_off = round(np.clip((lb + ub) / 2, lb, ub)) if lb <= ub else round(offsets[int(np.argmax(d['frac']))])

    # Achievable envelope
    ax.fill_between(p_plot, vmin_fn(p_plot), vmax_fn(p_plot),
                    alpha=0.15, color='grey', label='FK achievable range')
    ax.plot(p_plot, vmin_fn(p_plot), 'k--', lw=1)
    ax.plot(p_plot, vmax_fn(p_plot), 'k--', lw=1)

    # Current trajectory (offset=0)
    ax.scatter(d['p'], d['v'],
               c=COLORS[case_name], s=18, alpha=0.6,
               label=f'Desired (offset=0)')

    # Shifted trajectory at optimal offset
    ax.scatter(d['p'], d['v'] + opt_off,
               c=COLORS[case_name], s=18, marker='^', alpha=0.9,
               label=f'Desired (offset={opt_off:+d} mL)')

    ax.set_xlabel('Pressure (mmHg)', fontsize=10)
    ax.set_ylabel('Volume (mL)', fontsize=10)
    ax.set_title(f'{case_name}\n'
                 f'{"Feasible range: [" + f"{lb:+.1f}, {ub:+.1f}] mL" if lb<=ub else "❌ No feasible offset"}',
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig2.suptitle('Trajectory points vs FK achievable envelope\n'
              '(circles=current, triangles=at recommended offset)', fontsize=11)
plt.tight_layout()
plt.show()
