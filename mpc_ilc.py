"""
mpc_ilc.py
──────────────────────────────────────────────────────────────────────────────
Inverse Kinematics (IK) via SINDy FK model — generates itr 0 actuator
trajectories for all 3 simulation cases (healthy, diastolic, systolic).

No measured itr 0 data exists yet.  This script optimises actuator positions
point-by-point so that the SINDy FK prediction matches the desired trajectory.
A Q-filter is applied afterwards for smoothness.

Inputs
──────
  ILCFiles/Engineered_trajs/engineered_data_*.csv  — desired trajectories
  saved_models/sindy_data.pkl + norm_constants.npz  — FK model

Outputs (per case)
──────
  ILCFiles/Engineered_trajs/healthy_singleCycle_itr0.csv
  ILCFiles/Engineered_trajs/diastolicdys_singleCycle_itr0.csv
  ILCFiles/Engineered_trajs/systolicdys_singleCycle_itr0.csv
  mpc_ilc_{case}.png
"""

import os, pickle, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from pysindy.feature_library import PolynomialLibrary

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# FK model
ILC_MODEL = 'SINDy'

# Actuator home position (mm) — IK warm-start at the beginning of each trajectory
ACT_HOME = np.array([200.0, 202.0, 200.0])

# Physical actuator bounds (mm)
ACT_MIN = np.array([200.0, 202.0, 200.0])
ACT_MAX = np.array([248.0, 248.0, 248.0])

# Geometry weights [twist, height, volume] — twist boosted 5× (small errors matter)
GEOM_WEIGHTS = np.array([5.0, 1.0, 1.0])

# Q-filter for actuator smoothing after IK solve
Q_CUTOFF = 0.35
Q_ORDER  = 3

# Offsets applied to desired trajectory columns
HEIGHT_OFFSET = 70.0    # mm — from rig_config.py
VOLUME_OFFSET = 0.0     # mL

# PVT conversion — Zaber-readable output
PVT_SESSION_DIR       = 'init'   # subfolder under ILCFiles/ILC_traj/ (change per experiment)
PVT_N_CYCLES          = 15
PVT_SECONDS_PER_CYCLE = 20.0
PVT_RAMP_IN_S         = 20.0
PVT_RAMP_OUT_S        = 20.0

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
BASE             = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR         = pathlib.Path(BASE) / 'saved_models'
ENGINEERED_TRAJS = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'Engineered_trajs'
ILC_TRAJ         = pathlib.Path(BASE).parents[1] / 'ILCFiles' / 'ILC_traj'
OUT_DIR          = ENGINEERED_TRAJS   # IK solutions saved alongside the desired trajectories

_SIM_CASE_FILES = {
    'healthy':   ENGINEERED_TRAJS / 'engineered_data_healthy.csv',
    'diastolic': ENGINEERED_TRAJS / 'engineered_data_diastolic_dysfunction.csv',
    'systolic':  ENGINEERED_TRAJS / 'engineered_data_systolic_dysfunction.csv',
}

_CASE_FILENAMES = {
    'healthy':   'healthy_singleCycle_itr0.csv',
    'diastolic': 'diastolicdys_singleCycle_itr0.csv',
    'systolic':  'systolicdys_singleCycle_itr0.csv',
}

assert SAVE_DIR.exists(), \
    "Run pressure_fk_comparison.py first to generate saved_models/"

# ── Load normalisation constants ───────────────────────────────────────────────
norm  = np.load(SAVE_DIR / 'norm_constants.npz')
x_min = norm['x_min']; x_den = norm['x_den']
y_min = norm['y_min']; y_den = norm['y_den']

def norm_act(act_phys):
    return (np.asarray(act_phys) - x_min[:3]) / x_den[:3]

def denorm_act(act_n):
    return np.asarray(act_n) * x_den[:3] + x_min[:3]

def norm_y(y_phys):
    return (np.asarray(y_phys) - y_min) / y_den

def denorm_y(y_n):
    return np.asarray(y_n) * y_den + y_min

OUTPUT_NAMES = ['Twist_deg', 'Height_mm', 'Volume_mL']
OUTPUT_UNITS = ['deg', 'mm', 'mL']

# ══════════════════════════════════════════════════════════════════════════════
# LOAD FK MODEL
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading FK model: {ILC_MODEL} …")
POLY_DEGREE = 3

class DataDrivenFK(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 3))
    def forward(self, x): return self.net(x)

class PressureConditionedFK(nn.Module):
    def __init__(self, hidden_act=256, hidden_p=1024):
        super().__init__()
        self.act_branch = nn.Sequential(
            nn.Linear(3, hidden_act), nn.Tanh(),
            nn.Linear(hidden_act, hidden_act), nn.Tanh(),
            nn.Linear(hidden_act, 3))
        self.pressure_branch = nn.Sequential(
            nn.Linear(1, hidden_p), nn.Tanh(),
            nn.Linear(hidden_p, hidden_p), nn.Tanh(),
            nn.Linear(hidden_p, hidden_p), nn.Tanh(),
            nn.Linear(hidden_p, hidden_p), nn.Tanh(),
            nn.Linear(hidden_p, 6))
    def forward(self, actuators, pressure):
        y_nom = self.act_branch(actuators)
        cond  = self.pressure_branch(pressure)
        twist  = torch.sigmoid(cond[:, 0]) * y_nom[:, 0] + cond[:, 1]
        height = torch.sigmoid(cond[:, 3]) * y_nom[:, 1] + cond[:, 2]
        volume = torch.sigmoid(cond[:, 5]) * y_nom[:, 2] + cond[:, 4]
        return torch.stack([twist, height, volume], dim=1)

if ILC_MODEL in ('DataDriven', 'PINN'):
    if ILC_MODEL == 'DataDriven':
        model = DataDrivenFK()
        model.load_state_dict(torch.load(SAVE_DIR / 'dd_model.pth', weights_only=True))
    else:
        model = PressureConditionedFK()
        model.load_state_dict(torch.load(SAVE_DIR / 'pinn_model.pth', weights_only=True))
    model.eval()
    model.requires_grad_(True)

    def fk_predict_norm(act_n_batch, p_n_batch):
        """(N,3) normalised act, (N,) normalised pressure → (N,3) normalised output."""
        act_t = torch.tensor(act_n_batch, dtype=torch.float32)
        p_t   = torch.tensor(p_n_batch,   dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            if ILC_MODEL == 'PINN':
                return model(act_t, p_t).numpy()
            else:
                x = torch.cat([act_t, p_t], dim=1)
                return model(x).numpy()

    def jacobian_at(act_n, p_n):
        """3×3 Jacobian ∂FK_norm/∂act_norm via autograd at one point."""
        act_t = torch.tensor(act_n, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
        p_t   = torch.tensor([[p_n]], dtype=torch.float32)
        if ILC_MODEL == 'PINN':
            y = model(act_t, p_t)
        else:
            y = model(torch.cat([act_t, p_t], dim=1))
        J = np.zeros((3, 3))
        for oi in range(3):
            if act_t.grad is not None: act_t.grad.zero_()
            y[0, oi].backward(retain_graph=True)
            J[oi, :] = act_t.grad[0].numpy()
        return J

else:
    # SINDy / SINDy2
    with open(SAVE_DIR / 'sindy_data.pkl',  'rb') as f: sindy_data  = pickle.load(f)
    with open(SAVE_DIR / 'sindy2_data.pkl', 'rb') as f: sindy2_data = pickle.load(f)
    sindy_results = sindy_data['results']
    sindy_sc      = sindy_data['sc']
    poly_lib      = sindy_data['poly_lib']
    s2_stage1     = sindy2_data['stage1']
    s2_stage2     = sindy2_data['stage2']
    s2_sc_in1     = sindy2_data['sc_in1']
    poly_lib1     = sindy2_data['poly_lib1']
    s2_sc_p       = sindy2_data['sc_p']

    def _sindy2_stage1(act_phys):
        x_s = s2_sc_in1.transform(np.asarray(act_phys).reshape(-1, 3))
        theta = np.asarray(poly_lib1.transform(x_s))
        outs = []
        for col in OUTPUT_NAMES:
            r = s2_stage1[col]
            outs.append(r['sy'].inverse_transform((theta @ r['coef']).reshape(-1,1)).ravel())
        return np.stack(outs, axis=1)

    def _build_mult_basis(base_col, p_s):
        scale_cols  = [base_col * (p_s ** k) for k in range(POLY_DEGREE + 1)]
        offset_cols = [(p_s ** k)             for k in range(POLY_DEGREE + 1)]
        return np.column_stack(scale_cols + offset_cols)

    def _sindy_predict_phys(x_physical):
        x_s   = sindy_sc.transform(np.asarray(x_physical).reshape(-1, 4))
        theta = np.asarray(poly_lib.transform(x_s))
        outs  = []
        for col in OUTPUT_NAMES:
            r = sindy_results[col]
            outs.append(r['sy'].inverse_transform((theta @ r['coef']).reshape(-1,1)).ravel())
        return np.stack(outs, axis=1)

    def _sindy2_predict_phys(x_physical):
        x_physical = np.asarray(x_physical).reshape(-1, 4)
        act_phys   = x_physical[:, :3]
        p_raw      = x_physical[:, 3]
        base       = _sindy2_stage1(act_phys)
        p_s        = s2_sc_p.transform(p_raw.reshape(-1,1)).ravel()
        outs = []
        for i, col in enumerate(OUTPUT_NAMES):
            r     = s2_stage2[col]
            Theta = _build_mult_basis(base[:, i], p_s)
            Theta_sc = r['col_sc'].transform(Theta)
            outs.append(r['sy'].inverse_transform((Theta_sc @ r['coef']).reshape(-1,1)).ravel())
        return np.stack(outs, axis=1)

    _pred_fn = _sindy_predict_phys if ILC_MODEL == 'SINDy' else _sindy2_predict_phys
    FD_EPS   = 1e-4

    def fk_predict_norm(act_n_batch, p_n_batch):
        act_phys = denorm_act(act_n_batch)
        p_phys   = p_n_batch * x_den[3] + x_min[3]
        x_phys   = np.hstack([act_phys, p_phys.reshape(-1,1)])
        return norm_y(_pred_fn(x_phys))

    def jacobian_at(act_n, p_n):
        act_phys = denorm_act(act_n)
        p_phys   = p_n * x_den[3] + x_min[3]
        def fwd(a_n):
            a_p  = denorm_act(a_n)
            x_p  = np.append(a_p, p_phys).reshape(1,-1)
            return norm_y(_pred_fn(x_p)[0])
        f0 = fwd(act_n)
        J  = np.zeros((3, 3))
        for j in range(3):
            ap = act_n.copy(); ap[j] += FD_EPS
            J[:, j] = (fwd(ap) - f0) / FD_EPS
        return J

print("  FK model ready.")

# ══════════════════════════════════════════════════════════════════════════════
# IK SOLVER — optimise actuators point-by-point via SINDy FK
# ══════════════════════════════════════════════════════════════════════════════

# Normalised actuator bounds for the optimiser
_lb_n = norm_act(ACT_MIN)
_ub_n = norm_act(ACT_MAX)
_bounds_n = list(zip(_lb_n, _ub_n))

b_q, a_q = butter(Q_ORDER, Q_CUTOFF, btype='low')   # shared Q-filter coefficients

def to_pvt_csv(epi, trans, endo, out_path):
    """Convert a single-cycle actuator array to a Zaber PVT CSV with ramps and tiling."""
    n = len(epi)
    dt = PVT_SECONDS_PER_CYCLE / n
    n_in  = max(2, round(PVT_RAMP_IN_S  / dt))
    n_out = max(2, round(PVT_RAMP_OUT_S / dt))

    def cramp(a, b, k):
        t = np.linspace(0, 1, k)
        return a + (b - a) * 0.5 * (1 - np.cos(np.pi * t))

    epi_c, trans_c, endo_c = (np.tile(x, PVT_N_CYCLES) for x in (epi, trans, endo))

    epi_f   = np.concatenate([cramp(200.0, epi_c[0],   n_in), epi_c,   cramp(epi_c[-1],   200.0, n_out)])
    trans_f = np.concatenate([cramp(202.0, trans_c[0], n_in), trans_c, cramp(trans_c[-1], 202.0, n_out)])
    endo_f  = np.concatenate([cramp(200.0, endo_c[0],  n_in), endo_c,  cramp(endo_c[-1],  200.0, n_out)])

    for sig in (epi_f, trans_f, endo_f):
        sig[:] = np.clip(sig, None, 248.0)

    N   = len(epi_f)
    t   = np.linspace(0, dt * (N - 1), N)
    pd.DataFrame({
        'Time (s)':           t,
        'Position U1 (mm)':   epi_f,
        'Velocity U1 (mm/s)': np.gradient(epi_f,   dt),
        'Position U2 (mm)':   trans_f,
        'Velocity U2 (mm/s)': np.gradient(trans_f, dt),
        'Position U3 (mm)':   endo_f,
        'Velocity U3 (mm/s)': np.gradient(endo_f,  dt),
    }).to_csv(out_path, index=False, float_format='%.2f')
    return dt, n_in, n_out

def ik_solve(traj_phys, traj_p):
    """
    Solve IK for every time point in traj_phys (T,3) at pressures traj_p (T,).
    Returns act_phys_raw (T,3) — unsmoothed optimal actuators in mm.
    Warm-starts each point from the previous solution.
    """
    T = len(traj_phys)
    act_phys_raw = np.zeros((T, 3))
    u_n = norm_act(ACT_HOME)   # initial guess — actuator home

    for i in range(T):
        p_n = (traj_p[i] - x_min[3]) / x_den[3]
        y_d = traj_phys[i]   # (3,) desired in physical units

        def cost_and_grad(u_n_i):
            # FK prediction in physical space
            y_n   = fk_predict_norm(u_n_i.reshape(1, 3), np.array([p_n]))[0]
            y_p   = denorm_y(y_n)
            e     = (y_p - y_d) * GEOM_WEIGHTS        # weighted error
            cost  = float(np.dot(e, e))
            # Gradient:  dcost/du_n = 2 * J_n^T @ (y_den * GEOM_WEIGHTS^2 * (y_p - y_d))
            J_n   = jacobian_at(u_n_i, p_n)            # (3,3) dyn/dun
            grad  = 2.0 * J_n.T @ (y_den * GEOM_WEIGHTS**2 * (y_p - y_d))
            return cost, grad

        res = minimize(cost_and_grad, u_n, jac=True, method='L-BFGS-B',
                       bounds=_bounds_n,
                       options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-9})
        act_phys_raw[i] = np.clip(denorm_act(res.x), ACT_MIN, ACT_MAX)
        u_n = res.x   # warm-start next point

    return act_phys_raw


# ══════════════════════════════════════════════════════════════════════════════
# CASE LOOP
# ══════════════════════════════════════════════════════════════════════════════
for case_name, out_fname in _CASE_FILENAMES.items():
    print(f"\n{'═'*68}")
    print(f"  Case: {case_name.upper()}")
    print(f"{'═'*68}")

    # ── Load desired trajectory ───────────────────────────────────────────────
    eng_path = _SIM_CASE_FILES[case_name]
    assert eng_path.exists(), f"Trajectory not found: {eng_path}"
    eng        = pd.read_csv(eng_path)
    traj_time  = eng['time'].values
    traj_p     = eng['pressure'].values
    traj_twist = eng['twist'].values
    traj_h     = eng['height'].values + HEIGHT_OFFSET
    traj_v     = eng['volume'].values + VOLUME_OFFSET
    traj_phys  = np.stack([traj_twist, traj_h, traj_v], axis=1)   # (T, 3)
    traj_phase = (traj_time - traj_time[0]) / (traj_time[-1] - traj_time[0])
    n_traj     = len(traj_time)
    print(f"  Desired traj : {n_traj} pts  "
          f"pressure=[{traj_p.min():.0f}, {traj_p.max():.0f}] mmHg")

    # ── IK solve ──────────────────────────────────────────────────────────────
    print(f"  Solving IK  ({n_traj} pts) …", end='', flush=True)
    act_phys_raw = ik_solve(traj_phys, traj_p)
    print("  done.")

    # ── Q-filter for smoothness ───────────────────────────────────────────────
    act_n_smooth = filtfilt(b_q, a_q, norm_act(act_phys_raw), axis=0)
    act_phys_out = np.clip(denorm_act(act_n_smooth),
                           ACT_MIN.reshape(1, -1), ACT_MAX.reshape(1, -1))

    # ── FK verification (re-predict with smoothed actuators) ──────────────────
    pressure_n   = (traj_p - x_min[3]) / x_den[3]
    y_phys_pred  = denorm_y(fk_predict_norm(norm_act(act_phys_out), pressure_n))
    e_pred       = traj_phys - y_phys_pred

    print(f"  FK-predicted error (after Q-filter smoothing):")
    for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
        print(f"    {col:<14}: RMSE = {np.sqrt(np.mean(e_pred[:, ci]**2)):.4f} {unit}")

    # ── Save IK single-cycle CSV ──────────────────────────────────────────────
    out_path = OUT_DIR / out_fname
    pd.DataFrame({
        'phase':          traj_phase,
        'time_in_cycle':  traj_time,
        'pressure_mmhg':  traj_p,
        'epi':            act_phys_out[:, 0],
        'trans':          act_phys_out[:, 1],
        'endo':           act_phys_out[:, 2],
        'pred_twist_deg': y_phys_pred[:, 0],
        'pred_height_mm': y_phys_pred[:, 1],
        'pred_volume_mL': y_phys_pred[:, 2],
    }).to_csv(out_path, index=False)
    print(f"\n  Saved IK CSV → {out_fname}")

    # ── Convert to PVT (Zaber-readable) ──────────────────────────────────────
    pvt_dir  = ILC_TRAJ / PVT_SESSION_DIR / case_name
    pvt_dir.mkdir(parents=True, exist_ok=True)
    pvt_fname = f'PVT_ILC_itr0_{int(PVT_SECONDS_PER_CYCLE)}s_{PVT_N_CYCLES}cycles.csv'
    pvt_path  = pvt_dir / pvt_fname
    dt_pvt, n_in, n_out = to_pvt_csv(
        act_phys_out[:, 0], act_phys_out[:, 1], act_phys_out[:, 2], pvt_path)
    print(f"  PVT CSV       → ILC_traj/{PVT_SESSION_DIR}/{case_name}/{pvt_fname}")
    print(f"                  dt={dt_pvt:.3f}s  ramp-in={n_in}pts  ramp-out={n_out}pts  "
          f"total={(n_in + n_traj*PVT_N_CYCLES + n_out)} pts")

    # ── Per-case figure (shown immediately, then saved) ───────────────────────
    fig, axes = plt.subplots(4, 2, figsize=(16, 15))

    for ci, (col, unit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
        ax = axes[ci, 0]
        ax.plot(traj_phase, traj_phys[:, ci],  'k-',  lw=2.5, label='Desired')
        ax.plot(traj_phase, y_phys_pred[:, ci], 'r-', lw=1.8, label='IK prediction (FK model)')
        rmse = np.sqrt(np.mean(e_pred[:, ci]**2))
        ax.set_title(f'RMSE = {rmse:.3f} {unit}', fontsize=9)
        ax.set_ylabel(f'{col} ({unit})', fontsize=10)
        ax.set_xlabel('Cycle phase', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax2 = axes[ci, 1]
        motor_name = ['Epi', 'Trans', 'Endo'][ci]
        ax2.plot(traj_phase, act_phys_raw[:, ci], 'b--', lw=1.2, alpha=0.5, label='IK raw')
        ax2.plot(traj_phase, act_phys_out[:, ci], 'r-',  lw=1.8, label='IK + Q-filter')
        ax2.axhline(ACT_MIN[ci], color='grey', lw=1, ls=':')
        ax2.axhline(ACT_MAX[ci], color='grey', lw=1, ls=':', label='Bounds')
        ax2.set_ylabel(f'{motor_name} position (mm)', fontsize=10)
        ax2.set_xlabel('Cycle phase', fontsize=9)
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    ax_p = axes[3, 0]
    ax_p.plot(traj_phase, traj_p, 'k-', lw=2, label='Desired pressure')
    ax_p.set_ylabel('Pressure (mmHg)', fontsize=10)
    ax_p.set_xlabel('Cycle phase', fontsize=9)
    ax_p.set_title('Pressure profile (FK input at each point)', fontsize=10, color='grey')
    ax_p.legend(fontsize=8); ax_p.grid(True, alpha=0.3)
    axes[3, 1].axis('off')

    axes[0, 0].set_title(f'IK solution — desired vs FK prediction — {case_name}', fontsize=11)
    axes[0, 1].set_title('Actuator signals (IK solution)', fontsize=11)
    plt.suptitle(f'{case_name.upper()} — SINDy IK  (Q_cutoff={Q_CUTOFF}, model={ILC_MODEL})',
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(BASE, f'mpc_ilc_{case_name}.png'), dpi=150, bbox_inches='tight')
    plt.show()   # blocks until user closes — proceed to next case after
    print(f"  Figure → mpc_ilc_{case_name}.png")

print(f"\n{'═'*68}")
print(f"  DONE — 3 itr 0 CSVs written to:")
print(f"  {OUT_DIR}")
print(f"{'═'*68}")
