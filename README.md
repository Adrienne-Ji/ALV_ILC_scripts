# ILC Pipeline — RV Simulator Artificial Heart

Iterative Learning Control (ILC) pipeline for correcting the actuator trajectory
so the artificial heart tracks the desired deformation and pressure profile.

---

## Big Picture

The heart runs N cycles. After each run, the data is processed and the actuator
trajectory is corrected for the next run. This repeats until the tracking error
is small enough.

There is now **one correction script** — `ilcCorrection.py` — that corrects
geometry (twist, height, volume) **and** pressure together in a single pass.
There is no separate "Phase 1 / Phase 2" split anymore.

**The geometry Jacobian is a confidence-weighted blend, not a hard switch:**
```
J_geom = w · J_empirical + (1 − w) · J_static
```
A hard cutover (100% static FK below a threshold, then suddenly 100%
regression) leaves the regression with no stable floor of direction exactly
when it's most likely to be wrong — thin or collinear session data can give a
fit with fine R² but unstable individual epi/trans/endo coefficients,
producing a confidently-wrong Jacobian that makes ILC **diverge** instead of
converge. Blending keeps a sensible baseline direction from the static FK
while gradually trusting the data-driven estimate as it earns it.

`w = w_n · w_cond`:
- `w_n` ramps `0→1` linearly as session history goes from `MIN_HISTORY_ITERS` (5) to `FULL_TRUST_ITERS` (10)
- `w_cond` ramps `1→0` (log scale) as the design matrix `[epi,trans,endo,1]`'s condition number goes from `GOOD_COND_NUMBER` (1e2) to `MAX_COND_NUMBER` (1e4) — catches actuator collinearity directly, since iteration count alone doesn't guarantee independent actuator combinations (the three actuators likely move in a coordinated pattern each cycle)

Pressure correction uses the same `w_n` ramp (`λ_eff = LAMBDA_P · w_n`) — also
smooth, not a hard on/off cutover, since there's no static-pressure-model
equivalent to blend against. (A too-low threshold here previously caused
`R²_P ≈ 1.00` — a red flag for overfitting, not a good sign — on a single
phase-cycle of correlated data.)

**History never crosses sessions, but it does pool across cases within the
same session.** The soft robotic device's behaviour drifts day to day, so
reusing an old day's Jacobian would be misleading — each new date starts the
empirical Jacobian from scratch (static FK warm start on iteration 1).
*Within* one day, though, actuator→deformation sensitivity is a property of
the physical device, not of which target trajectory you're chasing — so if
you run healthy + diastolic + systolic cases the same day under one
`SESSION_DIR`, the Jacobian fit pools data from **all** of them (more
actuator-space diversity → better-conditioned fit). What stays case-specific
is the **tracking error** — that's always computed against `ENG_CSV`, the
desired trajectory for whichever case is actually running that iteration.
See `SESSION_DIR` below.

`ilcMotionCorrection.py`, `ilcPressureCorrection.py`, and the SINDy-C dynamic
model in `trainDynamicFK.py` are **superseded** and no longer part of the
active pipeline (SINDy-C was abandoned — ILC iterations don't produce enough
trajectory diversity for it to identify reliably). They're kept in the repo
for reference only.

---

## File Map

```
DataProcessingMatlabCodes/Optical Tracking/<date>/
├── markerProcessing.py              ← Step 1: process raw recording

PythonCodes/
├── MPC_depthCam/
│   ├── ilcCorrection.py             ← Step 3: the ILC correction (geometry + pressure, single pass)
│   ├── run_ilc_pipeline.py         ← runs ilcCorrection.py + PVTwrite.py in one go
│   ├── plotILCConvergence.py       ← visualises tracking + RMSE across all iterations in Exp_data
│   ├── pressure_fk_comparison.py   ← FK model benchmarking (not part of ILC loop)
│   └── saved_models/
│       ├── norm_constants.npz      ← normalisation from static FK training
│       └── ilc_history/
│           ├── iter001_Xraw.csv    ← state history per iteration (auto-saved)
│           └── ...
└── PVTwrite.py                      ← Step 4: tiles 1-cycle → N-cycle Zaber file

ILCFiles/
├── Exp_data/<SESSION_DIR>/<CASE>/<itr N | p_itr N>/ILCReadyData.csv  ← history
│         SESSION_DIR = date ('6_28'). ilcCorrection.py reads everything
│         recursively under Exp_data/<SESSION_DIR>/ (pooling all CASE
│         subfolders for the Jacobian fit) — never across different dates.
├── Engineered_trajs/                  ← the 3 literature-anchored desired trajectories —
│   │                                     SOLE source of desired trajectories for the active
│   │                                     pipeline (ilcCorrection.py, plotILCConvergence.py).
│   │                                     PythonCodes/engineered_data_withP.csv is a DIFFERENT,
│   │                                     older file (pressure differs by up to ~8.6 mmHg) —
│   │                                     no longer referenced by the active ILC pipeline.
│   ├── engineered_data_healthy.csv
│   ├── engineered_data_diastolic_dysfunction.csv
│   └── engineered_data_systolic_dysfunction.csv
└── ILC_traj/<SESSION_DIR>/<CASE>/
    ├── ILC_<tag>.csv                 ← single-cycle corrected actuators (git tracked)
    ├── ILC_<tag>_comparison.png      ← corrected vs previous actuator signal
    └── PVT_ILC_<tag>_20s_*cycles.csv ← Zaber-ready file (git tracked)

sharedCSVs/
├── ILCReadyData.csv                 ← input to ILC (from markerProcessing.py, current iteration only)
└── ilc_corrected_actuators.csv      ← output of ILC — transient handoff file, overwritten each iteration
```

---

## Step-by-Step Workflow

### Step 0 — Start of a new session (new date, or new trajectory case)

Decide today's `SESSION_DIR` (just the date, e.g. `'6_28'`) and which `CASE`
you're running (`'healthy'` / `'diastolic'` / `'systolic'`). Set **both** in
`run_ilc_pipeline.py` — it's the source of truth and passes them down to
`ilcCorrection.py` and `PVTwrite.py` automatically:
```python
SESSION_DIR = '6_28'
CASE        = 'healthy'
```
If you ever run `ilcCorrection.py` standalone (without the wrapper), it falls
back to its own hardcoded `SESSION_DIR`/`SIM_CASE` — keep those in sync, or
just always run through `run_ilc_pipeline.py`.

Every iteration this session writes/reads history under
`ILCFiles/Exp_data/<SESSION_DIR>/<CASE>/` — nothing from previous **dates**
is ever read, since the device's behaviour drifts day to day. (Other cases
the *same* day are still pooled for the Jacobian fit — see below.)

---

### Step 1 — Record data on the rig

Run N cycles using the current PVT file on the Zaber controller.
Record simultaneously:
- **OptiTrack** → `MDC_data_*.csv`
- **ALV motor log** → `ALV_10Hz_Log_*.csv`
- **Pressure** → `LVP.mat` + `LVP_endtime.txt`

Create a new iteration folder for this run, e.g.
`ILCFiles/Exp_data/<SESSION_DIR>/<CASE>/p_itr 7/`, and put the recording in there.

---

### Step 2 — Process the recording

In `markerProcessing.py`, set:
```python
DATA_DIR = r"...\ILCFiles\Exp_data\<SESSION_DIR>\<CASE>\p_itr 7"
```

Run it. It writes `ILCReadyData.csv` to **two places**:
- `sharedCSVs/ILCReadyData.csv` — the current-iteration input to `ilcCorrection.py`
- a copy inside `DATA_DIR` — this is what accumulates as this session's history for the empirical Jacobian and pressure regression

Check `alignment_check.png` to confirm OptiTrack / ALV / pressure aligned correctly.

---

### Step 3 — Run the ILC correction

In `run_ilc_pipeline.py`, set (alongside `SESSION_DIR`/`CASE` from Step 0):
```python
OUTPUT_TAG = 'P_iter7'   # increment each run
```

Run:
```
python MPC_depthCam/run_ilc_pipeline.py
```

This will:
1. Run `ilcCorrection.py` (with `SESSION_DIR`/`CASE` passed through automatically):
   - Fit the empirical geometry Jacobian, pooling every iteration file under `Exp_data/<SESSION_DIR>/` (all cases that day)
   - Fit the phase-varying pressure Jacobian (Fourier regression) from the same pooled history
   - Compute the weighted ILC update, Q-filter it, clip to actuator limits
   - Print PASS/FAIL per output (twist, height, volume, pressure) against threshold — tracking error is computed against `CASE`'s desired trajectory specifically
2. Save corrected actuators → `sharedCSVs/ilc_corrected_actuators.csv`
3. Run `PVTwrite.py` → `ILCFiles/ILC_traj/<SESSION_DIR>/<CASE>/PVT_ILC_<tag>_20s_*cycles.csv`

**Load that PVT file onto the Zaber controller and run the next trial.**

Repeat Steps 1 → 2 → 3 until twist, height, volume, and pressure are all `[PASS]`.

---

### Simulating the 3 trajectory cases (healthy / diastolic / systolic dysfunction)

`CASE` in `run_ilc_pipeline.py` (which sets `SIM_CASE` in `ilcCorrection.py`
via env var) picks the desired trajectory the ILC tracking error is computed
against — all 3 cases resolve to a file in `ILCFiles/Engineered_trajs/`. The
console prints which file resolved at startup — check that matches what you
intend to run.

`SESSION_DIR` always stays just the date. Running multiple cases the same day
means setting `CASE` differently across runs while keeping `SESSION_DIR` the
same — each case gets its own subfolder under both `Exp_data/<SESSION_DIR>/`
and `ILC_traj/<SESSION_DIR>/`, but the **geometry Jacobian fit pools across
all of them automatically** (it's a recursive search) since actuator→
deformation sensitivity is a device property, not case-specific. Only the
tracking error (via `CASE`) stays specific to whichever scenario you're
actively correcting.

---

### Checking convergence across all iterations

In `plotILCConvergence.py`, set:
```python
CASE = 'healthy'    # or 'diastolic', 'systolic' — must match what you ran
```
Then run:
```
python MPC_depthCam/plotILCConvergence.py
```

Unlike `ilcCorrection.py` (which scopes to one `SESSION_DIR`), this is a
retrospective view across **all dates** — but it still filters to iteration
folders matching `CASE`, so a diastolic run's RMSE never gets silently
compared against the healthy desired trajectory (or pooled with healthy
iterations in the plot). Folders with no case label at all (pre-existing data
from before disease cases were added) are treated as `'healthy'` only.

---

## Key Settings — Quick Reference

### `markerProcessing.py`
| Setting | What it does |
|---|---|
| `DATA_DIR` | Folder containing this iteration's raw recording — also where `ILCReadyData.csv` is copied for history |

### `run_ilc_pipeline.py` — set these first, every session
| Setting | What it does |
|---|---|
| `SESSION_DIR` | **The date folder**, e.g. `'6_28'`. Passed down to `ilcCorrection.py`/`PVTwrite.py` via env vars — single source of truth. All inputs/outputs nest under this. |
| `CASE` | **Which trajectory case**: `'healthy'`, `'diastolic'`, or `'systolic'`. Also passed down via env var — determines both the output subfolder and (in `ilcCorrection.py`) which desired trajectory the tracking error uses. |
| `OUTPUT_TAG` | Label on the output CSV and figure (e.g. `'P_iter7'`). Also passed down to `ilcCorrection.py`, which parses the trailing number out of it (`'P_iter7'` → `7`) to use as the iteration number shown in its console output and figure title — keeping that number in sync with the actual file you're loading onto the Zaber controller, instead of an ever-growing global count across every session ever run. |

### `ilcCorrection.py`
| Setting | What it does |
|---|---|
| `SESSION_DIR` | Scopes history reads to `Exp_data/<SESSION_DIR>/` (pooled across all `CASE` subfolders within it). Overridden automatically by env var `ILC_SESSION_DIR` when run via `run_ilc_pipeline.py` — only its hardcoded default matters for standalone runs. |
| `SIM_CASE` | Picks the desired trajectory for the tracking error — all 3 cases resolve to a file in `ILCFiles/Engineered_trajs/` (`engineered_data_healthy.csv`, `..._diastolic_dysfunction.csv`, `..._systolic_dysfunction.csv`). Overridden automatically by env var `ILC_CASE` when run via `run_ilc_pipeline.py`. |
| `ILC_ALPHA` | Learning gain (0–1). Default `0.55`. Lower if oscillating — but the confidence-blended Jacobian and `MAX_DELTA_U_MM` cap now guard against bad-Jacobian instability directly, so this doesn't need to be as conservative as before. |
| `Q_CUTOFF` | Low-pass filter cutoff for smoothing corrections (0–1 normalised). Default `0.35`. Higher passes more high-frequency correction through (better steady-state accuracy, more noise sensitivity). |
| `USE_EMPIRICAL_JACOBIAN` | `True` (default) — blend `J_geom` from iteration history with the static FK model, weighted by confidence (see Big Picture above), instead of using static FK alone. |
| `REGRESSION_FLOOR` | Session history count at which the empirical fit is even attempted (computed, conditioning checked) — separate from whether it's *trusted*. Default `1`. |
| `MIN_HISTORY_ITERS` | Session history count where the empirical fit's blend weight starts ramping up from 0. Default `1` — weight is exactly 0 at the very first iteration, then increases from iteration 2 onward. |
| `FULL_TRUST_ITERS` | Session history count where the empirical fit's blend weight reaches 1 (assuming good conditioning). Default `10`. |
| `GOOD_COND_NUMBER` | Condition number below which the empirical fit gets no conditioning penalty. Default `1e2`. |
| `MAX_COND_NUMBER` | Condition number at/above which the empirical fit is fully rejected (blend weight forced to 0) — a direct check for actuator collinearity that iteration count alone doesn't catch. Default `1e4`. |
| `GEOM_WEIGHTS` | `[twist, height, volume]` weighting in the ILC update. Default `[1.0, 0.5, 1.0]` — de-emphasise height once it's converged, push more correction toward twist/volume. |
| `MAX_DELTA_U_MM` | Hard cap (mm) on the per-iteration actuator correction, regardless of Jacobian source. Default `8.0`. Safety net against an ill-conditioned Jacobian producing an oversized correction via `pinv()`. Console warns how many points got clamped. |
| `ACT_MIN` / `ACT_MAX` | Physical actuator limits in mm — must match the rig |
| `HEIGHT_OFFSET` / `VOLUME_OFFSET` | Constants added to measured height/volume to match desired trajectory coordinates |
| `LAMBDA_P` | Pressure weight, ramped smoothly (`λ_eff = LAMBDA_P · w_n`, same ramp as geometry's iteration-count confidence) from `MIN_HISTORY_ITERS` to `FULL_TRUST_ITERS`. Default `0.3` (dropped from 0.5 — early R²_P≈1.00 signalled overfitting on thin data) |
| `N_FOURIER` | Harmonics in the phase-varying pressure Jacobian regression. Default `1` (dropped from 2 to halve parameter count and reduce overfitting risk while session data is thin) |
| `GEOMETRY_RMSE_THRESHOLD` / `PRESSURE_RMSE_THRESHOLD` | PASS/FAIL thresholds per output |

---

## Output Files Per Iteration

| File | Location | Contents |
|---|---|---|
| `iter{k:03d}_Xraw.csv` | `saved_models/ilc_history/` | Full state log: phase, twist, height, volume, pressure, epi, trans, endo |
| `ilc_corrected_actuators.csv` | `sharedCSVs/` | Corrected 1-cycle actuators — transient handoff file, overwritten each iteration |
| `PVT_ILC_{OUTPUT_TAG}_20s_*cycles.csv` | `ILCFiles/ILC_traj/<SESSION_DIR>/<CASE>/` | Zaber-ready file — **git tracked**, pushed to experiment laptop |
| `ILC_{OUTPUT_TAG}.csv` | `ILCFiles/ILC_traj/<SESSION_DIR>/<CASE>/` | Single-cycle corrected actuators — **git tracked** |
| `ILC_{OUTPUT_TAG}_comparison.png` | `ILCFiles/ILC_traj/<SESSION_DIR>/<CASE>/` | Corrected vs previous actuator signal plot |

---

## Troubleshooting

**Geometry RMSE not decreasing**
- Check `alignment_check.png` — data misalignment will corrupt the error signal
- Check `HEIGHT_OFFSET` and `VOLUME_OFFSET` — if actual and desired are offset by a constant, these need adjusting
- Reduce `ILC_ALPHA` if oscillating (try 0.4–0.5)
- Check the console R² printed for the empirical Jacobian per output — a low R² for one output (e.g. volume, twist) means the history data doesn't cleanly support that direction; more iterations with varied actuator changes will help

**One output (e.g. volume/twist) lags while another (e.g. height) converges well**
- Lower that output's `GEOM_WEIGHTS` entry for the converged one, raise it for the lagging ones, so the pseudoinverse prioritises correcting what's still off
- This is a coupling/sensitivity issue, not necessarily a modelling bug — some outputs are just less actuated by the available degrees of freedom

**Pressure RMSE not decreasing**
- Check the console R² for the Fourier regression — low R² means the actuator→pressure correlation isn't being captured cleanly from the available history
- Increase `LAMBDA_P` (try 0.6–0.8) — but check this isn't fighting the geometry correction (temporarily set `LAMBDA_P = 0` to confirm geometry converges better without it)
- Remember volume and pressure are physically linked (Frank-Starling) — if volume is under-tracking, pressure will lag for that reason alone, independent of the pressure Jacobian quality

**History not found / pressure correction disabled**
- `ilcCorrection.py` only searches inside `Exp_data/<SESSION_DIR>/` — check `SESSION_DIR` matches the folder you've actually been writing iterations into this session
- Console prints `"Session history scope: ..."` at startup — confirm that path is right, and `"Only N iteration file(s) found — need ≥REGRESSION_FLOOR — pressure correction disabled"` if it found fewer than 2 inside it (pressure ramps smoothly from there, not a hard cutover — see Big Picture above)
- Check `DATA_DIR` in `markerProcessing.py` pointed inside the same `SESSION_DIR`, not a different/old date folder
- This is expected (not a bug) on the first iteration of a brand-new session (0 or 1 history files) — geometry-only with the static FK warm start is correct there

**Trajectory is diverging — each iteration looks worse than the last, further from a previously-working waveform**
- Check the printed `"Geometry Jacobian blend weight: ..."` line — if it's already high (`>0.7`, "mostly/fully empirical") with only a handful of session iterations, the regression may be getting trusted too quickly; lower it by raising `FULL_TRUST_ITERS`
- Check the printed R² and epi/trans/endo coefficients for the empirical Jacobian — a fit can pass the conditioning check while still being a poor model if the session's actuator-space coverage is narrow; this is the same "trajectory diversity" problem that made SINDy-C non-viable, recurring in a simpler linear form
- Raising `MIN_HISTORY_ITERS`/`FULL_TRUST_ITERS` (e.g. to 7/14) shifts more weight onto the static FK for longer — bounded and won't actively diverge the way a poorly-conditioned regression-derived Jacobian can, even partially blended in
- Don't conflate this with the earlier "violent correction"/oscillation issues (already fixed) — this is about the correction's *direction* being subtly wrong over many iterations, not the magnitude or per-point smoothness of any single iteration

**Actuator corrections being clipped every iteration**
- Check `ACT_MIN`/`ACT_MAX` — are they set to the correct physical limits?
- The correction may be too aggressive — reduce `ILC_ALPHA`

**Corrected trajectory looks wildly overcompensated (huge swing vs previous signal)**
- Check the console for `"WARNING: N actuator/phase corrections exceeded ±MAX_DELTA_U_MM mm and were clamped"` — this means the Jacobian's gradient scale produced an oversized correction via `pinv()`, most likely on the static FK cold-start iteration (first iteration of a session, before there's enough history for the empirical Jacobian)
- This is expected and self-limiting via `MAX_DELTA_U_MM` (default 8mm) — it does not indicate the correction direction is wrong, only that its raw magnitude was too large before clamping
- Once a second iteration exists this session, the empirical Jacobian takes over and is generally much better-scaled since it's fit directly from real actuator/output deltas
- If it's still happening on the empirical Jacobian (not the static FK fallback), check the printed R² — a near-singular fit (very low R² or a Jacobian column close to zero) will still produce oversized `pinv()` corrections

**Corrected actuator signal oscillates / has multiple extra bumps not in the previous signal (especially on the static FK cold-start iteration)**
- The static FK fallback Jacobian is evaluated **once**, at the trajectory's mean operating point, and held constant over the whole cycle — not per phase point. If you still see this, check the code wasn't reverted to evaluating `jacobian_at()` inside the per-phase loop, since pointwise finite-difference evaluation on the static model lets noise vary point to point, producing exactly this kind of multi-bump artifact
- This does not happen on the empirical Jacobian, since it's already a single constant matrix for the whole cycle by construction
