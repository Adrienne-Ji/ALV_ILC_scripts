"""
fix_diastolic_trajectory.py
────────────────────────────
Rescale diastolic dysfunction volume to SV=50 mL.

Keep EDV (max volume) fixed.
Scale the deviation below EDV so the new ESV = EDV - 50.
  v_new = v_max - (v_max - v) * (sv_target / sv_current)

Waveform shape (time profile) is preserved — only amplitude changes.
Backs up original before overwriting.
"""

import pandas as pd
import numpy as np
import pathlib
import shutil

TRAJ_DIR = pathlib.Path(
    r'C:\Users\z5448472\OneDrive - UNSW\Desktop\Project2RVsimulator'
    r'\ArtificialHeart\ILCFiles\Engineered_trajs')

src = TRAJ_DIR / 'engineered_data_diastolic_dysfunction.csv'
bak = TRAJ_DIR / 'engineered_data_diastolic_dysfunction_original.csv'

SV_TARGET = 50.0   # mL

df = pd.read_csv(src)
v  = df['volume'].values

v_max   = v.max()
v_min   = v.min()
sv_old  = v_max - v_min
scale   = SV_TARGET / sv_old

v_new = v_max - (v_max - v) * scale

print(f"Original:  EDV={v_max:.2f}  ESV={v_min:.2f}  SV={sv_old:.2f} mL")
print(f"Rescaled:  EDV={v_new.max():.2f}  ESV={v_new.min():.2f}  SV={v_new.max()-v_new.min():.2f} mL")
print(f"EF:  original={sv_old/v_max*100:.1f}%  →  new={SV_TARGET/v_max*100:.1f}%")

# Backup then overwrite
shutil.copy(src, bak)
print(f"\nBackup saved → {bak.name}")

df['volume'] = v_new
df.to_csv(src, index=False)
print(f"Updated   → {src.name}")
