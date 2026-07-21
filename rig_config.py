"""
rig_config.py
──────────────────────────────────────────────────────────────────────────────
Single source of truth for rig-specific calibration constants.

Change these values here and every script that imports them (ilcCorrection.py,
plotILCConvergence.py, etc.) automatically uses the updated values — no
hunting through multiple files.  Values must NOT be baked into CSV files since
they are a property of the current rig setup, not the desired trajectory.
"""

# Offsets applied to the DESIRED trajectory to align it with the rig's
# natural coordinate reference.
#
# HEIGHT_OFFSET: the engineered height values are relative to some reference
#   position; this offset shifts the desired curve to match the rig's
#   physical zero.  If the measured height systematically sits above the
#   desired, increase this value.  Current: 75 mm (raised from 70 mm —
#   measured was ~5 mm above desired, causing epi to saturate at 200 mm floor).
#
# VOLUME_OFFSET: shifts the desired volume trajectory upward.  Set to +15 mL
#   to move the minimum volume above the FK model's achievable floor at 120 mmHg
#   (~68.5 mL).  Resulting range: healthy 75–135 mL (EF≈44%), systolic 99–135 mL
#   (EF≈27%).  All within physiological RV range (EDV 100–160, ESV 30–100 mL).
HEIGHT_OFFSET = 70.0   # mm — matched to 6_18 session
VOLUME_OFFSET = 0.0    # mL
