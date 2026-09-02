"""Tunables for the Microduck ground_pick task.

Direct port of the module-level constants from the old
``tasks/microduck_ground_pick_env_cfg.py``.
"""

# Symmetry -- disabled for v1.5: SYMMETRY_CFG's obs permutation is hardcoded
# for the old 51D obs layout and breaks on the new 61D obs (which includes
# the head_command/body_command padding). All v1.5 envs run with symmetry
# off until SYMMETRY_CFG gets rewritten for the new obs structure.
ENABLE_SYMMETRY = False

# -- Domain randomisation (matched to the velocity env) ---------------------
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = False  # off, like velocity
ENABLE_KD_RANDOMIZATION = False
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # scales BAM friction budget per-env
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # reflected rotor inertia (affects BAM)
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # applied at obs level (per-env rotation)
ENABLE_ENCODER_BIAS = True  # actor obs sees joint_pos + per-env bias
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False
ENABLE_NECK_OFFSET_RANDOMIZATION = False  # disabled -- head is used for the task

# -- Ranges (matched to the velocity env) ------------------------------------
COM_RANDOMIZATION_RANGE = 0.003  # +-3mm initial, ramped via curriculum
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # +-3mm initial, ramped via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
# Quasi-static reaching motion -> gentle pushes (+-0.3 knocked it over even
# standing straight).
VELOCITY_PUSH_RANGE = (-0.15, 0.15)
VELOCITY_PUSH_PLAY_INTERVAL_S = (2.0, 4.0)  # spaced out for judging the motion, not a stress test
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # match velocity (was 1.0)
ENCODER_BIAS_RANGE = (-0.015, 0.015)

# -- Task constants -----------------------------------------------------------
# Segmented phase profile (independent durations) instead of a sinusoidal
# weighting (which couples descent/hold/rise): descent and rise are SLOW,
# the low hold is SHORT, standing rest is long. At GP_PERIOD = 4s:
#   descent [0, DESCENT_END)        1.5s  STAND -> low transition
#   hold    [DESCENT_END, HOLD_END) 0.2s  brush the ground (short)
#   rise    [HOLD_END, RISE_END)    1.5s  low -> STAND transition
#   rest    [RISE_END, 1)           0.8s  standing
# NOTE: RISE_END=0.80 > the phi=0.7 cutoff in infer_policy.py -- the rise is
# only complete if the runtime slot plays through to phi~1.0 (the whole
# period). Verify the runtime's actual window if this changes.
# NOTE: --ground-pick-period at deployment must match GP_PERIOD (4.0).
GP_PERIOD = 4.0
DESCENT_END = 0.375
HOLD_END = 0.425
RISE_END = 0.80

LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
NECK_JOINTS = [5, 6, 7, 8]
