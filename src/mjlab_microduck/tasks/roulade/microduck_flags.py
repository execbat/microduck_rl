"""Tunables for the Microduck roulade (forward-roll) task.

Direct port of the module-level constants from the old
``tasks/microduck_roulade_env_cfg.py``. See that file's original docstring
history (kept verbatim on ``MicroduckRouladeEnvCfg``) for the full run-1 ->
run-2 design rationale.
"""

import math

# The roll is sagittal / left-right symmetric; the mirror loss directly
# fights the sideways-collapse failure seen in run 2. Enabled after
# migrating symmetry.py to the 61-dim layout (2026-08-13) -- roulade is the
# first env to use it. NOTE: this is the only microduck task with symmetry
# ON; every other v1.5+ task has it off pending a SYMMETRY_CFG rewrite.
ENABLE_SYMMETRY = True

# -- Domain randomisation (matched to standup/velocity for sim2real parity) --
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = False  # a push mid-roll is incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

# -- Ranges (matched to the standup env) -------------------------------------
COM_RANDOMIZATION_RANGE = 0.003  # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # unused (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# A CONTROLLED roll takes ~2s + rise ~1.5s + settle.
EPISODE_LENGTH_S = 5.0

# Empirically-measured standing trunk height (standup lesson: don't guess).
STAND_Z = 0.115

# -- Elan (run-up) hook -------------------------------------------------------
# (0, 0) = roll from a standstill. Widen to e.g. (0.0, 0.3) to train rolls
# entered with forward momentum -- standing spawns then get a random initial
# forward base velocity, approximating a hand-off from the walking policy
# without simulating the walk itself.
ROULADE_FORWARD_VEL_RANGE = (0.0, 0.0)

# -- Mid-roll spawn (reverse curriculum) --------------------------------------
# 90deg = balanced on the head, 180deg = on the back, 270deg = supine,
# ~340deg = seated leaning back, >260deg opens the landing gate. MAX widened
# to 340deg so the second half of the roll (supine -> seated -> rise) gets
# spawned and learned; spawns past ~300deg open the landing gate at birth,
# giving dense on-policy data on the crouch->stand last mile.
MIDROLL_PITCH_MIN = math.radians(50.0)
MIDROLL_PITCH_MAX = math.radians(340.0)
MIDROLL_OMEGA_RANGE = (0.0, 3.0)  # rad/s forward momentum at spawn

# Tuck anchor: legs folded (crouch-anchor values from the velstand crouch
# reset) + CHIN TUCK (neck_pitch -1 / head_pitch +1 puts the flat head top
# squarely on the floor -- measured axis_z -0.99 vs +0.6 for the passive
# face-plant; the head-top latch requires this, so mid-roll spawns must
# demonstrate the tucked configuration). Servo-index keyed; mid-roll spawns
# lerp HOME->tuck by a per-env factor.
TUCK_OVERRIDES = {
    2: -1.15,  # left  hip_pitch
    3: 1.25,  # left  knee
    4: 1.05,  # left  ankle
    5: -1.0,  # neck_pitch  (chin tuck)
    6: 1.0,  # head_pitch  (chin tuck)
    11: 1.15,  # right hip_pitch
    12: -1.25,  # right knee
    13: -1.05,  # right ankle
}

# Rotation thresholds (rad) for the state-based gates.
LANDING_GATE_LO = math.radians(260.0)
LANDING_GATE_HI = math.radians(330.0)
RISE_GATE_LO = math.radians(180.0)
RISE_GATE_HI = math.radians(260.0)

LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
NECK_JOINTS = [5, 6, 7, 8]
