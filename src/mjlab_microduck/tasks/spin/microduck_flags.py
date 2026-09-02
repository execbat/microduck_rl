"""Tunables for the Microduck spin task.

Cyclic trick triggered by button A via the runtime's --ground-pick slot:
~1 turn counter-clockwise at ~3 rad/s then a clean stop, standing.

Hybrid task, built from two other tasks' machinery:
  - physics / roller robot           <- velocity_rollers
  - cyclic phase machinery           <- roller_crouch
    (GroundPickPhaseCommand: [cos(2*pi*phase), sin(2*pi*phase), 0], see
    microduck_commands_cfg.py)

Core difference from crouch: the phase drives a TARGET YAW RATE (an outcome
goal) rather than a joint pose. Two decaying bootstraps push toward
differential rolling -- the only physically-certain mechanism on 4 passive
wheels: left skate backward, right skate forward.

Unified 61D obs -> hot-swappable at runtime with roller/ground_pick/crouch.
See docs/superpowers/specs/2026-08-04-spin-env-design.md.
"""

# Left/right symmetry would turn a left spin into a right spin: forbidden here.
ENABLE_SYMMETRY = False

# -- Domain randomisation (matched to velocity_rollers) ----------------------
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_WHEEL_FRICTION_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

COM_RANDOMIZATION_RANGE = 0.003
HEAD_COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.2, 0.2)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE = (-0.015, 0.015)

# The button can be pressed standing still OR while rolling slowly: the
# policy learns to kill residual momentum before/during the spin launch.
ENTRY_VELOCITY_X = (0.0, 0.3)

# Neck/head are held near-neutral EXCEPT head_yaw, left free: it can act as
# a flywheel to help launch the rotation.
NECK_PATTERN_NO_YAW = r"^(neck_pitch|head_pitch|head_roll)$"
