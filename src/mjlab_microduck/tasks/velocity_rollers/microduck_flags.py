"""Tunables for the Microduck velocity_rollers (roller skate) task.

Direct port of the module-level constants from the old
``tasks/microduck_velocity_rollers_env_cfg.py``. See that history / this
package's ``MicroduckVelocityRollersEnvCfg`` docstring for the task design
rationale.
"""

# Symmetry -- OFF: SYMMETRY_CFG's obs permutation is hardcoded for the old
# 51D layout and breaks on the unified 61D obs (same situation as every
# other v1.5+ env).
ENABLE_SYMMETRY = False

# -- Domain randomisation (matched to the velocity env) ---------------------
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # BAM friction budget per-env (legs)
ENABLE_ARMATURE_RANDOMIZATION = True  # legs only -- NOT the wheel bearings
ENABLE_WHEEL_FRICTION_RANDOMIZATION = True  # bearing frictionloss on passive wheels
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # obs-level per-env rotation
ENABLE_ENCODER_BIAS = True

# -- Ranges (matched to the velocity env unless roller-specific) ------------
COM_RANDOMIZATION_RANGE = 0.003  # +-3mm initial, ramped via curriculum
HEAD_COM_RANDOMIZATION_RANGE = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_RANGE = (-0.2, 0.2)  # roller-specific: gentler than walk +-0.3
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE = (-0.015, 0.015)
