"""Tunables for the Microduck velocity (walking) task.

Every ``microduck_*_cfg.py`` file next to this one imports from here instead
of duplicating magic numbers, so there is exactly one place to flip a DR
toggle or change a curriculum stage. This is the direct successor of the
module-level constants that used to sit at the top of the old monolithic
``microduck_velocity_env_cfg.py``.

The main locomotion task: velocity-command tracking + head-pose commands.
The reward/regularization recipe is locomotion-focused (lean tracking +
gait/feet terms, curriculum-ramped action-rate smoothing), with:

  - foot_slip kept at -0.1 (deliberately weak -- stronger was too restrictive
    for this robot's pivot-heavy turning)
  - fixed, modest command ranges (ang +-1.0 makes turning learnable) instead of
    a widening curriculum that outpaced the robot's capability
  - turn-in-place: 15% of envs get lin=0 + |ang| in [0.4, 1.0] (2026-07 audit:
    independent uniform sampling makes spin-on-the-spot ~2% of data -> untrained)
  - head_pose_tracking as a primary objective, plus an EMA-based head_pose_bias
    penalty that prices only the escapable DC head droop (see below)
  - body_pose tracking infra kept intact but DISABLED (weight 0) so the obs
    slot stays alive for envs that use it
"""

NUM_STEPS_PER_ENV = 24

# Fraction of envs commanded to spin on the spot (lin=0, |ang| in [0.4*max, max]).
TURN_IN_PLACE_FRACTION = 0.15

# Symmetry
ENABLE_SYMMETRY = False

# Domain randomization toggles
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # Randomize CoM of the head assembly bodies
ENABLE_KP_RANDOMIZATION = False  # Was True
ENABLE_KD_RANDOMIZATION = False  # Was True
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # Can enable once walking is stable
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # Scales BAM's friction budget per-env via FrictionDRBamActuator.friction_scale
ENABLE_JOINT_DAMPING_RANDOMIZATION = False
ENABLE_ARMATURE_RANDOMIZATION = True  # Reflected rotor inertia (microban-style). DOES affect BAM (armature is set, not zeroed).
ENABLE_VELOCITY_PUSHES = True  # Velocity-based pushes for robustness training
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # Simulates mounting errors
ENABLE_ENCODER_BIAS = True  # Per-env joint encoder calibration offset (actor obs sees joint_pos + bias)
ENABLE_BASE_ORIENTATION_RANDOMIZATION = False  # Randomize initial tilt to force reactive behavior

# Head/body pose command tracking (replaces the old neck-offset disturbance scheme).
# Head pose: 4D deltas-from-HOME on neck/head joints; vel env tracks these as a
# primary objective. Body pose: 6D delta in [x, y, z, roll, pitch, yaw]; vel env
# samples small ranges + tiny reward weight so input neurons stay alive but
# tracking isn't the priority (standup env raises the weight).
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)

# Observation configuration
USE_PROJECTED_GRAVITY = True  # If True, use projected gravity instead of raw accelerometer

# Domain randomization ranges (adjust as needed)
# Conservative ranges proven to be stable - can increase gradually if needed
COM_RANDOMIZATION_RANGE = 0.003  # +-3mm initial, ramped to +-8mm via curriculum
# Head CoM randomization: applied per-episode to every body of the head assembly
# (neck -> neck_pitch -> yaw_roll_motion -> head-roll body). Same non-accumulating
# mechanism as the trunk CoM randomization above. The head-roll body is named
# bottom_head_shell in the walk model and jaw_soft in the 2026-07 roller model,
# hence the alternation. NOTE: bearing_roll is NOT a head body -- in both models
# it is the right-hip-yaw link (child of trunk_base); it has always been listed
# here by mistake and is kept only to preserve existing DR behavior.
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # +-3mm initial, ramped via curriculum
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)  # +-5% applied to BOTH mass and inertia together.
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # +-15%
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # +-10% (can increase to 0.8-1.2)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_DAMPING_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)  # +-10% reflected rotor inertia (microban: dr.joint_armature, same range)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)  # Apply pushes every 3-6 seconds
VELOCITY_PUSH_PLAY_INTERVAL_S = (0.5, 1.0)  # Shorter interval in play mode, for visibility
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # Velocity change range in m/s. Was +-0.5 -- an
# ADDITIVE kick larger than max walk speed (0.4) every 3-6 s trains a permanently
# nervous fall-recovery gait (2026-07 audit). +-0.3 keeps push robustness while
# letting a calmer gait be optimal.
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # up-to-6 deg random-axis IMU mounting error. NOTE: zero-centered (random axis) -- trains tolerance to misalignment *magnitude*, NOT a pitch bias. The real board's systematic ~5 deg pitch offset is corrected at the source in the runtime (imu-pitch-offset), not here.
ENCODER_BIAS_RANGE = (-0.015, 0.015)  # +-0.86 deg per-joint encoder offset (constant per env)
BASE_ORIENTATION_MAX_PITCH_DEG = 10.0  # +-10 deg forward/backward tilt at episode start
BASE_ORIENTATION_MAX_ROLL_DEG = 5.0  # +-5 deg side-to-side tilt at episode start
