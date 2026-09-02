"""Tunables for the Microduck sitstand task.

Direct port of the module-level constants from the old
``tasks/microduck_sitstand_env_cfg.py``. Keyframes here (``SITTING_TARGET_
OVERRIDES``, ``SIT_Z``) must stay in sync with
``microduck_standup_env_cfg.SITTING_JOINT_OVERRIDES`` -- see that file's own
flags module once migrated.
"""

ENABLE_SYMMETRY = False

# -- Domain randomisation (matched to the velocity env for sim2real parity) --
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True  # match velocity: randomize head-assembly CoM
ENABLE_KP_RANDOMIZATION = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION = True  # match velocity: dr.pseudo_inertia (mass+inertia)
ENABLE_JOINT_FRICTION_RANDOMIZATION = True  # match velocity: FrictionDRBamActuator.friction_scale
ENABLE_ARMATURE_RANDOMIZATION = True  # match velocity: reflected rotor inertia
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True  # match velocity: obs-level per-env misalignment
ENABLE_ENCODER_BIAS = True  # match velocity: per-env joint encoder offset (actor obs)

# -- Ranges (matched to the velocity env) ------------------------------------
COM_RANDOMIZATION_RANGE = 0.003  # ramped to 0.015 via com_range curriculum
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ramped to 0.01 via head_com_range curriculum
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # unused (kd DR off)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_PLAY_INTERVAL_S = (0.5, 1.0)
# Final magnitude matches velocity's +-0.3 but the ramp is DELAYED (see the
# push_magnitude curriculum): pushes mid-descent before the transition
# motions have consolidated make the policy unlearn them and converge to
# "just stand doing nothing".
VELOCITY_PUSH_RANGE = (-0.3, 0.3)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0  # match velocity (obs-level, zero-centered random axis)

# Episode length: room for 2-3 posture segments (dwell 3.5-6.5s each), i.e.
# at least one full sit -> rest -> rise -> rest cycle per episode.
EPISODE_LENGTH_S = 12.0
# Dwell time in each commanded posture before a resample may flip it. The
# lower bound must comfortably exceed a gentle transition (~1.5s) plus some
# rest, so "arrive, then hold still" is always trained.
POSTURE_DWELL_S = (3.5, 6.5)
# Probability a resample commands SIT (vs STAND). 0.5 -> all four
# combinations of (reset state x command) get equal coverage, including
# both holds.
SIT_PROB = 0.5

# -- SIT keyframe (joint_pos index -> angle in rad). Single fixed target. ---
# STABILITY-VERIFIED 2026-07-27 (sweep_sit_pose2.py): knee +-1.35, hip_pitch
# = HOME -+0.05 lean, ankle 0, hip_roll 0 settles at 3-5deg tilt for 95-100%
# of noisy resets. The old keyframe (knee +-1.0472, hip_pitch HOME) is NOT
# statically stable -- it tips to ~88deg in 1s. If the robot or keyframe
# changes, RE-RUN THE SWEEP -- verify tilt, not z.
SITTING_TARGET_OVERRIDES = {
    1: 0.0,  # left  hip_roll   (HOME -0.0873)
    2: -0.4079,  # left  hip_pitch  (HOME -0.4579; +0.05 = slight fwd lean)
    3: 1.35,  # left  knee       (HOME -0.0049)
    4: 0.0,  # left  ankle      (HOME +0.4530)
    # neck/head intentionally omitted -> steered by the head_pose command.
    10: 0.0,  # right hip_roll   (HOME +0.0873)
    11: 0.4079,  # right hip_pitch  (HOME +0.4579)
    12: -1.35,  # right knee       (HOME +0.0049)
    13: 0.0,  # right ankle      (HOME -0.4530)
}

LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
NECK_JOINTS = [5, 6, 7, 8]

# Trunk height targets (m) -- both MEASURED in sim, never carried across
# robot or keyframe changes.
STAND_Z = 0.115
SIT_Z = 0.060

# Upright gating window for ``upright_while_tall``: full upright incentive
# above STAND_UPRIGHT_Z, fades to 0 at SIT_UPRIGHT_Z (committed to the sit).
# Blocks the "tip backward while still high" descent exploit; the always-on
# upright_linear floor covers the seated regime.
STAND_UPRIGHT_Z = 0.10
SIT_UPRIGHT_Z = 0.075

# Target-ramp duration (s): the command term slews an internal target blend
# STAND<->SIT over this time, and the posture rewards track the MOVING
# target. THE anti-crash mechanism (a binary target lets arriving early pay
# the full goal jackpot for every step saved). 55mm over 2s ~= 0.028 m/s,
# comfortably under both speed caps below.
POSTURE_RAMP_S = 2.0

# Vertical-speed caps (m/s) -- backstops for overshoot/bounce around the
# slewed target (see POSTURE_RAMP_S), not the primary gentleness mechanism.
# The rise cap is looser (rising against gravity needs some momentum to get
# over the heels) and is introduced by curriculum only after the rise
# motion has been discovered -- see the rise_speed_weight curriculum.
MAX_DESCENT_SPEED = 0.05
MAX_RISE_SPEED = 0.08

# MuJoCo physics robustness (contact NaN fix): the standup XML has full
# collisions on every body; the seated pose puts trunk + folded legs + head
# all in close ground/self contact. The default nconmax/solver-iters
# overflow the contact solver on sit attempts -> NaN -> nan_state
# terminations that punish the descent itself. Applied UNCONDITIONALLY
# (not just on rough terrain) -- see MicroduckSitStandFlatEnvCfg.__post_init__.
SIM_NCONMAX = 200
SIM_ITERATIONS = 30
SIM_LS_ITERATIONS = 50
