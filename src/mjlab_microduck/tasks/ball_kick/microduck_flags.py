"""Tunables for the Microduck BallKick task.

Direct port of the module-level constants that used to sit at the top of
``tasks/microduck_ball_kick_env_cfg.py``. See that file's original docstring
for the full task design rationale (kept verbatim on
``MicroduckBallKickEnvCfg`` in ``microduck_ball_kick_env_cfg.py``).
"""

# Kicking foot: "right" or "left". Flips the ball spawn side and the
# support-foot (anti-hop) sensor. Train the two policies as separate runs --
# wandb experiment/run name follows this flag.
KICK_FOOT = "right"
assert KICK_FOOT in ("right", "left")

# Symmetry -- must stay OFF: the kick task is inherently one-footed.
ENABLE_SYMMETRY = False

# -- Domain randomisation (matched to velocity / standup) -------------------
ENABLE_COM_RANDOMIZATION = True
ENABLE_HEAD_COM_RANDOMIZATION = True
ENABLE_KP_RANDOMIZATION = False
ENABLE_KD_RANDOMIZATION = False
ENABLE_MASS_INERTIA_RANDOMIZATION = True
ENABLE_JOINT_FRICTION_RANDOMIZATION = True
ENABLE_ARMATURE_RANDOMIZATION = True
ENABLE_VELOCITY_PUSHES = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS = True

# -- Ranges (matched to velocity / standup) ----------------------------------
COM_RANDOMIZATION_RANGE = 0.003  # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE = 0.003  # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ENCODER_BIAS_RANGE = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE = (0.9, 1.1)  # unused (kd DR off)
VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)
VELOCITY_PUSH_PLAY_INTERVAL_S = (0.5, 1.0)
VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # ramped in via push curriculum
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# -- Task constants -----------------------------------------------------------
# Long enough for kick + several seconds of ball-rolling reward + settle-back.
EPISODE_LENGTH_S = 5.0

# 70mm-diameter / 15g ball (see ball.xml).
BALL_RADIUS = 0.035
# Nominal ball-center offset in the robot's yaw frame. Measured at HOME: foot
# centers at (0, +-0.042), toe tip x~0.034. With radius 0.035 and +-0.015
# noise the ball's rear surface is at worst x=0.040 -> always >=6mm clear of
# the toe. The lateral sign follows the kicking foot (right = -y, left = +y).
BALL_OFFSET_X = 0.09
BALL_OFFSET_ABS_Y = 0.042
# Uniform +- placement noise per axis. This is the DR that makes the BLIND
# policy's swing robust to real-world aiming error.
BALL_POS_NOISE_XY = 0.015

# Target kick speed (m/s). NOTE: the kick reward weights in
# microduck_rewards_cfg.py are scaled to keep the at-target payoff ~= +3/step
# regardless of this value -- if you change the target, rescale the weights
# with it.
BALL_TARGET_SPEED = 1.0

# Trunk standing height (measured natural equilibrium at HOME -- see standup env).
STAND_Z = 0.115

LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
NECK_JOINTS = [5, 6, 7, 8]


def support_foot_of(kick_foot: str) -> str:
    """The non-kicking foot, which must stay planted through the kick."""
    assert kick_foot in ("right", "left")
    return "left" if kick_foot == "right" else "right"


def ball_offset_y_of(kick_foot: str) -> float:
    """Signed lateral ball spawn offset for the given kicking foot."""
    return -BALL_OFFSET_ABS_Y if kick_foot == "right" else BALL_OFFSET_ABS_Y
