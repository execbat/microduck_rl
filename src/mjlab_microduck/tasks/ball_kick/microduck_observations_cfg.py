"""Observation specifications for the Microduck BallKick task.

Ported 1:1 from the observation-editing block of the old
``microduck_ball_kick_env_cfg.py``. As with the velocity task's split, the
old ``deepcopy(...)`` calls that existed purely to avoid corrupting state
shared with *other* env cfgs are unnecessary now -- every ``@configclass``
instance gets its own deep-copied term objects (see
``mjlab_microduck.utils.configclass``).
"""

from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.ball_kick.microduck_flags import IMU_ORIENTATION_RANDOMIZATION_ANGLE
from mjlab_microduck.utils.configclass import configclass
from mjlab_microduck.tasks.velocity.cfg.observations_cfg import CriticCfg, ObservationsCfg, PolicyCfg

# Excludes the passive_* joints (jaw linkage) so the observation dim matches
# the action dim (14) instead of the raw articulation (16).
_PASSIVE_EXCLUDED = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))

# Command-obs slots for unified-layout parity: [twist(3), head(4), body(6)].
# This task has no head/body pose command, so both groups get zero-padding.
_HEAD_COMMAND = ObsTerm(func=microduck_mdp.zero_command_padding, params={"dim": 4})
_BODY_COMMAND = ObsTerm(func=microduck_mdp.zero_command_padding, params={"dim": 6})


@configclass
class MicroduckPolicyCfg(PolicyCfg):
    """Actor observations for the BallKick task -- blind to the ball."""

    base_lin_vel: ObsTerm | None = None  # not observable by the actor; critic-only below.
    height_scan: ObsTerm | None = None  # flat terrain only, no terrain sensor.

    base_ang_vel: ObsTerm | None = ObsTerm(
        func=microduck_mdp.base_ang_vel_imu_misaligned,
        params={"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE},
        noise=Unoise(n_min=-0.03, n_max=0.03),
        delay_min_lag=0,
        delay_max_lag=1,
        delay_update_period=64,
    )
    projected_gravity: ObsTerm | None = ObsTerm(
        func=microduck_mdp.projected_gravity_imu_misaligned,
        params={"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        delay_min_lag=0,
        delay_max_lag=1,
        delay_update_period=64,
    )
    joint_pos: ObsTerm | None = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.001, n_max=0.001),
        params={"asset_cfg": _PASSIVE_EXCLUDED, "biased": True},
    )
    # 1-ctrl-step lag: the Dynamixel firmware computes present_velocity via a
    # moving-average over the previous position-sample window (see velocity task).
    joint_vel: ObsTerm | None = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-0.25, n_max=0.25),
        params={"asset_cfg": _PASSIVE_EXCLUDED},
        delay_min_lag=1,
        delay_max_lag=1,
        delay_update_period=0,
    )
    head_command: ObsTerm | None = _HEAD_COMMAND
    body_command: ObsTerm | None = _BODY_COMMAND


@configclass
class MicroduckCriticCfg(CriticCfg):
    """Critic (privileged) observations for the BallKick task.

    Sees ground-truth base velocity plus the ball's position/velocity in the
    robot's base frame (asymmetric actor-critic: the real robot has no ball
    sensing, but the value function can use it to predict the kick payoff).
    """

    height_scan: ObsTerm | None = None  # flat terrain only, no terrain sensor.
    foot_height: ObsTerm | None = None  # no foot-height terrain sensor on this task.

    base_lin_vel: ObsTerm | None = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
    joint_pos: ObsTerm | None = ObsTerm(
        func=mdp.joint_pos_rel, params={"asset_cfg": _PASSIVE_EXCLUDED, "biased": False}
    )
    joint_vel: ObsTerm | None = ObsTerm(
        func=mdp.joint_vel_rel, params={"asset_cfg": _PASSIVE_EXCLUDED}
    )
    head_command: ObsTerm | None = _HEAD_COMMAND
    body_command: ObsTerm | None = _BODY_COMMAND

    ball_position: ObsTerm | None = ObsTerm(
        func=microduck_mdp.ball_pos_in_base, params={"asset_name": "ball"}
    )
    ball_velocity: ObsTerm | None = ObsTerm(
        func=microduck_mdp.ball_vel_in_base, params={"asset_name": "ball"}
    )


@configclass
class MicroduckObservationsCfg(ObservationsCfg):
    actor: MicroduckPolicyCfg = MicroduckPolicyCfg()
    critic: MicroduckCriticCfg = MicroduckCriticCfg()
