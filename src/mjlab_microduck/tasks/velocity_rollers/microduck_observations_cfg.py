"""Observation specifications for the Microduck velocity_rollers task.

Ported 1:1 from the observation-editing block of the old
``microduck_velocity_rollers_env_cfg.py``. The old ``deepcopy(...)`` calls
that existed purely to avoid corrupting state shared with other env cfgs are
unnecessary now -- every ``@configclass`` instance gets its own
deep-copied term objects (see ``mjlab_microduck.utils.configclass``).
"""

from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.locomotion.velocity.cfg.observations_cfg import CriticCfg, ObservationsCfg, PolicyCfg
from mjlab_microduck.tasks.velocity_rollers.microduck_flags import (
    ENABLE_ENCODER_BIAS,
    ENABLE_IMU_ORIENTATION_RANDOMIZATION,
    IMU_ORIENTATION_RANDOMIZATION_ANGLE,
)
from mjlab_microduck.utils.configclass import configclass

# Excludes the passive wheel joints so obs dim matches the action dim (14).
_PASSIVE_EXCLUDED = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
_WHEEL_ONLY = SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",))

_HEAD_COMMAND = ObsTerm(func=microduck_mdp.zero_command_padding, params={"dim": 4})
_BODY_COMMAND = ObsTerm(func=microduck_mdp.zero_command_padding, params={"dim": 6})

if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
    _base_ang_vel_func = microduck_mdp.base_ang_vel_imu_misaligned
    _base_ang_vel_params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
    _gravity_func = microduck_mdp.projected_gravity_imu_misaligned
    _gravity_params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
else:
    _base_ang_vel_func = mdp.builtin_sensor
    _base_ang_vel_params = {"sensor_name": "robot/imu_ang_vel"}
    _gravity_func = mdp.projected_gravity
    _gravity_params = {}


@configclass
class MicroduckPolicyCfg(PolicyCfg):
    """Actor observations for the velocity_rollers task."""

    base_lin_vel: ObsTerm | None = None  # not observable by the actor; critic-only below.
    height_scan: ObsTerm | None = None  # flat terrain only, no terrain sensor.

    base_ang_vel: ObsTerm | None = ObsTerm(
        func=_base_ang_vel_func,
        params=_base_ang_vel_params,
        noise=Unoise(n_min=-0.03, n_max=0.03),
        delay_min_lag=0,
        delay_max_lag=1,
        delay_update_period=64,
    )
    projected_gravity: ObsTerm | None = ObsTerm(
        func=_gravity_func,
        params=_gravity_params,
        noise=Unoise(n_min=-0.01, n_max=0.01),
        delay_min_lag=0,
        delay_max_lag=1,
        delay_update_period=64,
    )
    joint_pos: ObsTerm | None = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.001, n_max=0.001),
        params={"asset_cfg": _PASSIVE_EXCLUDED, "biased": ENABLE_ENCODER_BIAS},
    )
    # 1-ctrl-step lag: the Dynamixel firmware computes present_velocity via a
    # moving-average over the previous position-sample window.
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
    """Critic (privileged) observations for the velocity_rollers task."""

    height_scan: ObsTerm | None = None  # flat terrain only, no terrain sensor.
    foot_height: ObsTerm | None = None  # no foot-height terrain sensor on this task.

    base_lin_vel: ObsTerm | None = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
    joint_pos: ObsTerm | None = ObsTerm(
        func=mdp.joint_pos_rel, params={"asset_cfg": _PASSIVE_EXCLUDED, "biased": False}
    )
    joint_vel: ObsTerm | None = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _PASSIVE_EXCLUDED})
    head_command: ObsTerm | None = _HEAD_COMMAND
    body_command: ObsTerm | None = _BODY_COMMAND

    # Privileged wheel speeds for the critic (4 wheels in the current model).
    wheel_vel: ObsTerm | None = ObsTerm(func=mdp.joint_vel_rel, scale=1.0, params={"asset_cfg": _WHEEL_ONLY})


@configclass
class MicroduckObservationsCfg(ObservationsCfg):
    actor: MicroduckPolicyCfg = MicroduckPolicyCfg()
    critic: MicroduckCriticCfg = MicroduckCriticCfg()
