"""Observation specifications for the Microduck standup task.

Identical layout to the walking/sitstand policies (unified 61D obs).
``head_command`` carries the real ``head_pose`` command; ``body_command``
carries the real ``body_pose`` command when ``ENABLE_BODY_CONTROL`` is on
(zero padding otherwise -- obs shape stays 61D either way).

The retained sensor-derived critic terms (``foot_contact_forces``,
``foot_air_time``) get NaN-safe wrappers: a non-finite contact force slips
past ``robot_state_is_nan`` (it checks joint + root state only) and a
single NaN there kills the run via rsl_rl's ``check_nan`` (a real crash this
project hit once). Standup lands and flips constantly, so degenerate
contacts are MORE likely here than elsewhere.
"""

from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.standup.microduck_flags import ENABLE_BODY_CONTROL, ENABLE_ENCODER_BIAS, IMU_ORIENTATION_RANDOMIZATION_ANGLE
from mjlab_microduck.tasks.locomotion.velocity.cfg.observations_cfg import CriticCfg, ObservationsCfg, PolicyCfg
from mjlab_microduck.utils.configclass import configclass

_PASSIVE_EXCLUDED = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))

_HEAD_COMMAND = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
_BODY_COMMAND = (
    ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})
    if ENABLE_BODY_CONTROL
    else ObsTerm(func=microduck_mdp.zero_command_padding, params={"dim": 6})
)


@configclass
class MicroduckPolicyCfg(PolicyCfg):
    """Actor observations for the standup task."""

    base_lin_vel: ObsTerm | None = None  # not observable by the actor; critic-only below.
    height_scan: ObsTerm | None = None  # flat terrain only (even on rough, see MicroduckStandupRoughEnvCfg).

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
    """Critic (privileged) observations for the standup task."""

    height_scan: ObsTerm | None = None  # flat terrain only (even on rough).
    foot_height: ObsTerm | None = None  # no foot-height terrain sensor on this task.

    base_lin_vel: ObsTerm | None = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
    joint_pos: ObsTerm | None = ObsTerm(
        func=mdp.joint_pos_rel, params={"asset_cfg": _PASSIVE_EXCLUDED, "biased": False}
    )
    joint_vel: ObsTerm | None = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _PASSIVE_EXCLUDED})
    foot_air_time: ObsTerm | None = ObsTerm(
        func=microduck_mdp.foot_air_time_safe, params={"sensor_name": "feet_ground_contact"}
    )
    foot_contact_forces: ObsTerm | None = ObsTerm(
        func=microduck_mdp.foot_contact_forces_safe, params={"sensor_name": "feet_ground_contact"}
    )
    head_command: ObsTerm | None = _HEAD_COMMAND
    body_command: ObsTerm | None = _BODY_COMMAND


@configclass
class MicroduckObservationsCfg(ObservationsCfg):
    actor: MicroduckPolicyCfg = MicroduckPolicyCfg()
    critic: MicroduckCriticCfg = MicroduckCriticCfg()
