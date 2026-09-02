"""Observation specifications for the Microduck sitstand task.

Identical layout to the walking/sit/standup policies (unified 61D obs).
``head_command`` carries the REAL ``head_pose`` command (commandable head
control, like velocity/standup); ``body_command`` stays zero-padded (body
control isn't used here).
"""

from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.sitstand.microduck_flags import ENABLE_ENCODER_BIAS, IMU_ORIENTATION_RANDOMIZATION_ANGLE
from mjlab_microduck.tasks.locomotion.velocity.cfg.observations_cfg import CriticCfg, ObservationsCfg, PolicyCfg
from mjlab_microduck.utils.configclass import configclass

_PASSIVE_EXCLUDED = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))

_HEAD_COMMAND = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
_BODY_COMMAND = ObsTerm(func=microduck_mdp.zero_command_padding, params={"dim": 6})


@configclass
class MicroduckPolicyCfg(PolicyCfg):
    """Actor observations for the sitstand task."""

    base_lin_vel: ObsTerm | None = None  # not observable by the actor; critic-only below.
    height_scan: ObsTerm | None = None  # flat terrain only (even on rough, see MicroduckSitStandRoughEnvCfg).

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
    """Critic (privileged) observations for the sitstand task.

    ``foot_air_time``/``foot_contact``/``foot_contact_forces`` are kept
    inherited unchanged from the base ``CriticCfg`` -- they use the
    ``feet_ground_contact`` sensor, which this task does define.
    """

    height_scan: ObsTerm | None = None  # flat terrain only (even on rough).
    foot_height: ObsTerm | None = None  # no foot-height terrain sensor on this task.

    base_lin_vel: ObsTerm | None = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
    joint_pos: ObsTerm | None = ObsTerm(
        func=mdp.joint_pos_rel, params={"asset_cfg": _PASSIVE_EXCLUDED, "biased": False}
    )
    joint_vel: ObsTerm | None = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _PASSIVE_EXCLUDED})
    head_command: ObsTerm | None = _HEAD_COMMAND
    body_command: ObsTerm | None = _BODY_COMMAND


@configclass
class MicroduckObservationsCfg(ObservationsCfg):
    actor: MicroduckPolicyCfg = MicroduckPolicyCfg()
    critic: MicroduckCriticCfg = MicroduckCriticCfg()
