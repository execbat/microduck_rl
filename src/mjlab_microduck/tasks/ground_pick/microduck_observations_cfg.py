"""Observation specifications for the Microduck ground_pick task.

Ported 1:1 from the observation-editing block of the old
``microduck_ground_pick_env_cfg.py``. Note the IMU delay window here
(``delay_max_lag=3``) is wider than velocity/ball_kick/velocity_rollers's
``1`` -- kept exactly as in the original, not "fixed" to match.
"""

from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.ground_pick.microduck_flags import ENABLE_ENCODER_BIAS, IMU_ORIENTATION_RANDOMIZATION_ANGLE
from mjlab_microduck.tasks.locomotion.velocity.cfg.observations_cfg import CriticCfg, ObservationsCfg, PolicyCfg
from mjlab_microduck.utils.configclass import configclass

# Passive-exclusion regex is a harmless no-op under mjlab 1.3.0 + canonical
# BAM (no passive joints in this articulation) but kept for parity with the
# other envs.
_PASSIVE_EXCLUDED = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))

_HEAD_COMMAND = ObsTerm(func=microduck_mdp.zero_command_padding, params={"dim": 4})
_BODY_COMMAND = ObsTerm(func=microduck_mdp.zero_command_padding, params={"dim": 6})


@configclass
class MicroduckPolicyCfg(PolicyCfg):
    """Actor observations for the ground_pick task -- identical 61D layout
    to the walking policy so the two can be switched at runtime."""

    base_lin_vel: ObsTerm | None = None  # not observable by the actor; critic-only below.
    height_scan: ObsTerm | None = None  # no terrain-height sensor on this task.

    base_ang_vel: ObsTerm | None = ObsTerm(
        func=microduck_mdp.base_ang_vel_imu_misaligned,
        params={"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE},
        noise=Unoise(n_min=-0.03, n_max=0.03),
        delay_min_lag=0,
        delay_max_lag=3,
        delay_update_period=64,
    )
    projected_gravity: ObsTerm | None = ObsTerm(
        func=microduck_mdp.projected_gravity_imu_misaligned,
        params={"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        delay_min_lag=0,
        delay_max_lag=3,
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
    # No head/body pose command on this task (the head is driven by the
    # task's own phase motion), but every microduck policy shares the same
    # 61D obs shape so the runtime can feed a single command buffer -- the
    # 10 trailing slots (head 4 + body 6) are constant zero.
    head_command: ObsTerm | None = _HEAD_COMMAND
    body_command: ObsTerm | None = _BODY_COMMAND


@configclass
class MicroduckCriticCfg(CriticCfg):
    """Critic (privileged) observations for the ground_pick task."""

    height_scan: ObsTerm | None = None  # no terrain-height sensor on this task.
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
