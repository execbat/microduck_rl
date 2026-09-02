"""Observation specifications for the Microduck velocity task.

Overrides/extends the generic ``cfg.observations_cfg`` groups:

- drops ``base_lin_vel``/``height_scan`` from the actor (no body-mounted
  terrain sensor on this robot; lin-vel goes to the critic only, as
  privileged information)
- swaps in IMU-mounting-misalignment-aware variants of gravity/ang-vel when
  ``ENABLE_IMU_ORIENTATION_RANDOMIZATION`` is set
- adds sensor-delay to mimic real actuator/IMU sampling latency
- excludes the passive jaw-linkage joints from joint_pos/joint_vel so the obs
  dim matches the action dim
- feeds the actor a per-env encoder-bias-corrupted joint_pos while the critic
  keeps ground truth (``ENABLE_ENCODER_BIAS``)
- adds ``head_command``/``body_command`` terms (see microduck_commands_cfg.py)
- routes the critic's sensor-derived terms through NaN-safe wrappers, since
  ``nan_state`` termination (see microduck_terminations_cfg.py) can't catch a
  NaN raycast/contact read that leaves joint/root state clean
"""

from mjlab.managers.observation_manager import ObservationTermCfg as ObsTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.utils.configclass import configclass

from mjlab_microduck.tasks.locomotion.velocity.cfg.observations_cfg import CriticCfg, ObservationsCfg, PolicyCfg
from .microduck_flags import ENABLE_ENCODER_BIAS, ENABLE_IMU_ORIENTATION_RANDOMIZATION, IMU_ORIENTATION_RANDOMIZATION_ANGLE, USE_PROJECTED_GRAVITY

# Exclude passive_* joints (jaw linkage) so the observation dim matches the
# action dim (14) instead of the raw articulation (16).
_PASSIVE_EXCLUDED = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))

_ANG_VEL_FUNC = microduck_mdp.base_ang_vel_imu_misaligned if ENABLE_IMU_ORIENTATION_RANDOMIZATION else mdp.builtin_sensor
_ANG_VEL_PARAMS = (
    {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION
    else {"sensor_name": "robot/imu_ang_vel"}
)

# Gravity/accelerometer term: exactly one of `projected_gravity` /
# `raw_accelerometer` below is non-None depending on USE_PROJECTED_GRAVITY --
# both are declared as real fields (rather than one dynamically-named one) so
# this stays a plain, statically-inspectable dataclass.
_GRAVITY_NOISE = Unoise(n_min=-0.01, n_max=0.01)  # was 0.15
_GRAVITY_DELAY = dict(delay_min_lag=0, delay_max_lag=1, delay_update_period=64)


@configclass
class MicroduckPolicyCfg(PolicyCfg):
    """Actor observations for the Microduck velocity task."""

    base_lin_vel: ObsTerm | None = None  # not observable by the actor; critic-only below.
    height_scan: ObsTerm | None = None  # no body-mounted terrain sensor on this robot.

    base_ang_vel: ObsTerm | None = ObsTerm(
        func=_ANG_VEL_FUNC,
        params=_ANG_VEL_PARAMS,
        noise=Unoise(n_min=-0.03, n_max=0.03),  # was 0.2
        delay_min_lag=0,
        delay_max_lag=1,  # was 3 (=60 ms worst case); real dxl IMU path is fast -- +-20 ms envelope (2026-07 audit)
        delay_update_period=64,
    )

    joint_pos: ObsTerm | None = ObsTerm(
        func=mdp.joint_pos_rel,
        noise=Unoise(n_min=-0.001, n_max=0.001),  # was 0.05
        params=(
            {"asset_cfg": _PASSIVE_EXCLUDED, "biased": True}
            if ENABLE_ENCODER_BIAS
            else {"asset_cfg": _PASSIVE_EXCLUDED}
        ),
    )
    # 1-ctrl-step lag: the Dynamixel firmware computes present_velocity via a
    # moving-average over the previous position-sample window, so the value the
    # policy actually reads is ~1 control period old. Matches reality and stops
    # the policy relying on instantaneous qdot feedback.
    joint_vel: ObsTerm | None = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-0.25, n_max=0.25),  # was 2.0
        params={"asset_cfg": _PASSIVE_EXCLUDED},
        delay_min_lag=1,
        delay_max_lag=1,
        delay_update_period=0,
    )

    head_command: ObsTerm | None = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "head_pose"}
    )
    body_command: ObsTerm | None = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "body_pose"}
    )

    projected_gravity: ObsTerm | None = (
        ObsTerm(
            func=microduck_mdp.projected_gravity_imu_misaligned
            if ENABLE_IMU_ORIENTATION_RANDOMIZATION
            else mdp.projected_gravity,
            params={"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
            if ENABLE_IMU_ORIENTATION_RANDOMIZATION
            else {},
            noise=_GRAVITY_NOISE,
            **_GRAVITY_DELAY,
        )
        if USE_PROJECTED_GRAVITY
        else None
    )
    raw_accelerometer: ObsTerm | None = (
        ObsTerm(func=microduck_mdp.raw_accelerometer, noise=_GRAVITY_NOISE, **_GRAVITY_DELAY)
        if not USE_PROJECTED_GRAVITY
        else None
    )


@configclass
class MicroduckCriticCfg(CriticCfg):
    """Critic (privileged) observations for the Microduck velocity task."""

    height_scan: ObsTerm | None = None

    # Ground-truth base linear velocity (not the noisy IMU-derived one the
    # actor would see, since the actor doesn't get base_lin_vel at all here).
    base_lin_vel: ObsTerm | None = ObsTerm(func=mdp.base_lin_vel, scale=1.0)

    joint_pos: ObsTerm | None = ObsTerm(
        func=mdp.joint_pos_rel,
        params=(
            {"asset_cfg": _PASSIVE_EXCLUDED, "biased": False}
            if ENABLE_ENCODER_BIAS
            else {"asset_cfg": _PASSIVE_EXCLUDED}
        ),
    )
    joint_vel: ObsTerm | None = ObsTerm(
        func=mdp.joint_vel_rel, params={"asset_cfg": _PASSIVE_EXCLUDED}
    )

    # NaN-safe wrappers: these sensor-derived terms are the one obs path
    # `nan_state` (see microduck_terminations_cfg.py) cannot protect -- it
    # checks joint + root state, while these read raycast/contact sensor data,
    # which MuJoCo can return non-finite for while the state is still clean.
    # A single NaN here previously killed the whole run via rsl_rl's
    # check_nan (the 2026-08-21 Velocity2-Rough-Backlash crash). Critic-only,
    # so sanitizing costs the policy nothing.
    foot_height: ObsTerm | None = ObsTerm(
        func=microduck_mdp.foot_height_safe, params={"sensor_name": "foot_height_scan"}
    )
    foot_air_time: ObsTerm | None = ObsTerm(
        func=microduck_mdp.foot_air_time_safe, params={"sensor_name": "feet_ground_contact"}
    )
    foot_contact_forces: ObsTerm | None = ObsTerm(
        func=microduck_mdp.foot_contact_forces_safe, params={"sensor_name": "feet_ground_contact"}
    )

    head_command: ObsTerm | None = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "head_pose"}
    )
    body_command: ObsTerm | None = ObsTerm(
        func=mdp.generated_commands, params={"command_name": "body_pose"}
    )


@configclass
class MicroduckObservationsCfg(ObservationsCfg):
    """Observation specifications for the Microduck velocity task."""

    actor: MicroduckPolicyCfg = MicroduckPolicyCfg()
    critic: MicroduckCriticCfg = MicroduckCriticCfg()
