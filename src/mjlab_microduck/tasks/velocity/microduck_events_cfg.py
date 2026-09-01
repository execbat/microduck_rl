"""Event (reset / domain-randomization) specifications for the Microduck
velocity task.

Every "ENABLE_*" toggle below just decides whether a field is a real
``EventTerm`` or ``None`` (disabled) -- see ``microduck_flags.py`` for the
single source of truth on which DR channels are currently on.
"""

import math

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.utils.configclass import configclass

from .cfg.events_cfg import EventsCfg
from .microduck_flags import (
    ARMATURE_RANDOMIZATION_RANGE,
    BASE_ORIENTATION_MAX_PITCH_DEG,
    BASE_ORIENTATION_MAX_ROLL_DEG,
    COM_RANDOMIZATION_RANGE,
    ENABLE_ARMATURE_RANDOMIZATION,
    ENABLE_BASE_ORIENTATION_RANDOMIZATION,
    ENABLE_COM_RANDOMIZATION,
    ENABLE_ENCODER_BIAS,
    ENABLE_HEAD_COM_RANDOMIZATION,
    ENABLE_JOINT_DAMPING_RANDOMIZATION,
    ENABLE_JOINT_FRICTION_RANDOMIZATION,
    ENABLE_KD_RANDOMIZATION,
    ENABLE_KP_RANDOMIZATION,
    ENABLE_MASS_INERTIA_RANDOMIZATION,
    ENABLE_VELOCITY_PUSHES,
    ENCODER_BIAS_RANGE,
    HEAD_BODY_NAMES,
    HEAD_COM_RANDOMIZATION_RANGE,
    JOINT_DAMPING_RANDOMIZATION_RANGE,
    JOINT_FRICTION_RANDOMIZATION_RANGE,
    KD_RANDOMIZATION_RANGE,
    KP_RANDOMIZATION_RANGE,
    MASS_INERTIA_RANDOMIZATION_RANGE,
    VELOCITY_PUSH_INTERVAL_S,
    VELOCITY_PUSH_RANGE,
)
from .microduck_scene_cfg import FOOT_FRICTION_GEOM_NAMES

_mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
_kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
_kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)


@configclass
class MicroduckEventsCfg(EventsCfg):
    """Event terms for the Microduck velocity task."""

    reset_base: EventTerm | None = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.12, 0.13),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        },
    )

    foot_friction: EventTerm | None = EventTerm(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_FRICTION_GEOM_NAMES),
            "operation": "abs",
            # Grippier footpad -- narrowed from (0.3, 1.2).
            "ranges": (0.7, 1.3),
            "shared_random": True,
        },
    )

    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    expand_bam_friction_fields: EventTerm | None = EventTerm(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup"
    )
    reset_action_history: EventTerm | None = EventTerm(
        func=microduck_mdp.reset_action_history, mode="reset"
    )

    # Velocity-based pushes for robustness training. Interval is overridden to
    # a shorter window in play mode for visibility -- see
    # MicroduckVelocity*EnvCfg_PLAY in microduck_velocity_env_cfg.py.
    push_robot: EventTerm | None = (
        EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        if ENABLE_VELOCITY_PUSHES
        else None
    )

    # Domain randomization -- re-sampled per episode at reset. mjlab 1.3.0's
    # stock dr.* ops with operation="add"/"scale" read from the compile-time
    # default field each reset (Operation.uses_defaults=True), so they are
    # NON-accumulating natively.
    randomize_com: EventTerm | None = (
        EventTerm(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )
        if ENABLE_COM_RANDOMIZATION
        else None
    )
    randomize_head_com: EventTerm | None = (
        EventTerm(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )
        if ENABLE_HEAD_COM_RANDOMIZATION
        else None
    )
    randomize_motor_gains: EventTerm | None = (
        EventTerm(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": _kp_range,
                "kd_range": _kd_range,
            },
        )
        if (ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION)
        else None
    )
    # Physics-consistent mass + inertia randomization via mjlab's
    # pseudo_inertia: alpha scales BOTH mass and inertia by e^(2*alpha) with
    # the CoM unchanged (so it does NOT conflict with randomize_com).
    # alpha_range is derived from the +-5% mass scale range.
    randomize_mass_inertia: EventTerm | None = (
        EventTerm(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )
        if ENABLE_MASS_INERTIA_RANDOMIZATION
        else None
    )
    # Joint-friction DR under BAM: scales BAM's velocity-independent friction
    # budget (Coulomb + Stribeck + load) per-env via the FrictionDRBamActuator
    # friction_scale hook. MuJoCo's dof_frictionloss is zeroed under BAM, so
    # the stock dr.dof_frictionloss is a no-op -- this is the BAM-native path.
    randomize_joint_friction: EventTerm | None = (
        EventTerm(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )
        if ENABLE_JOINT_FRICTION_RANDOMIZATION
        else None
    )
    # No-op under BAM (dof_damping zeroed in edit_spec); only affects the XML
    # position actuator.
    randomize_joint_damping: EventTerm | None = (
        EventTerm(
            func=microduck_mdp.randomize_dof_field_scaled,
            mode="reset",
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "field": "dof_damping",
                "scale_range": JOINT_DAMPING_RANDOMIZATION_RANGE,
            },
        )
        if ENABLE_JOINT_DAMPING_RANDOMIZATION
        else None
    )
    # Reflected rotor inertia (armature), microban-exact (dr.joint_armature,
    # scale, +-10%). Non-accumulating (uses_defaults). DOES affect the BAM
    # actuator -- BAM sets dof_armature (~0.0018), it isn't zeroed.
    randomize_armature: EventTerm | None = (
        EventTerm(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )
        if ENABLE_ARMATURE_RANDOMIZATION
        else None
    )
    # IMU orientation randomization (mounting error) is applied at the
    # OBSERVATION level instead (see microduck_observations_cfg.py) -- the old
    # event-based randomize_imu_orientation wrote site_quat, which under mjlab
    # 1.3.0 is neither per-env expanded nor read by these obs -- a no-op.
    randomize_base_orientation: EventTerm | None = (
        EventTerm(
            func=microduck_mdp.randomize_base_orientation,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "max_pitch_deg": BASE_ORIENTATION_MAX_PITCH_DEG,
                "max_roll_deg": BASE_ORIENTATION_MAX_ROLL_DEG,
            },
        )
        if ENABLE_BASE_ORIENTATION_RANDOMIZATION
        else None
    )

    # Encoder-bias DR: samples a per-env constant joint-encoder offset. The
    # actor's joint_pos obs reads it (biased=True); the critic stays
    # unbiased/privileged (biased=False) -- see microduck_observations_cfg.py.
    encoder_bias: EventTerm | None = (
        EventTerm(
            mode="startup",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": ENCODER_BIAS_RANGE},
        )
        if ENABLE_ENCODER_BIAS
        else None
    )
