"""Event specifications for the Microduck standup task."""

import math

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.standup.microduck_flags import (
    ARMATURE_RANDOMIZATION_RANGE,
    COM_RANDOMIZATION_RANGE,
    ENABLE_ARMATURE_RANDOMIZATION,
    ENABLE_COM_RANDOMIZATION,
    ENABLE_ENCODER_BIAS,
    ENABLE_HEAD_COM_RANDOMIZATION,
    ENABLE_JOINT_FRICTION_RANDOMIZATION,
    ENABLE_KD_RANDOMIZATION,
    ENABLE_KP_RANDOMIZATION,
    ENABLE_MASS_INERTIA_RANDOMIZATION,
    ENABLE_VELOCITY_PUSHES,
    ENCODER_BIAS_RANGE,
    HEAD_COM_RANDOMIZATION_RANGE,
    JOINT_FRICTION_RANDOMIZATION_RANGE,
    KD_RANDOMIZATION_RANGE,
    KP_RANDOMIZATION_RANGE,
    MASS_INERTIA_RANDOMIZATION_RANGE,
    SITTING_JOINT_OVERRIDES,
    VELOCITY_PUSH_INTERVAL_S,
    VELOCITY_PUSH_RANGE,
)
from mjlab_microduck.tasks.velocity.cfg.events_cfg import EventsCfg
from mjlab_microduck.tasks.velocity.microduck_flags import HEAD_BODY_NAMES
from mjlab_microduck.utils.configclass import configclass

from .microduck_scene_cfg import FOOT_FRICTION_GEOM_NAMES

_mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
_kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
_kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)


@configclass
class MicroduckEventsCfg(EventsCfg):
    """Event terms for the Microduck standup task.

    ``reset_base``/``reset_robot_joints`` are inherited completely unchanged
    from the base ``EventsCfg`` -- the actual starting pose comes entirely
    from ``set_ground_state`` (standing / sitting / face-down / face-up,
    chosen per-env), which runs after them and overwrites the pose anyway.
    """

    foot_friction: EventTerm | None = EventTerm(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_FRICTION_GEOM_NAMES),
            "operation": "abs",
            "ranges": (0.7, 1.3),  # match velocity
            "shared_random": True,
        },
    )
    encoder_bias: EventTerm | None = (
        EventTerm(
            mode="startup",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": ENCODER_BIAS_RANGE},
        )
        if ENABLE_ENCODER_BIAS
        else None
    )
    # Interval is overridden to a shorter window in play mode -- see
    # MicroduckStandupFlatEnvCfg.__post_init__ / VELOCITY_PUSH_PLAY_INTERVAL_S.
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

    # -- New event terms (appended after all inherited fields above, in this order) --
    expand_bam_friction_fields: EventTerm | None = EventTerm(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup"
    )
    reset_action_history: EventTerm | None = EventTerm(func=microduck_mdp.reset_action_history, mode="reset")
    # Start from any of 4 poses; the ground_state_mix curriculum ramps this
    # mix easy->hard over training (see microduck_curriculum_cfg.py). The
    # values below are the INITIAL (stage-0) mix.
    set_ground_state: EventTerm | None = EventTerm(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.20,  # belly to floor (+90deg pitch)
            "face_up_prob": 0.00,  # back to floor (-90deg pitch) -- introduced late
            "sitting_prob": 0.40,  # sit keyframe (deployment hand-off)
            "standing_prob": 0.40,  # already upright at standing height
            # Prone reset height: trunk rests at ~0.044m face-down (measured),
            # so spawn just above the ground rather than free-falling.
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            # Partial-roll noise on face-up spawns (+-90deg about the body
            # long axis): near-on-side spawns put starts partway along the
            # roll -> a built-in reverse curriculum (the reward landscape
            # from flat supine to prone has no gradient until the roll
            # completes, so a pure supine start is seed-lucky).
            "face_up_roll_max": math.radians(90),
            "sitting_joint_overrides": SITTING_JOINT_OVERRIDES,
            "sitting_joint_noise_std": 0.12,  # ~= 7deg per joint
            "sitting_tilt_max": math.radians(10),
            # Seated equilibrium is SIT_Z=0.060 -- band is -1cm/+3cm around it.
            "sitting_z_min": 0.05,
            "sitting_z_max": 0.09,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )
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
    randomize_joint_friction: EventTerm | None = (
        EventTerm(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE},
        )
        if ENABLE_JOINT_FRICTION_RANDOMIZATION
        else None
    )
