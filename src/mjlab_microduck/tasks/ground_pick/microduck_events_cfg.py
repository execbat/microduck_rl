"""Event specifications for the Microduck ground_pick task.

Field declaration order matters (see ``ball_kick``'s ``microduck_events_cfg.py``
docstring for why) -- new fields below are declared in the exact order the
original file added them.
"""

import math

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.ground_pick.microduck_flags import (
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
    VELOCITY_PUSH_INTERVAL_S,
    VELOCITY_PUSH_RANGE,
)
from mjlab_microduck.tasks.locomotion.velocity.cfg.events_cfg import EventsCfg
from mjlab_microduck.tasks.velocity.microduck_flags import HEAD_BODY_NAMES
from mjlab_microduck.utils.configclass import configclass

from .microduck_scene_cfg import FOOT_FRICTION_GEOM_NAMES

_mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
_kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
_kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)


@configclass
class MicroduckEventsCfg(EventsCfg):
    """Event terms for the Microduck ground_pick task."""

    reset_base: EventTerm | None = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.12, 0.13),  # overridden for the pick's stand height.
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
    # Velocity-based pushes for robustness training. Interval is overridden
    # to a shorter (spaced-out) window in play mode -- see the env cfg
    # __post_init__ / VELOCITY_PUSH_PLAY_INTERVAL_S.
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
    # Random "payload in the mouth" per episode (10-40g), applied on the rise
    # by the mouth_payload_force reward hook. Imagines the robot lifting an object.
    sample_mouth_payload: EventTerm | None = EventTerm(
        func=microduck_mdp.sample_mouth_payload, mode="reset", params={"min_kg": 0.01, "max_kg": 0.04}
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
    # Dormant (KP/KD off, like velocity). NOTE: randomize_delayed_actuator_gains
    # predates canonical BAM; only enable after porting it to BamActuator.set_gains.
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
