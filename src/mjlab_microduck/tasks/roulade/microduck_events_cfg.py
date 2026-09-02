"""Event specifications for the Microduck roulade task.

Field declaration order matters here the same way it does for other tasks
(see ``ball_kick``'s ``microduck_events_cfg.py`` docstring): ``set_roulade_
state`` must be declared (and therefore run) after ``reset_robot_joints``,
since a mid-roll spawn's tuck lerps FROM the HOME pose that
``reset_robot_joints`` writes. It's declared after all inherited fields
(which already include ``reset_robot_joints``), so this is automatic.
"""

import math

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.roulade.microduck_flags import (
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
    ENCODER_BIAS_RANGE,
    HEAD_COM_RANDOMIZATION_RANGE,
    JOINT_FRICTION_RANDOMIZATION_RANGE,
    KD_RANDOMIZATION_RANGE,
    KP_RANDOMIZATION_RANGE,
    MASS_INERTIA_RANDOMIZATION_RANGE,
    MIDROLL_OMEGA_RANGE,
    MIDROLL_PITCH_MAX,
    MIDROLL_PITCH_MIN,
    ROULADE_FORWARD_VEL_RANGE,
    TUCK_OVERRIDES,
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
    """Event terms for the Microduck roulade task."""

    foot_friction: EventTerm | None = EventTerm(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_FRICTION_GEOM_NAMES),
            "operation": "abs",
            "ranges": (0.7, 1.3),
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
    # A push mid-roll is incoherent -- dropped entirely (ENABLE_VELOCITY_PUSHES
    # is False for this task; unlike other tasks there's no conditional
    # push_robot event at all).
    push_robot: EventTerm | None = None

    # -- New event terms (appended after all inherited fields above, in this order) --
    expand_bam_friction_fields: EventTerm | None = EventTerm(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup"
    )
    reset_action_history: EventTerm | None = EventTerm(func=microduck_mdp.reset_action_history, mode="reset")
    # Standing start + mid-roll reverse-curriculum spawns; also resets the
    # rotation accumulator. MUST run after reset_robot_joints -- see this
    # file's docstring.
    set_roulade_state: EventTerm | None = EventTerm(
        func=microduck_mdp.reset_roulade_state,
        mode="reset",
        params={
            "standing_prob": 0.5,
            "midroll_prob": 0.5,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
            "standing_tilt_max": math.radians(5.0),
            "forward_vel_range": ROULADE_FORWARD_VEL_RANGE,
            "midroll_pitch_min": MIDROLL_PITCH_MIN,
            "midroll_pitch_max": MIDROLL_PITCH_MAX,
            "midroll_z_min": 0.05,
            "midroll_z_max": 0.10,
            "midroll_omega_range": MIDROLL_OMEGA_RANGE,
            "tuck_overrides": TUCK_OVERRIDES,
            "tuck_factor_range": (0.3, 1.0),
            "joint_noise_std": 0.08,
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
