"""Event specifications for the Microduck velocity_rollers task.

Field declaration order matters here the same way it does for ``ball_kick``
(see that task's ``microduck_events_cfg.py`` docstring) -- dataclass field
order determines dict insertion order, and mjlab runs same-mode events in
that order. New fields below are declared in the exact order the original
file added them.
"""

import math

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity.cfg.events_cfg import EventsCfg
from mjlab_microduck.tasks.velocity.microduck_flags import HEAD_BODY_NAMES
from mjlab_microduck.tasks.velocity_rollers.microduck_flags import (
    ARMATURE_RANDOMIZATION_RANGE,
    COM_RANDOMIZATION_RANGE,
    ENABLE_ARMATURE_RANDOMIZATION,
    ENABLE_COM_RANDOMIZATION,
    ENABLE_ENCODER_BIAS,
    ENABLE_HEAD_COM_RANDOMIZATION,
    ENABLE_JOINT_FRICTION_RANDOMIZATION,
    ENABLE_MASS_INERTIA_RANDOMIZATION,
    ENABLE_VELOCITY_PUSHES,
    ENABLE_WHEEL_FRICTION_RANDOMIZATION,
    ENCODER_BIAS_RANGE,
    HEAD_COM_RANDOMIZATION_RANGE,
    JOINT_FRICTION_RANDOMIZATION_RANGE,
    MASS_INERTIA_RANDOMIZATION_RANGE,
    VELOCITY_PUSH_INTERVAL_S,
    VELOCITY_PUSH_RANGE,
)
from mjlab_microduck.utils.configclass import configclass

_mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE


@configclass
class MicroduckEventsCfg(EventsCfg):
    """Event terms for the Microduck velocity_rollers task."""

    reset_base: EventTerm | None = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.1335, 0.1435),  # overridden for the roller's stand height.
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        },
    )
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
    # Wheels roll; ground friction lives in the XML (no geom-friction DR needed).
    foot_friction: EventTerm | None = None
    encoder_bias: EventTerm | None = (
        EventTerm(
            mode="startup",
            func=dr.encoder_bias,
            params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": ENCODER_BIAS_RANGE},
        )
        if ENABLE_ENCODER_BIAS
        else None
    )

    # -- New event terms (appended after all inherited fields above, in this order) --
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    expand_bam_friction_fields: EventTerm | None = EventTerm(
        func=microduck_mdp.expand_bam_friction_fields, mode="startup"
    )
    reset_action_history: EventTerm | None = EventTerm(func=microduck_mdp.reset_action_history, mode="reset")

    # Wheel-bearing friction DR: real bearings have a little drag; the XML
    # keeps frictionloss=0 for trainability and the curriculum ramps it in.
    randomize_wheel_friction: EventTerm | None = (
        EventTerm(
            func=dr.dof_frictionloss,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",)),
                "operation": "abs",
                "ranges": (0.000, 0.000),  # ramped up by wheel_friction_curriculum
            },
        )
        if ENABLE_WHEEL_FRICTION_RANDOMIZATION
        else None
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
            # Legs/head only -- the wheel bearings' tiny armature is excluded
            # (its DR is the frictionloss event above).
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )
        if ENABLE_ARMATURE_RANDOMIZATION
        else None
    )
