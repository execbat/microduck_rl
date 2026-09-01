"""Event specifications for the Microduck BallKick task.

Field declaration order matters here: ``@configclass``/dataclass field order
determines dict insertion order (see ``group_to_dict``), and mjlab runs
same-mode events in that order. In particular ``reset_ball`` MUST be
declared after ``set_ground_state`` -- the ball position is derived from the
robot's final reset pose.
"""

import math

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.ball_kick.microduck_flags import (
    ARMATURE_RANDOMIZATION_RANGE,
    BALL_OFFSET_X,
    BALL_POS_NOISE_XY,
    BALL_RADIUS,
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
    ball_offset_y_of,
)
from mjlab_microduck.tasks.velocity.microduck_flags import HEAD_BODY_NAMES
from mjlab_microduck.utils.configclass import configclass
from mjlab_microduck.tasks.velocity.cfg.events_cfg import EventsCfg

from .microduck_scene_cfg import FOOT_FRICTION_GEOM_NAMES

_mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
_kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
_kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)


@configclass
class MicroduckEventsCfg(EventsCfg):
    """Event terms for the Microduck BallKick task."""

    reset_robot_joints: EventTerm | None = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # Joint noise on the standing start: deployment hands off from
            # the walk/velstand policy, whose settled stand won't match HOME
            # exactly.
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
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
    # to a shorter window in play mode -- see MicroduckBallKickEnvCfg.
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
    reset_action_history: EventTerm | None = EventTerm(
        func=microduck_mdp.reset_action_history, mode="reset"
    )
    # Standing-only start (reuses the standup env's ground-state machinery
    # for the noisy upright spawn: random yaw +- tilt noise, z near
    # equilibrium).
    set_ground_state: EventTerm | None = EventTerm(
        func=microduck_mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.0,
            "standing_prob": 1.0,
            "sitting_tilt_max": math.radians(5),  # +-5 deg pitch/roll on the stand
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )
    # Ball placement -- MUST come after set_ground_state (events run in
    # field/dict insertion order; the ball position derives from the final
    # robot pose). Also stores the per-env kick direction (robot heading at
    # reset). ``offset``'s y-component is filled in by
    # ``MicroduckBallKickEnvCfg.__post_init__`` once ``self.kick_foot`` is
    # known (see ``ball_offset_y_of``).
    reset_ball: EventTerm | None = EventTerm(
        func=microduck_mdp.reset_ball_in_front_of_foot,
        mode="reset",
        params={
            "offset": (BALL_OFFSET_X, ball_offset_y_of("right")),  # y fixed up per kick_foot in __post_init__
            "noise_xy": BALL_POS_NOISE_XY,
            "ball_radius": BALL_RADIUS,
            "asset_name": "ball",
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
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )
        if ENABLE_JOINT_FRICTION_RANDOMIZATION
        else None
    )
