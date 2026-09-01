"""Event (reset / domain-randomization) specifications for the velocity task."""

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg as EventTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.utils.configclass import configclass


@configclass
class EventsCfg:
    """Event terms for the MDP."""

    reset_base: EventTerm | None = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.01, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        },
    )
    reset_robot_joints: EventTerm | None = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    )
    push_robot: EventTerm | None = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
        },
    )
    foot_friction: EventTerm | None = EventTerm(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
            "operation": "abs",
            "ranges": (0.3, 1.2),
            "shared_random": True,
        },
    )
    encoder_bias: EventTerm | None = EventTerm(
        mode="startup",
        func=dr.encoder_bias,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "bias_range": (-0.015, 0.015),
        },
    )
    base_com: EventTerm | None = EventTerm(
        mode="startup",
        func=dr.body_com_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            "operation": "add",
            "ranges": {
                0: (-0.025, 0.025),
                1: (-0.025, 0.025),
                2: (-0.03, 0.03),
            },
        },
    )
