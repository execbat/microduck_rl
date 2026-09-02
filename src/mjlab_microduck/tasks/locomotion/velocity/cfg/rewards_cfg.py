"""Reward specifications for the velocity locomotion task."""

import math

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.utils.configclass import configclass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_linear_velocity: RewTerm | None = RewTerm(
        func=mdp.track_linear_velocity,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
    )
    track_angular_velocity: RewTerm | None = RewTerm(
        func=mdp.track_angular_velocity,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.5)},
    )
    upright: RewTerm | None = RewTerm(
        func=mdp.upright,
        weight=1.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
        },
    )
    pose: RewTerm | None = RewTerm(
        func=mdp.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "command_name": "twist",
            "std_standing": {},  # Set per-robot.
            "std_walking": {},  # Set per-robot.
            "std_running": {},  # Set per-robot.
            "walking_threshold": 0.05,
            "running_threshold": 1.5,
        },
    )
    body_ang_vel: RewTerm | None = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=0.0,  # Override per-robot.
        params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set per-robot.
    )
    angular_momentum: RewTerm | None = RewTerm(
        func=mdp.angular_momentum_penalty,
        weight=0.0,  # Override per-robot.
        params={"sensor_name": "robot/root_angmom"},
    )
    dof_pos_limits: RewTerm | None = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    air_time: RewTerm | None = RewTerm(
        func=mdp.feet_air_time,
        weight=0.0,  # Override per-robot.
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.05,
            "threshold_max": 0.5,
            "command_name": "twist",
            "command_threshold": 0.5,
        },
    )
    foot_clearance: RewTerm | None = RewTerm(
        func=mdp.feet_clearance,
        weight=-2.0,
        params={
            "target_height": 0.1,
            "height_sensor_name": "foot_height_scan",
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        },
    )
    foot_swing_height: RewTerm | None = RewTerm(
        func=mdp.feet_swing_height,
        weight=-0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "height_sensor_name": "foot_height_scan",
            "target_height": 0.1,
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    )
    foot_slip: RewTerm | None = RewTerm(
        func=mdp.feet_slip,
        weight=-0.1,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        },
    )
    soft_landing: RewTerm | None = RewTerm(
        func=mdp.soft_landing,
        weight=-1e-5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    )
