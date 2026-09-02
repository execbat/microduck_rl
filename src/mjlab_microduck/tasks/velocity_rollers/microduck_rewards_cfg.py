"""Reward specifications for the Microduck velocity_rollers task."""

import math

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity.cfg.rewards_cfg import RewardsCfg
from mjlab_microduck.utils.configclass import configclass

# passive_.*: 999.0 -> passive wheel joints are matched but effectively ignored.
_STD_STANDING = {
    r".*hip_yaw.*": 0.05,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.05,
    r".*knee.*": 0.05,
    r".*ankle.*": 0.05,
    r".*neck.*": 0.05,
    r".*head.*": 0.05,
    r".*passive_.*": 999.0,
}
_STD_WALKING = {
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.6,  # loosened: skating requires wide lateral push.
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.25,
    r".*neck.*": 0.05,
    r".*head.*": 0.05,
    r".*passive_.*": 999.0,
}
_STD_RUNNING = {
    r".*hip_yaw.*": 0.5,
    r".*hip_roll.*": 0.8,  # loosened: skating requires wide lateral push.
    r".*hip_pitch.*": 0.8,
    r".*knee.*": 0.8,
    r".*ankle.*": 0.5,
    r".*neck.*": 0.05,
    r".*head.*": 0.05,
    r".*passive_.*": 999.0,
}


@configclass
class MicroduckRewardsCfg(RewardsCfg):
    """Reward terms for the Microduck velocity_rollers task.

    Only ``pose``/``upright``/``body_ang_vel``/``angular_momentum``/
    ``action_rate_l2`` are kept from the base ``RewardsCfg`` (re-declared
    below with roller-specific weights/params); everything else in the base
    is dropped -- this is a skating gait, not a walk.
    """

    track_linear_velocity: RewTerm | None = None
    track_angular_velocity: RewTerm | None = None
    dof_pos_limits: RewTerm | None = None
    air_time: RewTerm | None = None
    foot_clearance: RewTerm | None = None
    foot_swing_height: RewTerm | None = None
    foot_slip: RewTerm | None = None
    soft_landing: RewTerm | None = None

    pose: RewTerm | None = RewTerm(
        func=mdp.variable_posture,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "command_name": "twist",
            "std_standing": _STD_STANDING,
            "std_walking": _STD_WALKING,
            "std_running": _STD_RUNNING,
            "walking_threshold": 0.01,
            "running_threshold": 0.5,
        },
    )
    upright: RewTerm | None = RewTerm(
        func=mdp.upright,
        weight=2.0,
        params={"std": math.sqrt(0.2), "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    body_ang_vel: RewTerm | None = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    angular_momentum: RewTerm | None = RewTerm(
        func=mdp.angular_momentum_penalty, weight=-0.02, params={"sensor_name": "robot/root_angmom"}
    )
    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-1.0)

    com_height_target: RewTerm | None = RewTerm(
        func=microduck_mdp.com_height_target,
        weight=2.0,
        params={"target_height_min": 0.0935, "target_height_max": 0.1235},
    )
    self_collisions: RewTerm | None = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision"}
    )
    # Gated to the STANCE foot only (sensor_name) so lifting the swing foot is
    # no longer punished -- the old ungated -5.0 was minimised by keeping
    # both blades flat on the ground (the swizzle) and actively fought the
    # stride. Weight also softened -5.0 -> -2.0 to leave room for a slightly
    # angled push.
    feet_flat: RewTerm | None = RewTerm(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    neck_action_rate_l2: RewTerm | None = RewTerm(func=microduck_mdp.neck_action_rate_l2, weight=-0.5)
    neck_joint_pos_l2: RewTerm | None = RewTerm(func=microduck_mdp.neck_joint_pos_l2, weight=-0.5)
    joint_torques_l2: RewTerm | None = RewTerm(func=microduck_mdp.joint_torques_l2, weight=-1e-3)
    # Deter OVER-COMMANDING a joint past its hard stop (policy-side, transfers
    # via the ONNX). hip_roll's +-0.38 rad limit vs the +-10 rad ctrlrange let
    # the low-kp servo be commanded far past the stop and slam it with max
    # torque -- a fragile sim-only trick. This penalises only the COMMAND
    # beyond (limit + 0.3 overshoot), so the joint keeps its full reachable
    # range while the wild over-drive is discouraged.
    action_over_limit: RewTerm | None = RewTerm(
        func=microduck_mdp.action_over_limit_penalty,
        weight=-0.5,
        params={"action_name": "joint_pos", "overshoot": 0.3},
    )
    # Pull hip_roll back toward neutral so the stance stops resting splayed on
    # the hip_roll limits. L1 = constant gradient: gently closes the legs AT
    # REST without preventing the lateral push stroke.
    hip_roll_neutral: RewTerm | None = RewTerm(
        func=microduck_mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*hip_roll.*",))},
    )
    # Sole positive task reward -- robot must spin wheels to get anything.
    # vel_scale 0.3 = the tanh target speed, saturating near the achievable
    # measured max (~0.33 m/s) rather than over-driving toward an unreachable
    # target.
    wheel_speed: RewTerm | None = RewTerm(
        func=microduck_mdp.wheel_speed_reward,
        weight=10.0,
        params={"command_name": "twist", "vel_scale": 0.3},
    )
    # Brake: reward stopping when cmd_x < 0. Silent at cmd_x >= 0 (coast/push).
    braking: RewTerm | None = RewTerm(
        func=microduck_mdp.braking_reward,
        weight=1.0,
        params={"command_name": "twist", "vel_std": 0.3},
    )
    # Air time during push: pay the recovery-foot lift, but ONLY when the
    # body is actually moving forward (vel_gate_ref). threshold_min=0.25 (via
    # threshold_min/max window (0.15, 0.45)) forbids ultra-short swings.
    skating_air_time: RewTerm | None = RewTerm(
        func=microduck_mdp.skating_air_time_reward,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "threshold_min": 0.15,
            "threshold_max": 0.45,
            "vel_gate_ref": 0.2,
        },
    )
    # Glide phase (single-support required): reward coasting on one blade
    # with quiet legs so the policy commits to each stroke instead of
    # kicking frantically.
    glide: RewTerm | None = RewTerm(
        func=microduck_mdp.glide_reward,
        weight=4.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "vel_ref": 0.2},
    )
    # Single-support stride vs double-support swizzle: rewards exactly-one-
    # blade-down and penalises both-down while pushing -- the core
    # anti-swizzle signal. Gated on forward speed too.
    single_support: RewTerm | None = RewTerm(
        func=microduck_mdp.single_support_reward,
        weight=3.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "vel_gate_ref": 0.2},
    )
    # Balance left/right leg usage (symmetry augmentation is OFF, so nothing
    # else stops a lopsided stride). Penalises cumulative swing-time
    # imbalance |L-R|/(L+R).
    gait_symmetry: RewTerm | None = RewTerm(
        func=microduck_mdp.gait_symmetry_penalty, weight=-1.0, params={"sensor_name": "feet_ground_contact"}
    )
    # Slight forward lean when pushing, to counteract backward torque.
    forward_lean: RewTerm | None = RewTerm(
        func=microduck_mdp.forward_lean_reward,
        weight=1.5,
        params={"command_name": "twist", "target_pitch": 0.262, "std": 0.1},
    )
    # Heading command is DISABLED (straight-line focus); heading_hold rewards
    # the yaw ANGLE staying near the spawn heading so it doesn't drift.
    heading_hold: RewTerm | None = RewTerm(
        func=microduck_mdp.heading_hold_reward,
        weight=1.0,
        params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )
