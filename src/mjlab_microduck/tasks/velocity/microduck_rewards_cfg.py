"""Reward specifications for the Microduck velocity task."""

import math

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.utils.configclass import configclass

from .cfg.rewards_cfg import RewardsCfg
from .microduck_scene_cfg import SITE_NAMES

# Pose reward std (deg-of-freedom deviation tolerance), tighter when standing.
_STD_STANDING = {
    r".*hip_yaw.*": 0.1,
    r".*hip_roll.*": 0.05,  # 0.1->0.06->0.05 -- hold the 5deg-inward stance (sole sits flat), stop leg splay
    r".*hip_pitch.*": 0.15,
    r".*knee.*": 0.15,
    r".*ankle.*": 0.1,
}
_STD_WALKING = {
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.05,  # 0.1->0.06->0.05 -- hold the 5deg-inward stance, stop the leg splay to vertical
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.25,  # was 0.15
}

# Pose reward operates on LEG joints only. Head/neck are command-driven
# (head_pose_tracking) -- if they were in this reward too, it would pull them
# to HOME while head_pose_tracking pulls them to the command, and the policy
# converges to "ignore the command" because pose reward dominates once
# head_pose_tracking's gradient dies at large commands.
_POSE_ASSET_CFG = SceneEntityCfg("robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",))


@configclass
class MicroduckRewardsCfg(RewardsCfg):
    """Reward terms for the Microduck velocity task."""

    pose: RewTerm | None = RewTerm(
        func=mdp.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": _POSE_ASSET_CFG,
            "command_name": "twist",
            "std_standing": _STD_STANDING,
            "std_walking": _STD_WALKING,
            "std_running": _STD_WALKING,
            "walking_threshold": 0.01,
            "running_threshold": 1.5,
        },
    )

    # Deliberately strong (2.0 / std^2=0.05, was 1.0 / std^2=0.1). 2026-07
    # pitch-vs-speed eval: the policy walks with a +2-4 deg steady forward
    # lean (p90 ~6-8 deg) and ~2/3 of push-induced falls at speed are FORWARD.
    # At weight 1.0 / std^2=0.1 a 4 deg lean cost ~0.05/step -- effectively
    # free. At 2.0 / std^2=0.05 it costs ~0.19/step: enough gradient to hold
    # the trunk level in steady gait while transient lean (push recovery,
    # accel) stays affordable.
    upright: RewTerm | None = RewTerm(
        func=mdp.upright,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    foot_clearance: RewTerm | None = RewTerm(
        func=mdp.feet_clearance,
        weight=-2.0,
        params={
            "target_height": 0.02,  # Increased from 0.01 to penalize dragging
            "height_sensor_name": "foot_height_scan",
            "command_name": "twist",
            "command_threshold": 0.01,
            "asset_cfg": SceneEntityCfg("robot", site_names=SITE_NAMES),
        },
    )
    foot_swing_height: RewTerm | None = RewTerm(
        func=mdp.feet_swing_height,
        weight=-0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "height_sensor_name": "foot_height_scan",
            "target_height": 0.02,  # Increased from 0.01 to force foot lifting
            "command_name": "twist",
            "command_threshold": 0.01,
        },
    )
    # foot_slip deliberately weak (-0.1, not -1.0): -1.0 was too restrictive
    # for this robot's pivot-heavy turning.
    foot_slip: RewTerm | None = RewTerm(
        func=mdp.feet_slip,
        weight=-0.1,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.01,
            "asset_cfg": SceneEntityCfg("robot", site_names=SITE_NAMES),
        },
    )
    # soft_landing dropped entirely for this task.
    soft_landing: RewTerm | None = None

    # air_time window [0.125, 0.300] s. NOTE: standing still at zero command is
    # taught by the standing_envs curriculum (-> 25% standing envs by ~iter
    # 2000), not by an explicit stillness/no-stepping term.
    air_time: RewTerm | None = RewTerm(
        func=mdp.feet_air_time,
        weight=3.0,
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.125,
            "threshold_max": 0.300,
            "command_name": "twist",
            "command_threshold": 0.01,
        },
    )

    body_ang_vel: RewTerm | None = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    angular_momentum: RewTerm | None = RewTerm(
        func=mdp.angular_momentum_penalty,
        weight=-0.02,
        params={"sensor_name": "robot/root_angmom"},
    )

    track_linear_velocity: RewTerm | None = RewTerm(
        func=mdp.track_linear_velocity,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.1)},
    )
    track_angular_velocity: RewTerm | None = RewTerm(
        func=mdp.track_angular_velocity,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.5)},
    )

    # Action smoothness: stage-0 value; the action_rate_weight curriculum (see
    # microduck_curriculum_cfg.py) ramps it -0.1 -> -1.0 by iter 1500.
    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # Self-collision penalty: discourages legs from crashing into the trunk
    # battery holder (the self_collision_only-classed geoms on leg, leg_2,
    # battery_holder). With proper joint-range limits the policy can't actually
    # reach the body, but a positive signal here keeps it well clear.
    self_collisions: RewTerm | None = RewTerm(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )

    # === Pose tracking rewards ===
    # head_pose: primary objective in vel env -- the whole point of the
    # rewrite. std=0.5 with per-joint Gaussian (see head_pose_tracking in
    # mdp.py): at the full +-1.0 rad command, a non-tracking policy still sees
    # per-joint reward exp(-(1/0.5)^2)=exp(-4)~0.018 -- a small but non-zero
    # gradient -- so the curriculum widening doesn't kill the signal. Final
    # reward is the mean over 4 joints, so partial tracking is partial reward
    # (no all-or-nothing).
    head_pose_tracking: RewTerm | None = RewTerm(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    # body_pose: infra kept intact but DISABLED (weight 0) -- the obs slot and
    # command stay alive for envs that raise the weight (standup).
    body_pose_tracking: RewTerm | None = RewTerm(
        func=microduck_mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )
    # Head droop fix (2026-08-20). The head walks pitched ~15 deg down
    # (measured: run ww1g2198 head_pose_tracking 1.544/2.0 -> 14.6 deg mean
    # joint error). DO NOT fix this by tightening head_pose_tracking's std:
    # run 5yay13u4 tried fine_std=0.1 and the policy stopped walking entirely
    # by iter 300 (air_time 1.01 -> 0.02, peak foot height 15 mm -> 2 mm,
    # entropy collapsed 10.9 -> 1.9). An instantaneous tight tolerance taxes
    # walking 0.77/step -- 76% of the whole air_time reward -- and is
    # UNESCAPABLE, since a 280 g head (38% of robot mass) must oscillate
    # while stepping. Standing still scored higher, so it stood still.
    # The DC bias, unlike the oscillation, IS escapable (bias the neck
    # command up to cancel gravity sag), so price only that: L1 on a 1 s EMA
    # of the error. At the optimum this costs a walking policy nothing.
    # Weight ramped 0.0 -> 3.0 by the head_pose_bias_weight curriculum.
    head_pose_bias: RewTerm | None = RewTerm(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,
        params={"command_name": "head_pose", "tau_s": 1.0},
    )
