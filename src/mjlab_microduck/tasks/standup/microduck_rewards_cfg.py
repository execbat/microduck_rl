"""Reward specifications for the Microduck standup task.

Single fixed target (STAND = HOME pose + STAND_Z), active from t=0. No
trajectory, no waypoints, no episode-progress gating -- the policy is free
to discover any rise path that satisfies: (1) end-state matches HOME +
STAND_Z, (2) the rise is gentle (low |a_z| throughout), (3) the trunk stays
upright throughout, (4) joint/action motion stays smooth.

2026-07 TRANSFER FIX (violent/shaky on the real robot): ALL task weights
below are the /4-rescaled values so the total task mass matches velocity's,
and the shared sim2real regularisers act at the same RELATIVE strength as
in the well-transferring velocity env. Internal ratios between task terms
are unchanged (uniform scaling) -- multiply the absolute numbers below by 4
to recover the original per-term rationale's units. PPO normalises
advantages, so the global scale itself doesn't matter; only the
task<->regulariser ratio does.

WARNING preserved from the original: ``gentle_rise`` uses a POSITIVE weight
deliberately -- ``trunk_vertical_accel_penalty`` ALREADY returns ``-|a_z|``.
A negative weight here double-negates into a reward for vertical shocks --
the same sign bug ``roller_standup`` found and fixed in its ``gentle_rise``,
and confirmed again on a sitstand run whose ``Episode_Reward/gentle_motion``
logged positive.
"""

import math

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.standup.microduck_flags import ENABLE_BODY_CONTROL, LEG_JOINTS, STAND_Z
from mjlab_microduck.tasks.velocity.cfg.rewards_cfg import RewardsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckRewardsCfg(RewardsCfg):
    """Reward terms for the Microduck standup task.

    ``dof_pos_limits`` is kept inherited unchanged from the base
    ``RewardsCfg``. ``upright`` is dropped -- standup uses its own
    ``upright_linear``/``upright_sharp`` instead.
    """

    track_linear_velocity: RewTerm | None = None
    track_angular_velocity: RewTerm | None = None
    air_time: RewTerm | None = None
    foot_clearance: RewTerm | None = None
    foot_swing_height: RewTerm | None = None
    foot_slip: RewTerm | None = None
    pose: RewTerm | None = None
    soft_landing: RewTerm | None = None
    upright: RewTerm | None = None

    # -- Pose target -- legs+hips+knees+ankles. target_overrides=None -> HOME. --
    pose_stand_legs: RewTerm | None = RewTerm(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={"std": 0.5, "joint_indices": LEG_JOINTS, "target_overrides": None},
    )
    # Head pose tracking (commandable head control, like velocity). Replaces
    # a fixed pose_stand_neck reward -- the neck/head are steered by the
    # head_pose command instead, so no reward fights its gradient.
    head_pose_tracking: RewTerm | None = RewTerm(
        func=microduck_mdp.head_pose_tracking, weight=0.75, params={"command_name": "head_pose", "std": 0.5}
    )
    # Head DC-droop penalty. L1 on a 1s EMA of the head tracking error --
    # prices only the sustained gravity sag the policy can cancel by biasing
    # the neck command up; transient motion averages out. Gated on
    # height+tilt (same values as arrival_damping) so the ground/rising
    # phase accumulates NOTHING; starts at 0, introduced by curriculum.
    head_pose_bias: RewTerm | None = RewTerm(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,  # ramped by head_pose_bias_weight curriculum
        params={
            "command_name": "head_pose",
            "tau_s": 1.0,
            "gate_height_low": 0.09,
            "gate_height_high": 0.11,
            "gate_tilt_full_deg": 20.0,
            "gate_tilt_zero_deg": 45.0,
        },
    )
    # L1 bootstrap -- constant gradient even when far from HOME. Legs only
    # -- neck/head are steered by head_pose_tracking.
    pose_stand_l1: RewTerm | None = RewTerm(
        func=microduck_mdp.pose_l1_penalty,
        weight=1.25,
        params={"joint_indices": LEG_JOINTS, "target_overrides": None},
    )
    # Trunk height -- two-layer Gaussian: wide std for bootstrap reach from
    # sit, sharp std for a real gradient in the final cm (the wide layer
    # alone saturates near the target and leaves no pull for the last cm).
    height_stand: RewTerm | None = RewTerm(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={"std": 0.04, "target_height": STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    height_stand_sharp: RewTerm | None = RewTerm(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={"std": 0.015, "target_height": STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    height_stand_l1: RewTerm | None = RewTerm(
        func=microduck_mdp.height_l1_penalty,
        weight=7.5,
        params={"target_height": STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # Reward upward CoM velocity below STAND_Z -- pays for the *motion* of
    # rising, not just the destination (destination-only rewards leave
    # "stay seated collecting most-of-pose+upright" as the dominant local
    # optimum). Gates off above max_height so the policy can't farm it by
    # bobbing. NO max_vz cap: capping the rewarded rise speed shrinks the
    # payoff of noisy recovery ATTEMPTS during discovery, and face-up/
    # face-down recovery stops being learned. Smoothing instead comes from
    # the LATE-phased penalty curricula (arrival_damping etc.) below.
    com_upward_velocity: RewTerm | None = RewTerm(
        func=microduck_mdp.com_upward_velocity,
        weight=0.75,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "max_height": 0.125},
    )
    # Gentle rise -- penalty on |a_z|. GLOBAL (not phase-gated): prone flips
    # pay it in full (impacts + push-off are |a_z| spikes). See the
    # module-level sign-convention warning.
    gentle_rise: RewTerm | None = RewTerm(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.005,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # Arrival damper -- trunk omega_xy^2, gated on height AND tilt. Targets
    # the real-robot failure loop: rise -> overshoot vertical -> tip ->
    # retry. STARTS AT WEIGHT 0, introduced by curriculum only after the
    # recovery skills exist (any attempt-tax during discovery prevents the
    # flip from ever being found -- proven twice).
    arrival_damping: RewTerm | None = RewTerm(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low": 0.09,
            "height_high": 0.11,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # Upright -- two-layer: a linear floor with strong gradient at high tilt
    # (bootstrap pull from any orientation, e.g. inverted at recovery
    # start), and a sharp Gaussian (gated by trunk z, so it only pays at
    # standing height) whose gradient is strongest near-vertical, where the
    # linear term runs out of steam.
    upright_linear: RewTerm | None = RewTerm(
        func=microduck_mdp.body_upright_linear,
        weight=1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    upright_sharp: RewTerm | None = RewTerm(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=1.5,
        params={
            "std": 0.3,
            "height_low": 0.060,  # SIT_Z
            "height_high": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # Smooth multiplicative goal-state score (broad stds keep gradient
    # visible far from the goal, e.g. at a partial lean-basin recovery).
    standing_composite: RewTerm | None = RewTerm(
        func=microduck_mdp.standing_composite_score,
        weight=3.75,
        params={
            "target_height": STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": LEG_JOINTS,  # neck/head steered by head_pose_tracking.
            "target_overrides": None,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # Body pose tracking -- z/roll/pitch only (axis_weights), the runtime
    # body-control axes. Locomotion variant (not the 6D one) so the unused
    # x/y axes don't reference the spawn origin, which the robot leaves
    # during prone flips. Weight starts at 0, ramped in by curriculum after
    # ground_state_mix finishes -- while prone/rising the reward is ~=0 on
    # all tracked axes, so before the robot stands it's just another
    # standing attractor, not a motion penalty that could tax flip/rise attempts.
    body_pose_tracking: RewTerm | None = (
        RewTerm(
            func=microduck_mdp.body_pose_tracking_locomotion,
            weight=0.0,
            params={
                "command_name": "body_pose",
                "nominal_height": STAND_Z,
                "z_std": 0.01,
                "angle_std": math.radians(5),
                "axis_weights": (0.0, 0.0, 1.0, 1.0, 1.0, 0.0),
                "vel_gate_command_name": None,
            },
        )
        if ENABLE_BODY_CONTROL
        else None
    )

    # -- Sim2real regularisers -- MATCHED to velocity ---------------------------
    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_torque_rate_l2: RewTerm | None = RewTerm(func=microduck_mdp.joint_torque_rate_l2, weight=0.0)
    body_ang_vel: RewTerm | None = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,  # motion-blocker: kept LIGHT (velocity value).
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    angular_momentum: RewTerm | None = RewTerm(
        func=mdp.angular_momentum_penalty, weight=-0.02, params={"sensor_name": "robot/root_angmom"}
    )
    self_collisions: RewTerm | None = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision"}
    )
