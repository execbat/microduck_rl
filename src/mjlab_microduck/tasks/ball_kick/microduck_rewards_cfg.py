"""Reward specifications for the Microduck BallKick task."""

import math

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.ball_kick.microduck_flags import BALL_TARGET_SPEED, LEG_JOINTS, NECK_JOINTS, STAND_Z
from mjlab_microduck.utils.configclass import configclass
from mjlab_microduck.tasks.velocity.cfg.rewards_cfg import RewardsCfg


@configclass
class MicroduckRewardsCfg(RewardsCfg):
    """Reward terms for the Microduck BallKick task."""

    # -- Drop walking-specific terms (see the base RewardsCfg for the ones
    # not mentioned here at all -- upright/body_ang_vel/angular_momentum/
    # dof_pos_limits/action_rate_l2 are kept, just re-declared below where a
    # weight/param differs from the base default). --------------------------
    track_linear_velocity: RewTerm | None = None
    track_angular_velocity: RewTerm | None = None
    air_time: RewTerm | None = None
    foot_clearance: RewTerm | None = None
    foot_swing_height: RewTerm | None = None
    foot_slip: RewTerm | None = None
    pose: RewTerm | None = None  # gait-conditioned; replaced by pose_stand_* below.
    soft_landing: RewTerm | None = None

    # -- Kick objective: TARGET speed, not max speed ------------------------
    # Two-sided landscape peaking at BALL_TARGET_SPEED:
    #   - ball_forward_velocity, linear and CAPPED at the target: dense
    #     bootstrap gradient from the first touch. Weight 12.0 = 3.0/target so
    #     the at-target payoff stays ~= +3/step.
    #   - ball_speed_overshoot_penalty (weight -4.0): each m/s above target
    #     costs -4/step while it persists -- needed because the cap alone
    #     does NOT tame the kick (a harder kick keeps the ball at the cap for
    #     longer, so total reward would still grow with strike speed).
    ball_forward_velocity: RewTerm | None = RewTerm(
        func=microduck_mdp.ball_forward_velocity,
        weight=12.0,
        params={"asset_name": "ball", "max_speed": BALL_TARGET_SPEED},
    )
    ball_speed_overshoot: RewTerm | None = RewTerm(
        func=microduck_mdp.ball_speed_overshoot_penalty,
        weight=-4.0,
        params={"asset_name": "ball", "target_speed": BALL_TARGET_SPEED},
    )

    # Support foot: binary +1 while the non-kicking foot touches the ground.
    # Always-on anti-hop -- swinging the kicking leg is free, lifting the
    # support foot costs this every step.
    support_foot_grounded: RewTerm | None = RewTerm(
        func=microduck_mdp.single_foot_grounded_reward,
        weight=2.0,
        params={"sensor_name": "support_foot_ground_contact"},
    )

    # -- Stand cleanly before/after the kick ---------------------------------
    # Legs at HOME. std=0.5 is deliberately loose: the kick itself is a big
    # transient leg deviation and must stay affordable.
    pose_stand_legs: RewTerm | None = RewTerm(
        func=microduck_mdp.pose_target_match,
        weight=2.0,
        params={"std": 0.5, "joint_indices": LEG_JOINTS, "target_overrides": None},
    )
    # Neck/head at HOME (no head command in this task; tighter std -- the
    # head takes no part in the kick).
    pose_stand_neck: RewTerm | None = RewTerm(
        func=microduck_mdp.pose_target_match,
        weight=1.0,
        params={"std": 0.3, "joint_indices": NECK_JOINTS, "target_overrides": None},
    )

    # Upright -- velocity's exact recipe (weight 2.0, std^2=0.05).
    upright: RewTerm | None = RewTerm(
        func=mdp.upright,
        weight=2.0,
        params={"std": math.sqrt(0.05), "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Trunk at standing height -- discourages crouching/squatting as a kick prep.
    height_stand: RewTerm | None = RewTerm(
        func=microduck_mdp.height_target_gaussian,
        weight=1.0,
        params={
            "std": 0.04,
            "target_height": STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # -- Sim2real regularisers -- velocity parity (see standup env rationale) --
    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    body_ang_vel: RewTerm | None = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    angular_momentum: RewTerm | None = RewTerm(
        func=mdp.angular_momentum_penalty, weight=-0.02, params={"sensor_name": "robot/root_angmom"}
    )

    self_collisions: RewTerm | None = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision"}
    )
