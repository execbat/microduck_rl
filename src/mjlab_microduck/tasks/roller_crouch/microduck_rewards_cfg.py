"""Reward specifications for the Microduck roller_crouch task."""

import math

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.roller_crouch.microduck_flags import (
    CROUCH_LEAN_PITCH,
    CROUCH_POSE,
    CROUCH_POSE_STD,
    DESCENT_END,
    HOLD_END,
    RISE_END,
    STAND_POSE,
)
from mjlab_microduck.tasks.velocity.cfg.rewards_cfg import RewardsCfg
from mjlab_microduck.utils.configclass import configclass

_PHASE_PARAMS = {
    "command_name": "twist",
    "crouch_pose": CROUCH_POSE,
    "stand_pose": STAND_POSE,
    "descent_end": DESCENT_END,
    "hold_end": HOLD_END,
    "rise_end": RISE_END,
}


@configclass
class MicroduckRewardsCfg(RewardsCfg):
    """Reward terms for the Microduck roller_crouch task.

    Only ``upright``/``body_ang_vel``/``angular_momentum``/``action_rate_l2``
    are kept from the base ``RewardsCfg`` (re-declared below with
    task-specific weights/params, including ``dof_pos_limits``, which --
    unlike ``ground_pick`` -- is also dropped here).
    """

    track_linear_velocity: RewTerm | None = None
    track_angular_velocity: RewTerm | None = None
    pose: RewTerm | None = None  # replaced by the phase-interpolated crouch_glide_pose below.
    dof_pos_limits: RewTerm | None = None
    air_time: RewTerm | None = None
    foot_clearance: RewTerm | None = None
    foot_swing_height: RewTerm | None = None
    foot_slip: RewTerm | None = None
    soft_landing: RewTerm | None = None

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

    # Main reward: phase-interpolated POSE (STAND <-> CROUCH). Directive-style
    # -- tells the robot the exact joint configuration at each instant.
    # "Standing back up" (phase -> 1, target = HOME) is rewarded EXACTLY like
    # "crouching down" (the hold, target = CROUCH_POSE) -- symmetric.
    crouch_glide_pose: RewTerm | None = RewTerm(
        func=microduck_mdp.crouch_glide_pose_by_phase,
        weight=6.0,
        params={**_PHASE_PARAMS, "std": CROUCH_POSE_STD},
    )
    # L1 bootstrap: constant gradient toward the target even where the
    # gaussian saturates far from the pose.
    crouch_glide_pose_l1: RewTerm | None = RewTerm(
        func=microduck_mdp.crouch_glide_pose_l1, weight=2.0, params=_PHASE_PARAMS
    )
    # Preserve momentum (don't brake) -- independent of the command.
    forward_speed: RewTerm | None = RewTerm(
        func=microduck_mdp.forward_speed_reward, weight=1.0, params={"vel_ref": 0.2}
    )
    # Slight forward lean while crouched, against the backward tip-back
    # observed on the real robot during a fast descent. Gated by the blend
    # (crouch phase only).
    crouch_forward_lean: RewTerm | None = RewTerm(
        func=microduck_mdp.crouch_forward_lean,
        weight=1.0,
        params={
            "command_name": "twist",
            "target_pitch": CROUCH_LEAN_PITCH,
            "std": 0.1,
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )
    # Glide stability.
    feet_flat: RewTerm | None = RewTerm(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    self_collisions: RewTerm | None = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision"}
    )
    neck_action_rate_l2: RewTerm | None = RewTerm(func=microduck_mdp.neck_action_rate_l2, weight=-0.5)
    joint_torques_l2: RewTerm | None = RewTerm(func=microduck_mdp.joint_torques_l2, weight=-1e-3)
