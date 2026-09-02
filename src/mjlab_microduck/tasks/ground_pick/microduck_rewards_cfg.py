"""Reward specifications for the Microduck ground_pick task.

Regularisation is deliberately kept HEAVIER than velocity's -- the slow,
careful reaching motion wants more damping than walking does (unlike the
dynamic standup recovery, where heavy regularisation blocks the motion).
"""

import math

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.ground_pick.microduck_flags import DESCENT_END, HOLD_END, LEG_JOINTS, NECK_JOINTS, RISE_END
from mjlab_microduck.tasks.locomotion.velocity.cfg.rewards_cfg import RewardsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckRewardsCfg(RewardsCfg):
    """Reward terms for the Microduck ground_pick task.

    ``dof_pos_limits`` is kept inherited unchanged from the base
    ``RewardsCfg``. Everything else not re-declared below (track velocities,
    air_time, foot_clearance/swing/slip, ``pose``) is dropped -- ``pose`` is
    replaced by the phase-conditioned ``ground_pick_return_pose_*`` terms.
    """

    track_linear_velocity: RewTerm | None = None
    track_angular_velocity: RewTerm | None = None
    air_time: RewTerm | None = None
    foot_clearance: RewTerm | None = None
    foot_swing_height: RewTerm | None = None
    foot_slip: RewTerm | None = None
    pose: RewTerm | None = None  # replaced by ground_pick_return_pose_{legs,neck} below.

    # -- Main ground-pick objectives -----------------------------------------
    # Approach phase: reward the mouth tip getting AS CLOSE AS POSSIBLE to the
    # ground. target_height=0 pulls the mouth toward the ground; std=0.10
    # gives gradient from ~20cm out (standing height). The "WITHOUT
    # TOUCHING" guarantee comes from head_impact_penalty (strong) below --
    # the equilibrium point is the mouth just above the ground.
    mouth_ground_proximity: RewTerm | None = RewTerm(
        func=microduck_mdp.mouth_ground_proximity_phased,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "std": 0.10,
            "target_height": 0.0,
            "command_name": "twist",
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )
    # Approach phase: reward the mouth tip's x-axis pointing downward
    # (perpendicular to the ground). alignment in [-1, 1]: 1 = vertical,
    # 0 = horizontal, -1 = pointing up.
    mouth_perpendicular_to_ground: RewTerm | None = RewTerm(
        func=microduck_mdp.mouth_perpendicular_phased,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=["mouth_tip"]),
            "command_name": "twist",
            "descent_end": DESCENT_END,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )
    # Return phase -- legs. Joint layout: 0-4 left leg, 5-8 neck/head,
    # 9-13 right leg.
    ground_pick_return_pose_legs: RewTerm | None = RewTerm(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=6.0,
        params={
            "std": 0.3,
            "command_name": "twist",
            "joint_indices": LEG_JOINTS,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )
    # Return phase -- neck/head: tight std to prevent backward overshoot and
    # head-body collision (head geoms have no collision mesh, so
    # self_collisions can't catch it -- this pose reward is the only guard).
    ground_pick_return_pose_neck: RewTerm | None = RewTerm(
        func=microduck_mdp.ground_pick_return_pose_phased,
        weight=6.0,
        params={
            "std": 0.15,
            "command_name": "twist",
            "joint_indices": NECK_JOINTS,
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )
    # Rise assist: trunk upright, rewarded ONLY during the rise phase
    # (weighted like the pose return, max(0,-sin)). Pose return alone
    # doesn't guarantee dynamic balance while rising; this term keeps the
    # trunk vertical during the extension. Gated on the rise so it doesn't
    # interfere with the forward lean of the approach (the always-on
    # `upright` term stays weak at 0.2 for that reason).
    return_upright: RewTerm | None = RewTerm(
        func=microduck_mdp.ground_pick_return_upright_phased,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.4,
            "command_name": "twist",
            "hold_end": HOLD_END,
            "rise_end": RISE_END,
        },
    )
    # Anti-dive: penalise neck velocity during descent+hold (gate=0 during
    # rise, so it doesn't brake the rise). Slows the head's dive without
    # preventing it from coming back.
    neck_vel_descent: RewTerm | None = RewTerm(
        func=microduck_mdp.neck_vel_descent_penalty,
        weight=-0.1,
        params={"command_name": "twist", "joint_indices": NECK_JOINTS, "hold_end": HOLD_END},
    )
    # Random "payload in the mouth" on the rise (a lifted object, 10-40g per
    # episode). Weight 0: this term is a per-step hook that applies the
    # payload's WEIGHT as an external force at the mouth tip, gated on the
    # rise (phase >= hold_end). The payload itself is sampled at reset by
    # the sample_mouth_payload event.
    mouth_payload_force: RewTerm | None = RewTerm(
        func=microduck_mdp.apply_mouth_payload_force,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["jaw_soft"], site_names=["mouth_tip"]),
            "command_name": "twist",
            "hold_end": HOLD_END,
        },
    )

    # -- Stability (kept from the base RewardsCfg, weights tuned for this task) --
    # Upright: reduced weight -- the robot needs to lean forward during the approach.
    upright: RewTerm | None = RewTerm(
        func=mdp.upright,
        weight=0.2,
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
    soft_landing: RewTerm | None = RewTerm(
        func=mdp.soft_landing,
        weight=-1e-5,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "command_threshold": 0.05},
    )
    # Keep BOTH feet in contact throughout the pick. NOTE: this is CONTACT
    # only; a foot pivoting on the ankle (tipping onto its edge/toe) is
    # caught by feet_flat below, not this term.
    feet_grounded: RewTerm | None = RewTerm(
        func=microduck_mdp.feet_grounded_reward,
        weight=3.0,
        params={"sensor_name": "feet_ground_contact"},
    )
    # Feet FLAT. feet_grounded only sees CONTACT (found per foot): a foot
    # that pivots on the ankle while keeping one contact point slips through
    # -- feet_flat_penalty projects gravity into the foot site frame (flat =
    # site Z vertical, xy^2~=0; any tip -> xy^2>0), forbidding the foot from
    # rolling onto its edge.
    feet_flat: RewTerm | None = RewTerm(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=["left_foot", "right_foot"])},
    )

    # -- Regularisation (heavier than velocity -- slow careful reaching) --------
    # Action smoothness -- flat heavy weight (ramped in via the curriculum,
    # which ends at -2.0 rather than velocity's -1.0).
    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-2.0)
    # Neck/head smoothness -- higher weight because the head is heavily used.
    neck_action_rate_l2: RewTerm | None = RewTerm(func=microduck_mdp.neck_action_rate_l2, weight=-1.0)
    # Joint torque penalty -- increased to further discourage fast/forceful moves.
    joint_torques_l2: RewTerm | None = RewTerm(func=microduck_mdp.joint_torques_l2, weight=-5e-3)
    # Self-collision -- head and neck could clip the legs during a deep crouch.
    self_collisions: RewTerm | None = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision"}
    )
    # No-touch enforcement: we do NOT want ground contact (the mouth must
    # stay just above it). Strong penalty and low threshold -- any ground
    # contact costs a lot. Together with mouth_ground_proximity, this is
    # what fixes the equilibrium at "as close as possible without touching".
    head_impact_penalty: RewTerm | None = RewTerm(
        func=microduck_mdp.body_impact_cost,
        weight=-2.0,
        params={"sensor_name": "head_impact_contact", "threshold": 1.0},
    )
