"""Reward specifications for the Microduck roulade task.

Motion-blockers (``body_ang_vel``, ``|a_z|``, ``arrival_damping``) stay near
zero during discovery -- the roll IS a large angular-velocity, large-impact
event, and taxing attempts prevents discovery (proven twice on standup).
Style pressure is introduced late, by curriculum (see
``microduck_curriculum_cfg.py``).
"""

import math

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.roulade.microduck_flags import LANDING_GATE_HI, LANDING_GATE_LO, LEG_JOINTS, RISE_GATE_HI, RISE_GATE_LO, STAND_Z
from mjlab_microduck.tasks.locomotion.velocity.cfg.rewards_cfg import RewardsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckRewardsCfg(RewardsCfg):
    """Reward terms for the Microduck roulade task.

    ``dof_pos_limits`` is kept inherited unchanged from the base
    ``RewardsCfg``. ``upright`` is dropped entirely (an always-on upright
    would oppose the flip -- landing uprightness is handled by the
    completion-gated terms below instead).
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

    # -- The one dense task signal: progress increments -------------------------
    # During a ~1.5s roll it averages ~0.7/step; total payout per full roll
    # from a standing spawn ~= weight * (episode steps it took) * mean ~= weight * 50.
    # max_paid_rate=5.0: measured physics shows the over-the-top transit runs
    # at 3.5-5.5 rad/s (this robot is 10cm tall, its natural tumble timescale
    # is fast) -- a lower cap forfeits most of the physically-necessary
    # rotation. Style pressure lives in |a_z| / action_rate / the support
    # gate, not in fighting gravity's clock.
    roulade_progress: RewTerm | None = RewTerm(
        func=microduck_mdp.roulade_progress,
        weight=8.0,
        params={"target_angle": 2 * math.pi, "max_paid_rate": 5.0},
    )
    # Whip-speed tax: taxes genuine whips (threshold above the measured p90
    # transit speed of ~5.5 rad/s), not the natural tumble.
    roulade_overspeed: RewTerm | None = RewTerm(
        func=microduck_mdp.roulade_overspeed_penalty, weight=-0.1, params={"omega_max": 7.0}
    )
    # Head-as-pivot shaping: contact x mid-roll window x forward-rate factor
    # (the rate factor kills the "rest face-down with head on floor" farm).
    roulade_head_pivot: RewTerm | None = RewTerm(
        func=microduck_mdp.roulade_head_pivot,
        weight=0.5,
        params={
            "sensor_name": "head_ground_contact",
            "angle_lo": math.radians(30.0),
            "angle_hi": math.radians(240.0),
            "rate_norm": 2.0,
        },
    )
    # Completion-gated standing annuity -- the dominant attractor. Broad stds
    # so a partial landing still scores visibly (~0.2+).
    roulade_landing_composite: RewTerm | None = RewTerm(
        func=microduck_mdp.roulade_landing_composite,
        weight=4.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": LEG_JOINTS,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
            "target_overrides": None,
        },
    )
    # Completion-gated bootstrap layers (gradient far from the goal, where
    # the composite product is ~=0): linear upright + broad height gaussian.
    roulade_upright_after_roll: RewTerm | None = RewTerm(
        func=microduck_mdp.roulade_upright_after_roll,
        weight=1.5,
        params={"gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )
    roulade_height_after_roll: RewTerm | None = RewTerm(
        func=microduck_mdp.roulade_height_after_roll,
        weight=1.0,
        params={"target_height": STAND_Z, "std": 0.04, "gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )
    # Sharp landing layer: tight-std upright x height product on top of the
    # broad composite, for gradient at the final basin (every completed
    # episode was observed parking at the same pose, where the broad stds
    # give no gradient to finish).
    roulade_landing_sharp: RewTerm | None = RewTerm(
        func=microduck_mdp.roulade_landing_sharp,
        weight=2.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.015,
            "upright_std": 0.3,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
        },
    )
    # Completion-gated stand tax (the standup lesson): once the rotation is
    # done, every step spent below STAND_Z costs -- "crumple in a heap after
    # the roll" flips from free to net-negative. Gate closed during the
    # roll, so the roll itself is never taxed; mid/late-roll spawns are born
    # with it active, which is the point.
    roulade_stand_tax: RewTerm | None = RewTerm(
        func=microduck_mdp.roulade_stand_tax,
        weight=5.0,
        params={"target_height": STAND_Z, "gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )
    # Exit-rise bootstrap: upward CoM velocity, gated to the late-roll region
    # (supine -> up is the face-up-recovery problem; end-state rewards have
    # zero gradient at zero motion there).
    roulade_rise_velocity: RewTerm | None = RewTerm(
        func=microduck_mdp.roulade_rise_velocity,
        weight=0.75,
        params={"max_height": STAND_Z + 0.01, "gate_lo": RISE_GATE_LO, "gate_hi": RISE_GATE_HI},
    )
    # Straightness: dense per-step gradient back toward the sagittal plane
    # (a shoulder-roll is lower-energy than straight over the head, the same
    # cheat human beginners default to). Structural fix is the flatness gate
    # on the accumulator + the head-top latch (side rolls no longer count as
    # rotation at all); these penalties provide the gradient.
    roulade_sagittal: RewTerm | None = RewTerm(func=microduck_mdp.roulade_sagittal_penalty, weight=-0.1)
    roulade_lateral_vel: RewTerm | None = RewTerm(
        func=microduck_mdp.roulade_lateral_velocity_penalty, weight=-0.5
    )
    roulade_flatness: RewTerm | None = RewTerm(func=microduck_mdp.roulade_flatness_penalty, weight=-0.5)

    # -- Sim2real regularisers ---------------------------------------------------
    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_torque_rate_l2: RewTerm | None = RewTerm(func=microduck_mdp.joint_torque_rate_l2, weight=0.0)
    body_ang_vel: RewTerm | None = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.002,  # must stay ~=0: the roll IS omega.
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    angular_momentum: RewTerm | None = RewTerm(
        func=mdp.angular_momentum_penalty, weight=-0.001, params={"sensor_name": "robot/root_angmom"}
    )
    # Arrival damper: trunk omega_xy^2 gated on standing height AND low
    # tilt, so the roll itself is never taxed; introduced at 0 and ramped by
    # curriculum.
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
    # |a_z| impact shaping -- active from step 0 (discovery is easy in this
    # env, so shaping the landing style from the start is the priority).
    # Curriculum ramps it further. NOTE: trunk_vertical_accel_penalty is
    # SELF-NEGATING (returns -|a_z|) -> POSITIVE weight (a negative weight
    # here would reward violence).
    gentle_landing: RewTerm | None = RewTerm(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # Self-collision -- LIGHT: a tucked roll needs body-on-body contact
    # (knees against trunk); a heavier penalty would fight the tuck.
    self_collisions: RewTerm | None = RewTerm(
        func=mdp.self_collision_cost, weight=-0.1, params={"sensor_name": "self_collision"}
    )
