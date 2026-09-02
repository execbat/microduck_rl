"""Reward specifications for the Microduck roller_standup task.

Rise rewards -- transplant of ``standup``, remapped. Weights come from the
iterations documented in ``microduck_standup_env_cfg.py`` (the walking-duck
version) -- only touch them with a reason. Only the joint indices and the
two heights differ here.

Every reward with an ``asset_cfg`` gets its OWN fresh ``SceneEntityCfg``
instance (never a shared one) -- mjlab resolves and mutates these in place,
so a shared object across terms gives stale indices. Locked by
``tests/test_roller_standup_cfg.py::test_trunk_asset_cfgs_are_distinct_objects``.
"""

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.roller_standup.microduck_flags import ROLLER_PRONE_Z, ROLLER_STAND_Z, SERVO_LEG_JOINTS
from mjlab_microduck.tasks.velocity_rollers.microduck_rewards_cfg import MicroduckRewardsCfg as _RollersRewardsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckRewardsCfg(_RollersRewardsCfg):
    """Reward terms for the Microduck roller_standup task.

    Drops every ``SKATING_REWARDS`` name from ``velocity_rollers``'s
    21-term set (meaningless while on the ground). What remains untouched:
    ``body_ang_vel``, ``angular_momentum``, ``self_collisions``,
    ``neck_action_rate_l2``, ``neck_joint_pos_l2``, ``joint_torques_l2``,
    ``action_over_limit`` -- plus ``action_rate_l2``, re-weighted here (the
    roller env's own -2.0 gait-smoothness ramp is a motion-blocker that
    would slow the fast action a back-recovery needs; see the
    ``action_rate_weight`` curriculum override in
    ``microduck_curriculum_cfg.py`` for the full ramp).
    """

    wheel_speed: RewTerm | None = None
    braking: RewTerm | None = None
    skating_air_time: RewTerm | None = None
    glide: RewTerm | None = None
    single_support: RewTerm | None = None
    gait_symmetry: RewTerm | None = None
    forward_lean: RewTerm | None = None
    heading_hold: RewTerm | None = None
    feet_flat: RewTerm | None = None
    hip_roll_neutral: RewTerm | None = None
    pose: RewTerm | None = None
    com_height_target: RewTerm | None = None
    upright: RewTerm | None = None

    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-0.6)

    # Pose target = HOME (target_overrides=None), LEGS only: neck/head are
    # held by neck_joint_pos_l2 (inherited), which resolves by NAME.
    pose_stand_legs: RewTerm | None = RewTerm(
        func=microduck_mdp.pose_target_match,
        weight=8.0,
        params={"std": 0.5, "joint_indices": SERVO_LEG_JOINTS, "target_overrides": None},
    )
    # L1 bootstrap: constant gradient even far from HOME (the gaussian saturates).
    pose_stand_l1: RewTerm | None = RewTerm(
        func=microduck_mdp.pose_l1_penalty,
        weight=5.0,
        params={"joint_indices": SERVO_LEG_JOINTS, "target_overrides": None},
    )

    # Height in three layers: wide gaussian (pulls up from the ground),
    # narrow gaussian (forces the last cm, where the wide one saturates),
    # and a strong L1 that makes "stay on the ground" net NEGATIVE -- without
    # it, the policy settles for the lazy "motionless on the ground" optimum.
    height_stand: RewTerm | None = RewTerm(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={"std": 0.04, "target_height": ROLLER_STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    height_stand_sharp: RewTerm | None = RewTerm(
        func=microduck_mdp.height_target_gaussian,
        weight=4.0,
        params={"std": 0.015, "target_height": ROLLER_STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    height_stand_l1: RewTerm | None = RewTerm(
        func=microduck_mdp.height_l1_penalty,
        weight=30.0,
        params={"target_height": ROLLER_STAND_Z, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Pays for the MOTION of rising, not just the destination: without it,
    # "sit still collecting partial pose" dominates. Cutoff is 10mm ABOVE
    # the target, otherwise the policy parks at the cutoff altitude and
    # never finishes the rise.
    com_upward_velocity: RewTerm | None = RewTerm(
        func=microduck_mdp.com_upward_velocity,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "max_height": ROLLER_STAND_Z + 0.010},
    )
    # Gentle rise: penalises |a_z|. Compatible with com_upward_velocity -- a
    # constant vertical speed collects the one AND has a_z = 0, so both
    # pressures together select a smooth constant-speed rise.
    #
    # POSITIVE WEIGHT, and this is not a typo. mdp.py mixes two sign
    # conventions: trunk_vertical_accel_penalty already returns -|a_z|
    # (like height_l1_penalty/pose_l1_penalty above, used with +30/+5). The
    # standup-inherited -0.02 formed a double negative and REWARDED vertical
    # acceleration: measured at Episode_Reward/gentle_rise = +0.0118 (the
    # only penalty term logged positive) on run vweolw91. That's the cause
    # of the "very violent" symptom, and it also explains standup's
    # documented failed damping attempts, which were fighting a term
    # actively pushing the other way.
    #
    # Magnitude kept DELIBERATELY small (0.02, the originally-intended
    # value): |a_z| is necessarily high during a from-the-back flip, so a
    # big weight here would be a motion-blocker. Real damping is carried by
    # joint_torque_rate_l2, which penalises torque VARIATION, not motion.
    gentle_rise: RewTerm | None = RewTerm(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.02,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Trunk verticality in two layers: cos(tilt) has a strong gradient while
    # lying down but runs out of steam near vertical; the height-gated sharp
    # gaussian takes over and kills the lean-back failure mode (standup: it
    # tips backward while extending its legs).
    upright_linear: RewTerm | None = RewTerm(
        func=microduck_mdp.body_upright_linear,
        weight=6.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    upright_sharp: RewTerm | None = RewTerm(
        func=microduck_mdp.upright_gaussian_at_height,
        weight=6.0,
        params={
            "std": 0.3,
            "height_low": ROLLER_PRONE_Z,
            "height_high": ROLLER_STAND_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # MULTIPLICATIVE height x uprightness x pose score: because the factors
    # multiply, being good on 2 of 3 criteria pays nothing -- breaks the
    # "leaning at the right height" compromise that additive rewards let
    # through. Deliberately WIDE stds to stay visible during the rise
    # (tight stds gave a ~5e-5 score -- zero gradient).
    standing_composite: RewTerm | None = RewTerm(
        func=microduck_mdp.standing_composite_score,
        weight=15.0,
        params={
            "target_height": ROLLER_STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.40,
            "pose_std": 0.40,
            "joint_indices": SERVO_LEG_JOINTS,
            "target_overrides": None,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Anti-jitter: penalises torque VARIATION, not its magnitude or trunk
    # rotation -> damps the shakes without blocking the flip. standup
    # identified this as the only damper that doesn't kill a from-the-back
    # recovery, so it's THE safe lever to raise.
    #
    # -2e-3 (the standup-inherited value) only contributed -0.0002/step
    # against ~+41.6 of task reward saturated at 95-99% -- i.e. nothing.
    # Across every damper the ratio was ~35:1 in favor of the task, so no
    # reason to be gentle. Measured on run vweolw91 at iteration 7500.
    #
    # Recalibration: raw |dtau|^2 is ~0.1 at convergence, so
    # contribution ~= 0.1 x |weight|. Measured at -0.255/step with weight
    # -2.0 (run d8rnko6p) -- so NOT the cause of the freeze, but backed down
    # to -0.2 to free up damping budget while isolating the sign-bug effect
    # alone. If it's still violent, raise THIS term (formula above) rather
    # than body_ang_vel or action_rate, which are motion-blockers and froze
    # the from-the-back recovery.
    joint_torque_rate_l2: RewTerm | None = RewTerm(func=microduck_mdp.joint_torque_rate_l2, weight=-0.2)

    # NO head-impact penalty. Tried with velstand's values
    # (body_impact_cost, neck subtree, weight -1.0, threshold 2.0): the
    # policy converged to lying still, INERT. Measured (run d8rnko6p):
    # head_impact_penalty -1.01/step, the biggest negative term in the
    # table, while standing_composite collapsed from +14.3 to +3.3.
    #
    # The reasoning error was assuming a "targeted" penalty doesn't
    # constrain motion. False here: to get up from its back, this robot
    # PIVOTS on its head and shoulders. The head is the flip's pivot point,
    # not collateral damage -- penalising it blocks the only mechanism
    # available, and the back was already the failing case.
    #
    # Hypothesis under test: head-slamming was a SYMPTOM of the violence
    # (gentle_rise's sign bug was paying for brutality, and a violent rise
    # ends on the head), not a separate defect. If the slam comes back once
    # the sign is fixed, the fix should be a HEIGHT-GATED penalty -- like
    # upright_sharp is -- to spare the on-the-ground flip phase. Not this one.
    #
    # Watch out for the lazy optimum this freeze rides on: pose_stand_legs
    # stayed at +7.72/8 while the robot was lying down (legs at HOME while
    # prone -> the reward is collected almost for free). height_stand_l1
    # (weight +30) is what must make "stay on the ground" net negative.
