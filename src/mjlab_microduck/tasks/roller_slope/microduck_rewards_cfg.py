"""Reward specifications for the Microduck roller_slope task.

Free balance -- the robot places its own centre of gravity. No fixed-pose
reward: the standing pose from the flat gait (which prevented it from
bending/leaning) is dropped entirely, so the robot is free to move its CoM
(hips/knees, lean) to hold the slope. Just: stay upright, stay alive,
glide, go straight -- and don't fall (terminations).
"""

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity_rollers.microduck_rewards_cfg import MicroduckRewardsCfg as _RollersRewardsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckRewardsCfg(_RollersRewardsCfg):
    """Reward terms for the Microduck roller_slope task.

    Of ``velocity_rollers``'s 21-term set, only ``action_rate_l2`` (weight
    re-asserted, matches the inherited -1.0, kept for parity with the
    original) survives untouched. ``upright``/``feet_flat``/
    ``neck_action_rate_l2``/``neck_joint_pos_l2``/``joint_torques_l2``/
    ``heading_hold`` are REUSED NAMES with brand-new values (same term
    identity in logs/wandb, different reward function/params) -- everything
    else velocity_rollers had is dropped (set to ``None``).
    """

    pose: RewTerm | None = None
    body_ang_vel: RewTerm | None = None
    angular_momentum: RewTerm | None = None
    com_height_target: RewTerm | None = None
    self_collisions: RewTerm | None = None
    action_over_limit: RewTerm | None = None
    hip_roll_neutral: RewTerm | None = None
    wheel_speed: RewTerm | None = None
    braking: RewTerm | None = None
    skating_air_time: RewTerm | None = None
    glide: RewTerm | None = None
    single_support: RewTerm | None = None
    gait_symmetry: RewTerm | None = None
    forward_lean: RewTerm | None = None

    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-1.0)

    # Stay upright -- free balance, no fixed pose to fight the lean.
    # (Reuses the "upright" name; velocity_rollers's own upright used
    # mdp.upright -- this is a different function entirely.)
    upright: RewTerm | None = RewTerm(
        func=microduck_mdp.body_upright_gaussian,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "std": 0.2},
    )
    alive: RewTerm | None = RewTerm(func=microduck_mdp.is_alive, weight=1.0)
    # Let it GLIDE (roll), not accelerate/run: rewards wheel ROLLING
    # downhill, capped at cap_speed. Capped -> no incentive to push faster;
    # wheel-based -> "running" (pushing the base without rolling) doesn't
    # pay. Without a glide reward, the optimum would be standing still;
    # with it, it lets itself roll as long as it holds balance.
    wheel_glide: RewTerm | None = RewTerm(func=microduck_mdp.wheel_glide_reward, weight=2.0, params={"cap_speed": 0.35})
    # GO STRAIGHT: hold the spawn yaw (= 0 = facing downhill). Corrective
    # (the robot can recover), the right way to go straight. (Reuses the
    # "heading_hold" name with different params than velocity_rollers's own.)
    heading_hold: RewTerm | None = RewTerm(func=microduck_mdp.heading_hold_reward, weight=1.5, params={"std": 0.4})
    feet_flat: RewTerm | None = RewTerm(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    neck_action_rate_l2: RewTerm | None = RewTerm(func=microduck_mdp.neck_action_rate_l2, weight=-0.5)
    # KEEP THE HEAD STRAIGHT: penalise neck/head joint deviation from HOME.
    # The fixed LEG pose was removed (for free balance), but nothing was
    # holding the head -- it went anywhere. This constrains ONLY the
    # head/neck, not the legs. (Reuses the "neck_joint_pos_l2" name; no
    # ``pattern`` param here, unlike e.g. spin's use of the same function.)
    neck_joint_pos_l2: RewTerm | None = RewTerm(func=microduck_mdp.neck_joint_pos_l2, weight=-0.75)
    joint_torques_l2: RewTerm | None = RewTerm(func=microduck_mdp.joint_torques_l2, weight=-1e-3)
