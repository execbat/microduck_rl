"""Reward specifications for the Microduck sitstand task.

Every posture_* term below reads the commanded posture and selects its
target (SIT keyframe + SIT_Z vs HOME + STAND_Z) per env. Weights mirror the
sit env's proven stack -- positive task mass ~= velocity scale, so the
shared sim2real regularisers act at the same RELATIVE strength (the standup
transfer lesson).

WARNING preserved from the original: ``descent_speed``/``rise_speed``/
``gentle_motion`` use POSITIVE weights deliberately -- these functions
ALREADY return negative values (``-clamp(...)``, ``-|a_z|``), same
convention as the ``*_l1_penalty`` helpers. A negative weight here would
make them REWARDS for violence (this exact bug trained a butt-hopping,
crash-sitting policy in an earlier run). After any reward change, check
``Episode_Reward/<penalty>`` stays <= 0 in wandb.
"""

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.sitstand.microduck_flags import (
    LEG_JOINTS,
    MAX_DESCENT_SPEED,
    MAX_RISE_SPEED,
    SITTING_TARGET_OVERRIDES,
    SIT_UPRIGHT_Z,
    SIT_Z,
    STAND_UPRIGHT_Z,
    STAND_Z,
)
from mjlab_microduck.tasks.velocity.cfg.rewards_cfg import RewardsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckRewardsCfg(RewardsCfg):
    """Reward terms for the Microduck sitstand task.

    ``dof_pos_limits`` is kept inherited unchanged from the base
    ``RewardsCfg``. ``upright`` is dropped entirely -- replaced by the
    two-layer ``upright_linear``/``upright_while_tall`` below.
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

    # -- Posture-conditioned single-target stack ---------------------------------
    # Pose target -- legs only (head is command-steered). Generous std keeps
    # gradient alive from either end (~1.35 rad knee delta).
    posture_pose_legs: RewTerm | None = RewTerm(
        func=microduck_mdp.posture_pose_match,
        weight=4.0,
        params={
            "command_name": "twist",
            "std": 0.5,
            "joint_indices": LEG_JOINTS,
            "sit_overrides": SITTING_TARGET_OVERRIDES,
        },
    )
    # Head pose tracking -- active in BOTH postures. Weight kept light so a
    # transient head-assist during a transition only pays a small tracking cost.
    head_pose_tracking: RewTerm | None = RewTerm(
        func=microduck_mdp.head_pose_tracking, weight=0.75, params={"command_name": "head_pose", "std": 0.5}
    )
    # L1 bootstrap -- constant gradient toward the commanded pose.
    posture_pose_l1: RewTerm | None = RewTerm(
        func=microduck_mdp.posture_pose_l1,
        weight=1.0,
        params={"command_name": "twist", "joint_indices": LEG_JOINTS, "sit_overrides": SITTING_TARGET_OVERRIDES},
    )
    # Trunk height -- two-layer Gaussian (wide bootstrap pull across the
    # 55mm travel, sharp layer for real gradient in the final cm) + L1
    # transition driver.
    posture_height: RewTerm | None = RewTerm(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z, "std": 0.04},
    )
    posture_height_sharp: RewTerm | None = RewTerm(
        func=microduck_mdp.posture_height_gaussian,
        weight=1.0,
        params={"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z, "std": 0.015},
    )
    # L1 weight 6.0: resting in the WRONG posture must be clearly
    # net-negative in both directions.
    posture_height_l1: RewTerm | None = RewTerm(
        func=microduck_mdp.posture_height_l1,
        weight=6.0,
        params={"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z},
    )
    # Rise bootstrap -- pays for upward motion itself when STAND is
    # commanded and the trunk is below 0.125 (just above the target so the
    # final cm still pays). Destination-only rewards have zero gradient at
    # zero motion; without this the trunk parks seated. Zero under a SIT command.
    rise_bootstrap: RewTerm | None = RewTerm(
        func=microduck_mdp.posture_rise_bootstrap,
        weight=0.75,
        params={"command_name": "twist", "max_height": 0.125, "max_vz": MAX_RISE_SPEED},
    )

    # -- Gentleness (the point of this env) -- three complementary signals ------
    descent_speed: RewTerm | None = RewTerm(
        func=microduck_mdp.trunk_downward_velocity_penalty,
        weight=10.0,
        params={"max_down_vel": MAX_DESCENT_SPEED, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    rise_speed: RewTerm | None = RewTerm(
        func=microduck_mdp.trunk_upward_velocity_penalty,
        weight=0.0,  # ramped up by curriculum once the rise motion exists.
        params={"max_up_vel": MAX_RISE_SPEED, "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    gentle_motion: RewTerm | None = RewTerm(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # Two-layer upright pressure: always-on linear floor (holds the trunk
    # vertical at BOTH rests) + a height-gated booster (blocks the "tip
    # backward while tall" descent exploit; during the rise it doubles as an
    # arrival-uprightness pull).
    upright_linear: RewTerm | None = RewTerm(
        func=microduck_mdp.body_upright_linear,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    upright_while_tall: RewTerm | None = RewTerm(
        func=microduck_mdp.upright_while_tall,
        weight=1.5,
        params={
            "height_low": SIT_UPRIGHT_Z,
            "height_high": STAND_UPRIGHT_Z,
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )

    # Stillness at the commanded posture -- "arrive, then rest QUIETLY,
    # UPRIGHT" as an explicit positive peak. The z gate is a band around the
    # commanded height (inactive during transitions); the tilt gate pays
    # nothing for a tilted rest (back/face/side flops earn zero).
    posture_stillness: RewTerm | None = RewTerm(
        func=microduck_mdp.posture_stillness,
        weight=2.0,
        params={
            "command_name": "twist",
            "sit_z": SIT_Z,
            "stand_z": STAND_Z,
            "band_full": 0.012,
            "band_zero": 0.03,
            "vel_std": 0.05,
            "tilt_full_deg": 25.0,
            "tilt_zero_deg": 60.0,
        },
    )
    # Multiplicative goal score vs the COMMANDED target -- kills partial-sum
    # farming in both postures (plank, flop, lean, park-1cm-short). Broad
    # stds keep gradient visible far from the goal. head_std adds the
    # neck/head-at-command factor so resting with the head dangling to the
    # floor no longer scores as a full composite.
    posture_composite: RewTerm | None = RewTerm(
        func=microduck_mdp.posture_composite,
        weight=3.0,
        params={
            "command_name": "twist",
            "sit_overrides": SITTING_TARGET_OVERRIDES,
            "joint_indices": LEG_JOINTS,
            "sit_z": SIT_Z,
            "stand_z": STAND_Z,
            "height_std": 0.03,
            "upright_std": 0.40,  # ~= 23deg effective -- plank (~70deg+) scores ~0
            "pose_std": 0.40,
            "head_std": 0.40,  # head fully dropped (~1.2 rad) -> factor ~0.01
        },
    )

    # -- Sim2real regularisers -- MATCHED to velocity ---------------------------
    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_torque_rate_l2: RewTerm | None = RewTerm(func=microduck_mdp.joint_torque_rate_l2, weight=0.0)
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
