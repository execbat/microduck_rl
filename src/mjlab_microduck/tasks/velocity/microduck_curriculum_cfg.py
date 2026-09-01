"""Curriculum specifications for the Microduck velocity task."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.utils.configclass import configclass

from .cfg.curriculum_cfg import CurriculumCfg
from .microduck_flags import ENABLE_COM_RANDOMIZATION, ENABLE_HEAD_COM_RANDOMIZATION, NUM_STEPS_PER_ENV

_N = NUM_STEPS_PER_ENV


@configclass
class MicroduckCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the Microduck velocity task."""

    # No velocity-command-range curriculum -- ranges are fixed (see
    # microduck_commands_cfg.py).
    command_vel: CurrTerm | None = None

    # Disabled by default (flat terrain); MicroduckVelocityRoughEnvCfg
    # re-enables this in __post_init__.
    terrain_levels: CurrTerm | None = None

    # action_rate weight ramp: gentle smoothing while the gait bootstraps,
    # then tighten to -1.0 by iter 1500.
    action_rate_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * _N, "weight": -0.2},
                {"step": 750 * _N, "weight": -0.4},
                {"step": 1000 * _N, "weight": -0.6},
                {"step": 1250 * _N, "weight": -0.8},
                {"step": 1500 * _N, "weight": -1.0},
            ],
        },
    )

    # Gradually increase standing env fraction after walking is established.
    standing_envs: CurrTerm | None = CurrTerm(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0, "rel_standing_envs": 0.02},
                {"step": 500 * _N, "rel_standing_envs": 0.05},
                {"step": 750 * _N, "rel_standing_envs": 0.1},
                {"step": 1000 * _N, "rel_standing_envs": 0.15},
                {"step": 1500 * _N, "rel_standing_envs": 0.2},
                {"step": 2000 * _N, "rel_standing_envs": 0.25},
            ],
        },
    )

    # Head pose command range curriculum -- per-joint, scaled to each joint's
    # reachable delta from HOME (with ~10% margin from XML limits). 5-stage
    # shape (5% -> 15% -> 35% -> 65% -> 100% of each joint's final cap).
    # neck/head pitch final +-1.10 rad, head_yaw +-1.40, head_roll +-0.31.
    head_pose_range: CurrTerm | None = CurrTerm(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,             ranges = ((neck_pitch), (head_pitch), (head_yaw), (head_roll))
                {"step": 0, "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 500 * _N, "ranges": ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))},
                {"step": 1000 * _N, "ranges": ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))},
                {"step": 1500 * _N, "ranges": ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))},
                {"step": 2000 * _N, "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
            ],
        },
    )

    # Body pose command range curriculum: stay small in the vel env. The
    # standup env overrides this curriculum with wide ranges + heavy weight.
    body_pose_range: CurrTerm | None = CurrTerm(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {
                    "step": 0,
                    "ranges": (
                        (-0.005, 0.005),  # x (m)
                        (-0.005, 0.005),  # y (m)
                        (-0.005, 0.005),  # z (m)
                        (-0.05, 0.05),  # roll
                        (-0.05, 0.05),  # pitch
                        (-0.05, 0.05),  # yaw
                    ),
                },
            ],
        },
    )

    # CoM randomization range curriculum - start small, ramp up. Capped at
    # +-15 mm (2026-07 audit): the previous ramp to +-30 mm exceeded the foot
    # support polygon (heel is only 20 mm behind the ankle) -- the randomized
    # CoM could sit entirely outside support, forcing a wide/fast
    # hyper-reactive gait and making BACKWARD balance untrainable. Regression
    # timeline matched the ramp increases: 0.015 -> 0.02 -> 0.03 as policies
    # got worse.
    com_range: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * _N, "range": 0.005},
                    {"step": 1000 * _N, "range": 0.01},
                    {"step": 1500 * _N, "range": 0.015},
                ],
            },
        )
        if ENABLE_COM_RANDOMIZATION
        else None
    )

    # Head CoM randomization range curriculum - start small, ramp up. Capped
    # at +-10 mm (2026-07 audit -- same over-conservatism concern as trunk
    # CoM; head is a large lever arm).
    head_com_range: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * _N, "range": 0.005},
                    {"step": 1000 * _N, "range": 0.01},
                ],
            },
        )
        if ENABLE_HEAD_COM_RANDOMIZATION
        else None
    )

    # head_pose_bias ramp: OFF until iter 600, then 1.0 -> 3.0 by iter 1500.
    # Held at 0 early because a posture-precision term is a distraction before
    # a gait exists. At weight 3.0 a 15deg residual bias costs 0.79/step and a
    # 2deg bias costs 0.10/step.
    head_pose_bias_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 600 * _N, "weight": 1.0},
                {"step": 1000 * _N, "weight": 2.0},
                {"step": 1500 * _N, "weight": 3.0},
            ],
        },
    )
