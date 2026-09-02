"""Curriculum specifications for the Microduck sitstand task.

``terrain_levels`` is kept inherited unchanged from the base
``CurriculumCfg`` (rough-compatible) -- the Flat env cfg variant disables it
in its own ``__post_init__`` instead, same pattern as ``ground_pick``.
"""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.sitstand.microduck_flags import (
    ENABLE_COM_RANDOMIZATION,
    ENABLE_HEAD_COM_RANDOMIZATION,
    ENABLE_VELOCITY_PUSHES,
    VELOCITY_PUSH_RANGE,
)
from mjlab_microduck.tasks.velocity.cfg.curriculum_cfg import CurriculumCfg
from mjlab_microduck.utils.configclass import configclass

_N = 24  # num_steps_per_env (see MicroduckSitStandRlCfg)


@configclass
class MicroduckCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the Microduck sitstand task."""

    command_vel: CurrTerm | None = None  # no velocity-command curriculum -- SitStandCommand replaces it.

    # Head pose command range curriculum -- same per-joint widening as the
    # velocity/standup envs (5% -> 100% of each joint's reachable delta).
    head_pose_range: CurrTerm | None = CurrTerm(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": 0, "ranges": ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))},
                {"step": 500 * _N, "ranges": ((-0.17, 0.17), (-0.17, 0.17), (-0.21, 0.21), (-0.047, 0.047))},
                {"step": 1000 * _N, "ranges": ((-0.39, 0.39), (-0.39, 0.39), (-0.49, 0.49), (-0.11, 0.11))},
                {"step": 1500 * _N, "ranges": ((-0.72, 0.72), (-0.72, 0.72), (-0.91, 0.91), (-0.20, 0.20))},
                {"step": 2000 * _N, "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31))},
            ],
        },
    )
    # CoM-randomization range curricula -- match velocity (trunk capped at
    # +-15mm, head at +-10mm).
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
    # Delayed significantly (sit env lesson): a push mid-transition tips the
    # robot into configurations it can't recover from before the motions
    # have consolidated; early pushes made the policy unlearn sitting and
    # converge to "just stand doing nothing".
    push_magnitude: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                    {"step": 1000 * _N, "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}},
                    {"step": 1500 * _N, "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)}},
                    {"step": 2000 * _N, "velocity_range": {"x": (-0.20, 0.20), "y": (-0.20, 0.20)}},
                    {"step": 2500 * _N, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )
        if ENABLE_VELOCITY_PUSHES
        else None
    )
    # action_rate curriculum -- velocity's exact ramp (-0.1 -> -1.0 by iter 1500).
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
    # Descent-speed cap tightening: discover the sit under magnitude 10
    # (crash-sit already net-negative), then tighten to 20. POSITIVE
    # weights -- the function is self-negating.
    descent_speed_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "descent_speed",
            "weight_stages": [
                {"step": 0, "weight": 10.0},
                {"step": 500 * _N, "weight": 20.0},
            ],
        },
    )
    # Rise-speed cap -- introduced only AFTER the rise motion exists (the
    # standup attempt-tax lesson: any motion-tax during discovery makes
    # exploratory attempts net-negative and the skill is never found).
    rise_speed_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "rise_speed",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1500 * _N, "weight": 5.0},
                {"step": 2500 * _N, "weight": 10.0},
            ],
        },
    )
    # Torque-rate anti-jitter -- phased in once both transition motions exist.
    torque_rate_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 750 * _N, "weight": -5e-4},
                {"step": 1250 * _N, "weight": -1e-3},
            ],
        },
    )
