"""Curriculum specifications for the Microduck velocity_swizzle task."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity_rollers.microduck_curriculum_cfg import (
    MicroduckCurriculumCfg as _RollersCurriculumCfg,
)
from mjlab_microduck.tasks.velocity_swizzle.microduck_flags import HEAD_POSE_INITIAL_RANGES, NUM_STEPS_PER_ENV
from mjlab_microduck.utils.configclass import configclass

_N = NUM_STEPS_PER_ENV


@configclass
class MicroduckCurriculumCfg(_RollersCurriculumCfg):
    """Curriculum terms for the Microduck velocity_swizzle task.

    ``action_rate_weight``/``wheel_friction``/``com_range``/``head_com_range``
    are inherited unchanged from ``velocity_rollers``. The 4 fields below
    are genuinely new.
    """

    # Swap heading_hold -> heading_tracking: hold straight while the
    # swizzle solidifies, then fade to following the commanded heading.
    heading_hold_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "heading_hold",
            "weight_stages": [
                {"step": 0, "weight": 1.0},  # must match heading_hold's initial weight
                {"step": 1000 * _N, "weight": 1.0},  # hold straight while the swizzle solidifies
                {"step": 1750 * _N, "weight": 0.5},
                {"step": 2500 * _N, "weight": 0.0},
            ],
        },
    )
    heading_tracking_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "heading_tracking",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 1000 * _N, "weight": 0.0},  # straight-only until here
                {"step": 1750 * _N, "weight": 1.5},
                {"step": 2500 * _N, "weight": 3.0},
            ],
        },
    )

    # head_pose_tracking ramps 0 -> 4.0, staying 0 until ~1500 it. (swizzle
    # solid), so head control is added on top of a stable swizzle.
    head_pose_tracking_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_tracking",
            "weight_stages": [
                {"step": 0, "weight": 0.0},  # must match initial weight
                {"step": 1500 * _N, "weight": 0.0},  # head off while swizzle solidifies
                {"step": 2250 * _N, "weight": 2.0},
                {"step": 3000 * _N, "weight": 4.0},
            ],
        },
    )
    # Head-command range widens over the SAME window (tiny until 1500, full
    # by 3000), so the commanded head barely moves early and reaches full
    # range once the policy can handle it.
    head_pose_range: CurrTerm | None = CurrTerm(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                {"step": 0, "ranges": HEAD_POSE_INITIAL_RANGES},
                {"step": 1500 * _N, "ranges": HEAD_POSE_INITIAL_RANGES},
                {
                    "step": 2250 * _N,
                    "ranges": ((-0.55, 0.55), (-0.55, 0.55), (-0.70, 0.70), (-0.15, 0.15)),
                },
                {
                    "step": 3000 * _N,
                    "ranges": ((-1.10, 1.10), (-1.10, 1.10), (-1.40, 1.40), (-0.31, 0.31)),
                },
            ],
        },
    )
