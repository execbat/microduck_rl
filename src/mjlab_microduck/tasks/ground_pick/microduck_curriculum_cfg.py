"""Curriculum specifications for the Microduck ground_pick task.

The shared base leaves terrain progression disabled. The task's rough
variant enables it explicitly; flat variants retain the disabled slot.
"""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.ground_pick.microduck_flags import ENABLE_COM_RANDOMIZATION, ENABLE_HEAD_COM_RANDOMIZATION
from mjlab_microduck.tasks.locomotion.velocity.cfg.curriculum_cfg import CurriculumCfg
from mjlab_microduck.utils.configclass import configclass

_N = 24  # num_steps_per_env (see MicroduckGroundPickRlCfg)


@configclass
class MicroduckCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the Microduck ground_pick task."""

    command_vel: CurrTerm | None = None  # no velocity-command curriculum -- fixed phase command.

    # Action-rate curriculum: warm up light so the gross reaching motion can
    # form, then clamp down HARD (-2.0, heavier than velocity's -1.0) for smoothness.
    action_rate_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.8},
                {"step": 250 * _N, "weight": -1.5},
                {"step": 500 * _N, "weight": -2.0},
            ],
        },
    )

    # CoM-randomization range curricula -- match velocity (ramp 0.003 -> 0.02
    # trunk, 0.003 -> 0.01 head).
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
                    {"step": 2000 * _N, "range": 0.02},
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
