"""Curriculum specifications for the Microduck spin task."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.spin.microduck_flags import ENABLE_COM_RANDOMIZATION, ENABLE_HEAD_COM_RANDOMIZATION
from mjlab_microduck.tasks.velocity.cfg.curriculum_cfg import CurriculumCfg
from mjlab_microduck.utils.configclass import configclass

_N = 24  # num_steps_per_env (see MicroduckSpinRlCfg)


@configclass
class MicroduckCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the Microduck spin task."""

    terrain_levels: CurrTerm | None = None  # flat terrain only.
    command_vel: CurrTerm | None = None  # no velocity-command curriculum -- fixed phase command.

    action_rate_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.5},
                {"step": 250 * _N, "weight": -0.8},
                {"step": 500 * _N, "weight": -1.0},
            ],
        },
    )
    # The scissor bootstrap fades out: it launches the right mechanism,
    # then lets the policy refine its own gesture (free pumping frequency).
    leg_antisym_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "leg_antisymmetry",
            "weight_stages": [
                {"step": 0, "weight": 1.0},
                {"step": 1500 * _N, "weight": 0.5},
                {"step": 3000 * _N, "weight": 0.25},
            ],
        },
    )
    com_range: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 500 * _N, "range": 0.005},
                    {"step": 1000 * _N, "range": 0.01},
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
