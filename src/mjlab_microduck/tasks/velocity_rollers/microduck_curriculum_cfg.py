"""Curriculum specifications for the Microduck velocity_rollers task."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.locomotion.velocity.cfg.curriculum_cfg import CurriculumCfg
from mjlab_microduck.tasks.velocity_rollers.microduck_flags import (
    ENABLE_COM_RANDOMIZATION,
    ENABLE_HEAD_COM_RANDOMIZATION,
    ENABLE_WHEEL_FRICTION_RANDOMIZATION,
)
from mjlab_microduck.utils.configclass import configclass

_N = 24  # num_steps_per_env (see MicroduckRollersRlCfg)


@configclass
class MicroduckCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the Microduck velocity_rollers task."""

    terrain_levels: CurrTerm | None = None  # flat terrain only.
    command_vel: CurrTerm | None = None  # no velocity-command curriculum -- see MicroduckCommandsCfg.

    # action_rate penalty raised (-1.0 -> -2.0) for a CALMER gait: the main
    # "less movement" lever -- it penalises fast/large action changes, so
    # motions become smaller, smoother AND less frequent.
    action_rate_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -1.0},
                {"step": 250 * _N, "weight": -1.5},
                {"step": 500 * _N, "weight": -2.0},
            ],
        },
    )

    # Delayed + softened ramp: an earlier schedule started adding bearing
    # drag right when wheel_speed peaked, pushing the policy off skating into
    # a heading-farming local optimum. Keep the wheels free until skating is
    # robust, then add gentle, realistic drag.
    wheel_friction: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.wheel_friction_curriculum,
            params={
                "event_name": "randomize_wheel_friction",
                "ranges_stages": [
                    {"step": 0 * _N, "ranges": (0.0000, 0.0000)},
                    {"step": 2000 * _N, "ranges": (0.0005, 0.0005)},
                    {"step": 3500 * _N, "ranges": (0.0010, 0.0010)},
                    {"step": 5000 * _N, "ranges": (0.0015, 0.0015)},
                ],
            },
        )
        if ENABLE_WHEEL_FRICTION_RANDOMIZATION
        else None
    )

    # CoM randomisation curricula -- velocity's ramp, capped lower for the
    # balance-sensitive skating task.
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
