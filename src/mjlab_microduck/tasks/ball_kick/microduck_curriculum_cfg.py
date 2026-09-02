"""Curriculum specifications for the Microduck BallKick task."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.ball_kick.microduck_flags import (
    ENABLE_COM_RANDOMIZATION,
    ENABLE_HEAD_COM_RANDOMIZATION,
    ENABLE_VELOCITY_PUSHES,
    VELOCITY_PUSH_RANGE,
)
from mjlab_microduck.utils.configclass import configclass

from mjlab_microduck.tasks.locomotion.velocity.cfg.curriculum_cfg import CurriculumCfg

_N = 24  # num_steps_per_env (see MicroduckBallKickRlCfg)


@configclass
class MicroduckCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the Microduck BallKick task."""

    terrain_levels: CurrTerm | None = None  # flat terrain only.
    command_vel: CurrTerm | None = None  # no velocity-command curriculum -- fixed near-zero command.

    # action_rate ramp -- velocity's exact stages (-0.1 -> -1.0 by iter 1500).
    # NOTE: the kick is a fast one-shot swing; if the converged kick is too
    # weak, softening the ramp end (-1.0 -> -0.6) is the first knob to try
    # (motion-blocker vs dynamic-task tradeoff, see standup regularization notes).
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

    # Ramp pushes in AFTER the kick skill starts forming: a full-strength
    # shove during the one-legged strike phase at iter 0 would tax the
    # discovery of the swing itself (same timing lesson as standup).
    push_magnitude: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                    {"step": 500 * _N, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {
                        "step": 1000 * _N,
                        "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                    },
                ],
            },
        )
        if ENABLE_VELOCITY_PUSHES
        else None
    )
