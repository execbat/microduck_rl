"""Curriculum specifications for the Microduck roulade task."""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.roulade.microduck_flags import ENABLE_COM_RANDOMIZATION, ENABLE_HEAD_COM_RANDOMIZATION
from mjlab_microduck.tasks.locomotion.velocity.cfg.curriculum_cfg import CurriculumCfg
from mjlab_microduck.utils.configclass import configclass

_N = 24  # num_steps_per_env (see MicroduckRouladeRlCfg)


@configclass
class MicroduckCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the Microduck roulade task."""

    terrain_levels: CurrTerm | None = None  # flat terrain only.
    command_vel: CurrTerm | None = None  # no velocity-command curriculum -- fixed near-zero command.

    # Reverse-curriculum mix: heavy mid-roll early (the completion sub-task
    # is learnable from day 0 -- it overlaps face-up recovery), shift toward
    # standing starts as the full roll gets discovered. Mid-roll never goes
    # to zero: it keeps the second half practiced and is realistic DR anyway.
    roulade_spawn_mix: CurrTerm | None = CurrTerm(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_roulade_state",
            "param_stages": [
                {"step": 0, "params": {"standing_prob": 0.50, "midroll_prob": 0.50}},
                {"step": 3000 * _N, "params": {"standing_prob": 0.65, "midroll_prob": 0.35}},
                {"step": 6000 * _N, "params": {"standing_prob": 0.80, "midroll_prob": 0.20}},
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
    # -0.1 minimum from step 0 (near-zero smoothing at the start bred
    # violence in an earlier run); ceiling softened to -0.4 (a harder
    # tightening was observed squeezing the rise phase).
    action_rate_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 1500 * _N, "weight": -0.2},
                {"step": 3000 * _N, "weight": -0.4},
            ],
        },
    )
    # Smoothness polish -- introduced only after the roll skill exists (the
    # standup timing lesson: any attempt-tax active during discovery
    # prevents the maneuver from being found at all; fix is timing, not magnitude).
    arrival_damping_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "arrival_damping",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 2500 * _N, "weight": -0.025},
                {"step": 3500 * _N, "weight": -0.05},
            ],
        },
    )
    torque_rate_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 2500 * _N, "weight": -5e-4},
                {"step": 3500 * _N, "weight": -1e-3},
            ],
        },
    )
    gentle_landing_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            # POSITIVE weights: the func is self-negating (returns -|a_z|).
            "reward_name": "gentle_landing",
            "weight_stages": [
                {"step": 0, "weight": 0.002},
                {"step": 2500 * _N, "weight": 0.005},
            ],
        },
    )
