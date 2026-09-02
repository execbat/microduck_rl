"""Curriculum specifications for the Microduck roller_standup task.

Subclasses ``velocity_rollers``'s ``MicroduckCurriculumCfg`` -- ``com_range``/
``head_com_range`` flow through inherited unchanged. ``wheel_friction`` and
``action_rate_weight`` are the SAME field names as ``velocity_rollers``'s
own, but with entirely different ramps (see each one's docstring below).
``ground_state_mix`` and ``push_magnitude`` are genuinely new fields.
"""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity_rollers.microduck_curriculum_cfg import (
    MicroduckCurriculumCfg as _RollersCurriculumCfg,
)
from mjlab_microduck.utils.configclass import configclass

_N = 24  # num_steps_per_env (see MicroduckRollerStandUpRlCfg)

# Same value as the wheel_friction curriculum's stage-0 range and the
# randomize_wheel_friction event's default -- kept as one constant so they
# can't drift apart (see the defensive-redundancy note on the event below).
_WHEEL_FRICTION_STAGE0 = (0.0500, 0.0500)


@configclass
class MicroduckCurriculumCfg(_RollersCurriculumCfg):
    # Start-pose curriculum, easy -> hard. With a flat mix from the start,
    # the policy optimises the easy majority and leaves the back-start
    # under-trained (the standup lesson: it froze into "do nothing" on that
    # pose). So standing+belly are introduced first, the back late, and the
    # mix is biased toward the hard poses at the end so they get the most
    # training.
    ground_state_mix: CurrTerm | None = CurrTerm(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                {"step": 0, "params": {"standing_prob": 0.50, "sitting_prob": 0.00, "face_down_prob": 0.50, "face_up_prob": 0.00}},
                {"step": 600 * _N, "params": {"standing_prob": 0.35, "sitting_prob": 0.00, "face_down_prob": 0.45, "face_up_prob": 0.20}},
                {"step": 1500 * _N, "params": {"standing_prob": 0.25, "sitting_prob": 0.00, "face_down_prob": 0.40, "face_up_prob": 0.35}},
                {"step": 2500 * _N, "params": {"standing_prob": 0.20, "sitting_prob": 0.00, "face_down_prob": 0.40, "face_up_prob": 0.40}},
            ],
        },
    )

    # INVERTED wheel friction: braked -> free. This is the one genuinely new
    # piece of this env, and the crux of the difficulty: the wheels roll, so
    # there is NO longitudinal grip to push off the ground with. The roller
    # env RAISES this friction (0 -> 0.0015); here it DECREASES, to
    # bootstrap the gesture on an easy problem (near-locked wheels ~= feet)
    # before imposing the real rolling physics.
    #
    # WATCH: if Episode_Reward/standing_composite collapses at a stage, the
    # "sticky feet" gesture doesn't transfer to free wheels -> a skater
    # technique (intermediate knee support, one skate at a time) will need
    # to be guided in. That's an actionable result, not a failure.
    #
    # SIM2REAL WARNING: only checkpoints AFTER the last stage (iter 4000+)
    # are deployment candidates. Before that, the policy leans on a rolling
    # friction that doesn't exist on the real robot.
    wheel_friction: CurrTerm | None = CurrTerm(
        func=microduck_mdp.wheel_friction_curriculum,
        params={
            "event_name": "randomize_wheel_friction",
            "ranges_stages": [
                {"step": 0, "ranges": _WHEEL_FRICTION_STAGE0},
                {"step": 1000 * _N, "ranges": (0.0200, 0.0200)},
                {"step": 2000 * _N, "ranges": (0.0080, 0.0080)},
                {"step": 3000 * _N, "ranges": (0.0030, 0.0030)},
                {"step": 4000 * _N, "ranges": (0.0015, 0.0015)},
            ],
        },
    )

    # action_rate: the standup ramp, not the roller one. The roller env
    # raises this to -2.0 for a calm gait -- that's a motion-blocker, it
    # slows the fast action a from-the-back recovery needs (standup
    # documents that too strong an action_rate killed that recovery).
    # Smoothness here is instead carried by joint_torque_rate_l2.
    action_rate_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.4},
                {"step": 250 * _N, "weight": -0.8},
                {"step": 500 * _N, "weight": -1.0},
            ],
        },
    )

    # Ramped pushes. push_robot is inherited from the roller env (+-0.2m/s,
    # every 3-6s) but without a curriculum there -- a shove at step 0 would
    # disrupt the rise's bootstrap, so ramp it in like standup does.
    push_magnitude: CurrTerm | None = CurrTerm(
        func=microduck_mdp.push_curriculum,
        params={
            "event_name": "push_robot",
            "push_stages": [
                {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {"step": 500 * _N, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                {"step": 1000 * _N, "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
            ],
        },
    )
