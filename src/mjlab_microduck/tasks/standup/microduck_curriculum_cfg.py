"""Curriculum specifications for the Microduck standup task.

The shared base leaves terrain progression disabled. The task's rough
variant enables it explicitly; flat variants retain the disabled slot.
"""

import math

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.standup.microduck_flags import (
    BODY_CMD_ALIVE_ANGLE,
    BODY_CMD_ALIVE_XY,
    BODY_CMD_MAX_ANGLE,
    BODY_CMD_MAX_Z_DOWN,
    BODY_CMD_MAX_Z_UP,
    ENABLE_BODY_CONTROL,
    ENABLE_COM_RANDOMIZATION,
    ENABLE_HEAD_COM_RANDOMIZATION,
    ENABLE_VELOCITY_PUSHES,
    VELOCITY_PUSH_RANGE,
)
from mjlab_microduck.tasks.locomotion.velocity.cfg.curriculum_cfg import CurriculumCfg
from mjlab_microduck.utils.configclass import configclass

_N = 24  # num_steps_per_env (see MicroduckStandUpRlCfg)
_ALIVE_XY = (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY)
_ALIVE_ANG = (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE)


@configclass
class MicroduckCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the Microduck standup task."""

    command_vel: CurrTerm | None = None  # no velocity-command curriculum -- fixed near-zero command.

    # Init-pose curriculum: ramp the set_ground_state mix from EASY -> HARD
    # instead of a flat split from step 0 (a flat split let the policy
    # optimize the easy majority and left the hard poses under-trained).
    # Introduces standing/sitting first, then face-down, then face-up last,
    # and biases toward the hard poses late so they get the most practice.
    # (event_param_curriculum shallow-merges these keys into the live
    # set_ground_state event; the z-ranges / joint overrides are untouched.)
    ground_state_mix: CurrTerm | None = CurrTerm(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_ground_state",
            "param_stages": [
                # step,               standing, sitting, face_down(front), face_up(back)
                {"step": 0, "params": {"standing_prob": 0.40, "sitting_prob": 0.40, "face_down_prob": 0.20, "face_up_prob": 0.00}},
                {"step": 600 * _N, "params": {"standing_prob": 0.25, "sitting_prob": 0.30, "face_down_prob": 0.35, "face_up_prob": 0.10}},
                {"step": 1500 * _N, "params": {"standing_prob": 0.20, "sitting_prob": 0.25, "face_down_prob": 0.30, "face_up_prob": 0.25}},
                {"step": 2500 * _N, "params": {"standing_prob": 0.15, "sitting_prob": 0.20, "face_down_prob": 0.30, "face_up_prob": 0.35}},
            ],
        },
    )
    # Head pose command range curriculum -- same per-joint widening as the
    # velocity env (5% -> 100% of each joint's reachable delta from HOME).
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
    # CoM-randomization range curricula -- match velocity. Trunk capped at
    # +-15mm: beyond that the randomized CoM can leave the foot support
    # polygon entirely, training hyper-reactive correction.
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
    push_magnitude: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.push_curriculum,
            params={
                "event_name": "push_robot",
                "push_stages": [
                    {"step": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                    {"step": 500 * _N, "velocity_range": {"x": (-0.08, 0.08), "y": (-0.08, 0.08)}},
                    {"step": 1000 * _N, "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE}},
                ],
            },
        )
        if ENABLE_VELOCITY_PUSHES
        else None
    )
    # action_rate curriculum -- velocity's exact ramp (-0.1 -> -1.0 by iter
    # 1500). Gentler early stages than a steeper ramp: the rise skill gets
    # discovered under light smoothing, then damping tightens.
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
    # Smoothness-polish curricula -- introduce the anti-violence terms only
    # AFTER the recovery skills exist. ground_state_mix finishes ramping the
    # hard poses at iter 2500; from 3000 on, prone resets keep exercising
    # the learned flips while these penalties fine-tune their execution.
    # Two runs proved the same weights active from step 0 prevent the flips
    # from ever being DISCOVERED (attempt-tax on exploration). If recovery
    # degrades after 3000, soften the last stage -- do NOT move the
    # introduction earlier.
    arrival_damping_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "arrival_damping",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 3000 * _N, "weight": -0.025},
                {"step": 4000 * _N, "weight": -0.05},
            ],
        },
    )
    # head_pose_bias: same introduction timing as arrival_damping (timing,
    # not magnitude, is what protects recovery discovery).
    head_pose_bias_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "head_pose_bias",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 3000 * _N, "weight": 0.5},
                {"step": 4000 * _N, "weight": 1.5},
            ],
        },
    )
    torque_rate_weight: CurrTerm | None = CurrTerm(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 3000 * _N, "weight": -1e-3},
            ],
        },
    )

    # -- Body-control curricula (only when ENABLE_BODY_CONTROL) -----------------
    # Tracking weight ramps in at 2500 -- exactly when ground_state_mix
    # reaches its final (hardest) mix, so the recovery-discovery phase
    # trains without any body-command pressure.
    body_pose_tracking_weight: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "body_pose_tracking",
                "weight_stages": [
                    {"step": 0, "weight": 0.0},
                    {"step": 2500 * _N, "weight": 1.5},
                    {"step": 3000 * _N, "weight": 3.0},
                    {"step": 4000 * _N, "weight": 4.0},
                ],
            },
        )
        if ENABLE_BODY_CONTROL
        else None
    )
    # Command range widening, synced to the weight ramp. x/y/yaw stay at
    # their alive ranges (untracked); only z/roll/pitch widen.
    body_pose_range: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.pose_command_range_curriculum,
            params={
                "command_name": "body_pose",
                "range_stages": [
                    {"step": 0, "ranges": (_ALIVE_XY, _ALIVE_XY, (-0.005, 0.005), _ALIVE_ANG, _ALIVE_ANG, _ALIVE_ANG)},
                    {
                        "step": 2500 * _N,
                        "ranges": (
                            _ALIVE_XY,
                            _ALIVE_XY,
                            (-0.010, 0.005),
                            (-math.radians(8), math.radians(8)),
                            (-math.radians(8), math.radians(8)),
                            _ALIVE_ANG,
                        ),
                    },
                    {
                        "step": 3000 * _N,
                        "ranges": (
                            _ALIVE_XY,
                            _ALIVE_XY,
                            (-0.018, 0.008),
                            (-math.radians(12), math.radians(12)),
                            (-math.radians(12), math.radians(12)),
                            _ALIVE_ANG,
                        ),
                    },
                    {
                        "step": 4000 * _N,
                        "ranges": (
                            _ALIVE_XY,
                            _ALIVE_XY,
                            (-BODY_CMD_MAX_Z_DOWN, BODY_CMD_MAX_Z_UP),
                            (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
                            (-BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE),
                            _ALIVE_ANG,
                        ),
                    },
                ],
            },
        )
        if ENABLE_BODY_CONTROL
        else None
    )
    # Conflict relax: the sharp fixed-stand attractors directly out-bid
    # commanded deviations. Their bootstrap/polish job is done by 3000;
    # body_pose_tracking at cmd=0 (30% of resamples) takes over the "sharp
    # peak at nominal stand" role with even tighter stds. The broad
    # bootstrap layers (height_stand, upright_linear, height_stand_l1,
    # pose_stand_*) are left untouched -- they're what recovery leans on.
    height_stand_sharp_weight: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "height_stand_sharp",
                "weight_stages": [
                    {"step": 0, "weight": 1.0},
                    {"step": 3000 * _N, "weight": 0.5},
                    {"step": 4000 * _N, "weight": 0.2},
                ],
            },
        )
        if ENABLE_BODY_CONTROL
        else None
    )
    upright_sharp_weight: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "upright_sharp",
                "weight_stages": [
                    {"step": 0, "weight": 1.5},
                    {"step": 3000 * _N, "weight": 1.0},
                    {"step": 4000 * _N, "weight": 0.5},
                ],
            },
        )
        if ENABLE_BODY_CONTROL
        else None
    )
    standing_composite_weight: CurrTerm | None = (
        CurrTerm(
            func=microduck_mdp.reward_weight,
            params={
                "reward_name": "standing_composite",
                "weight_stages": [
                    {"step": 0, "weight": 3.75},
                    {"step": 3000 * _N, "weight": 2.5},
                    {"step": 4000 * _N, "weight": 1.5},
                ],
            },
        )
        if ENABLE_BODY_CONTROL
        else None
    )
