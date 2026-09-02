"""Reward specifications for the Microduck velstand task.

Subclasses ``tasks.velocity``'s own ``MicroduckRewardsCfg`` -- the walk
layer flows in verbatim (tracking weights, air_time base params, DR/noise,
turn-in-place bucket, etc.), and this file only adds/overrides what the
recovery layer needs. See ``microduck_flags.py`` for the full run-1..run-7
design rationale behind the gate constants used here.
"""

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity.microduck_rewards_cfg import MicroduckRewardsCfg as _VelocityRewardsCfg
from mjlab_microduck.tasks.velstand.microduck_flags import RECOVERED_UP_TILT_DEG, RECOVERED_UP_Z, REWARD_GATE_TILT_DEG
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckRewardsCfg(_VelocityRewardsCfg):
    """Reward terms for the Microduck velstand task."""

    # velocity's head_pose_bias flows in UNGATED (fine on a walk-only env --
    # fell_over terminates fallen episodes there). Velstand episodes
    # SURVIVE falls, so the ungated EMA would charge head "droop" all
    # through the ground phase -- a flat tax on being fallen that the
    # recovery economics never priced in. Add the upright gate: error
    # stops feeding the EMA below z=0.09 / beyond 40deg tilt, matching
    # REWARD_GATE_TILT_DEG, so the term prices exactly what it does in the
    # velocity env -- sustained droop while actually standing/walking --
    # and nothing during recovery.
    head_pose_bias: RewTerm | None = RewTerm(
        func=microduck_mdp.head_pose_bias_penalty,
        weight=0.0,
        params={
            "command_name": "head_pose",
            "tau_s": 1.0,
            "gate_height_low": 0.09,
            "gate_height_high": 0.11,
            "gate_tilt_full_deg": 20.0,
            "gate_tilt_zero_deg": REWARD_GATE_TILT_DEG,
        },
    )

    # -- Recovery reward layer ----------------------------------------------
    # LESSON (runs 1/2/4 -- sitting, lying, head-tripod): ANY positive
    # reward for BEING in a fallen-ish state gets farmed from some
    # comfortable pose. The orientation reward is therefore POTENTIAL-BASED
    # (delta cos tilt): rising pays, falling costs, holding anything pays
    # zero. Unfarmable, ungated, and also rewards catching a stumble while
    # walking. (Run 4 specifically: removing the head-impact penalty
    # unlocked a head-tripod at ~55deg farming a gated upright bonus -- run
    # 2 had only been protected from it by that penalty.)
    upright_progress: RewTerm | None = RewTerm(
        func=microduck_mdp.upright_progress,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # z-axis companion to upright_progress (run-5 crouch-endpoint lesson):
    # the crouch->stand last mile is mostly a HEIGHT change at modest tilt
    # -- where delta cos(tilt) is tiny and the gaussian upright/pose
    # rewards are flat. Same potential-based construction: unfarmable
    # (holding/bobbing nets zero), ungated, charges falls symmetrically.
    # Full prone->stand rise (0.05 -> 0.115m) collects delta~=+0.065*30~=+2;
    # the crouch->stand mile ~= +1.
    height_progress: RewTerm | None = RewTerm(
        func=microduck_mdp.height_progress,
        weight=30.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)), "ceiling": 0.115},
    )
    com_upward_velocity: RewTerm | None = RewTerm(
        func=microduck_mdp.com_upward_velocity,
        weight=0.0,  # recovery term -- ramped in at RECOVERY_ECON_KICKIN_ITER
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            # Height gate slightly above standing (standup uses 0.125) so
            # the rising reward keeps paying until fully up; the fallen
            # gate is what prevents gait-bounce farming, not this ceiling.
            "max_height": 0.125,
            # Tilt-only gate: z=0.0 never triggers (see LESSON above).
            "gate_z_below": 0.0,
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )
    # NO impact penalties (first run lesson #2): the standup SPECIALIST has
    # none -- the duck's recovery pushes off with head/trunk, and a head
    # impact penalty taxed exactly that strategy (falls stayed cheaper than
    # getting up). joint_torque_rate_l2 below covers landing harshness
    # instead. Standup's proven anti-jitter term: penalises torque CHANGE
    # (not magnitude or rotation) -> smooths transfer without blocking the
    # recovery flip.
    joint_torque_rate_l2: RewTerm | None = RewTerm(func=microduck_mdp.joint_torque_rate_l2, weight=-2e-3)

    # -- Recovery economics (first-run lessons #3-#5) ------------------------
    # air_time zeroed while fallen: a robot lying on its trunk can
    # rhythmically tap its feet through the swing window -- the observed
    # "shaking a leg" farm. Same base params as velocity's air_time, plus
    # the fallen gate.
    air_time: RewTerm | None = RewTerm(
        func=microduck_mdp.feet_air_time_upright,
        weight=3.0,
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.125,
            "threshold_max": 0.300,
            "command_name": "twist",
            "command_threshold": 0.01,
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
        },
    )
    # Flat tax while fallen: lying still must be strictly worse than
    # trying. (Without it, waiting 5s for the fallen_too_long recycle was
    # rational -- recovery attempts cost action-rate/torque penalties,
    # waiting cost 0.)
    fallen_tax: RewTerm | None = RewTerm(
        func=microduck_mdp.fallen_state_penalty,
        weight=0.0,  # ramped to -0.5 at RECOVERY_ECON_KICKIN_ITER (see curriculum)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "gate_tilt_above_deg": REWARD_GATE_TILT_DEG,
            # Hysteresis (run-5 crouch-endpoint lesson): recoveries parked
            # in a deep crouch just under the 40deg gate -- past every
            # recovery term's gate, but short of standing. With release
            # conditions matching the recovery_success bounty below, a
            # fall keeps taxing until the stand is actually FINISHED; the
            # sub-40deg crouch is no longer a zero-cost rest state. Arms
            # only on tilt > 40deg, so normal gait is never taxed.
            "release_tilt_below_deg": RECOVERED_UP_TILT_DEG,
            "release_z_above": RECOVERED_UP_Z,
        },
    )
    # One-shot bounty on a COMPLETED recovery (fallen >=0.5s -> genuinely
    # up), with hysteresis so gate-oscillation pays nothing. The strong
    # endpoint signal the dense gated terms lack.
    recovery_success: RewTerm | None = RewTerm(
        func=microduck_mdp.recovery_success,
        weight=0.0,  # ramped to +10 at RECOVERY_ECON_KICKIN_ITER (see curriculum)
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
            "fallen_tilt_deg": REWARD_GATE_TILT_DEG,
            "min_fallen_s": 0.5,
            "up_tilt_deg": RECOVERED_UP_TILT_DEG,
            "up_z": RECOVERED_UP_Z,  # was 0.105, unreachable; see microduck_flags.py
        },
    )
