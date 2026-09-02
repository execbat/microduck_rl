"""Reward specifications for the Microduck spin task."""

import math

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.spin.microduck_flags import NECK_PATTERN_NO_YAW
from mjlab_microduck.tasks.locomotion.velocity.cfg.rewards_cfg import RewardsCfg
from mjlab_microduck.utils.configclass import configclass

_ENVELOPE = {
    "rate_max": microduck_mdp.SPIN_RATE_MAX,
    "accel_end": microduck_mdp.SPIN_ACCEL_END,
    "hold_end": microduck_mdp.SPIN_HOLD_END,
    "brake_end": microduck_mdp.SPIN_BRAKE_END,
}


@configclass
class MicroduckRewardsCfg(RewardsCfg):
    """Reward terms for the Microduck spin task.

    Only ``upright``/``body_ang_vel``/``action_rate_l2`` are kept from the
    base ``RewardsCfg`` (re-declared below with task-specific weights).
    ``angular_momentum`` is deliberately NOT kept (unlike most other
    tasks): it penalises the 3D angular-momentum NORM, so it would directly
    fight the spin. ``body_ang_vel`` only penalises x/y ("don't penalize
    z-angular velocity" in mjlab), so it's kept -- it tames roll/pitch
    wobble without opposing the rotation. ``dof_pos_limits`` is also
    dropped here (unlike most other tasks, which keep it inherited).
    """

    track_linear_velocity: RewTerm | None = None
    track_angular_velocity: RewTerm | None = None
    dof_pos_limits: RewTerm | None = None
    pose: RewTerm | None = None
    air_time: RewTerm | None = None
    foot_clearance: RewTerm | None = None
    foot_swing_height: RewTerm | None = None
    foot_slip: RewTerm | None = None
    soft_landing: RewTerm | None = None
    angular_momentum: RewTerm | None = None

    upright: RewTerm | None = RewTerm(
        func=mdp.upright,
        weight=2.0,
        params={"std": math.sqrt(0.2), "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    body_ang_vel: RewTerm | None = RewTerm(
        func=mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    action_rate_l2: RewTerm | None = RewTerm(func=mdp.action_rate_l2, weight=-1.0)

    # Main objective: track the target yaw rate omega*(phase) (a trapezoid
    # envelope -- see SPIN_* constants in mdp/rewards.py).
    spin_rate_track: RewTerm | None = RewTerm(
        func=microduck_mdp.spin_rate_track,
        weight=6.0,
        params={"command_name": "twist", "std": 1.5, **_ENVELOPE},
    )
    # L1 bootstrap: constant gradient when the gaussian saturates far from target.
    spin_rate_l1: RewTerm | None = RewTerm(
        func=microduck_mdp.spin_rate_l1, weight=0.5, params={"command_name": "twist", **_ENVELOPE}
    )
    # Spin ON THE SPOT, and kill entry momentum. Strong weight: an earlier
    # calibration run showed the trunk translating at ~0.35 m/s
    # (~omega*half-track), the signature of a pivot on a single skate
    # rather than a spin centred on the body -- this is the only term that
    # tells the two apart. Attenuated during the launch ramp [0, ACCEL_END):
    # that's when the robot must push off the ground to inject angular
    # momentum, and the entry momentum (up to 0.3 m/s) must be CONVERTED
    # into rotation -- charging it full price there would oppose the launch.
    # Full price during cruise/brake/rest.
    spin_stay_in_place: RewTerm | None = RewTerm(
        func=microduck_mdp.spin_stay_in_place,
        weight=-3.0,
        params={
            "command_name": "twist",
            "launch_scale": microduck_mdp.SPIN_LAUNCH_DRIFT_SCALE,
            "accel_end": microduck_mdp.SPIN_ACCEL_END,
        },
    )
    # Bootstrap 1: spin via ROLLING (skates in opposite directions), not skidding.
    spin_wheel_differential: RewTerm | None = RewTerm(
        func=microduck_mdp.spin_wheel_differential,
        weight=1.0,
        params={"command_name": "twist", "omega_scale": microduck_mdp.SPIN_WHEEL_OMEGA_SCALE, **_ENVELOPE},
    )
    # Bootstrap 2: leg scissor (decays via curriculum, see microduck_curriculum_cfg.py).
    leg_antisymmetry: RewTerm | None = RewTerm(
        func=microduck_mdp.leg_antisymmetry,
        weight=1.0,
        params={"command_name": "twist", "joint_bases": ("hip_pitch", "knee"), **_ENVELOPE},
    )
    # Both blades on the ground during the spin (no mid-air twist).
    spin_grounded: RewTerm | None = RewTerm(
        func=microduck_mdp.spin_grounded,
        weight=0.5,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", **_ENVELOPE},
    )

    # -- Stability / sim2real -----------------------------------------------
    feet_flat: RewTerm | None = RewTerm(
        func=microduck_mdp.feet_flat_penalty,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
            "sensor_name": "feet_ground_contact",
        },
    )
    self_collisions: RewTerm | None = RewTerm(
        func=mdp.self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision"}
    )
    neck_action_rate_l2: RewTerm | None = RewTerm(func=microduck_mdp.neck_action_rate_l2, weight=-0.5)
    # head_yaw excluded (see NECK_PATTERN_NO_YAW): left free to act as a
    # flywheel to help launch the rotation.
    neck_joint_pos_l2: RewTerm | None = RewTerm(
        func=microduck_mdp.neck_joint_pos_l2, weight=-0.2, params={"pattern": NECK_PATTERN_NO_YAW}
    )
    joint_torques_l2: RewTerm | None = RewTerm(func=microduck_mdp.joint_torques_l2, weight=-1e-3)
