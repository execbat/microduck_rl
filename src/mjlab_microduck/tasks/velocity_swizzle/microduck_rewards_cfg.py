"""Reward specifications for the Microduck velocity_swizzle task."""

from mjlab.managers.reward_manager import RewardTermCfg as RewTerm
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity_rollers.microduck_rewards_cfg import MicroduckRewardsCfg as _RollersRewardsCfg
from mjlab_microduck.utils.configclass import configclass

# LEG-ONLY std dicts: velocity_rollers's own _STD_STANDING/_STD_WALKING/
# _STD_RUNNING with the neck/head/passive patterns dropped, matching the
# leg-only asset_cfg below (see the ``pose`` override's docstring).
_STD_STANDING = {
    r".*hip_yaw.*": 0.05,
    r".*hip_roll.*": 0.05,
    r".*hip_pitch.*": 0.05,
    r".*knee.*": 0.05,
    r".*ankle.*": 0.05,
}
_STD_WALKING = {
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.6,
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.25,
}
_STD_RUNNING = {
    r".*hip_yaw.*": 0.5,
    r".*hip_roll.*": 0.8,
    r".*hip_pitch.*": 0.8,
    r".*knee.*": 0.8,
    r".*ankle.*": 0.5,
}


@configclass
class MicroduckRewardsCfg(_RollersRewardsCfg):
    """Reward terms for the Microduck velocity_swizzle task.

    Drops the stride/anti-swizzle terms from ``velocity_rollers``'s 21-term
    set (``single_support``/``glide``/``skating_air_time``/
    ``gait_symmetry``/``hip_roll_neutral``) plus ``braking`` (backward
    locomotion replaces "negative = brake" with "negative = go backward")
    and ``neck_joint_pos_l2`` (would fight ``head_pose_tracking`` for the
    neck/head joints -- ``pose`` is rescoped to legs only instead).
    Everything else velocity_rollers had (``upright``, ``body_ang_vel``,
    ``angular_momentum``, ``action_rate_l2``, ``com_height_target``,
    ``self_collisions``, ``feet_flat``, ``neck_action_rate_l2``,
    ``joint_torques_l2``, ``action_over_limit``, ``forward_lean``,
    ``heading_hold``) flows through inherited unchanged.
    """

    single_support: RewTerm | None = None
    glide: RewTerm | None = None
    skating_air_time: RewTerm | None = None
    gait_symmetry: RewTerm | None = None
    hip_roll_neutral: RewTerm | None = None
    braking: RewTerm | None = None
    neck_joint_pos_l2: RewTerm | None = None

    # Legs mirror each other (the swizzle's defining symmetry).
    leg_symmetry: RewTerm | None = RewTerm(
        func=microduck_mdp.leg_symmetry_reward, weight=2.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    # Keep both blades on the ground (classic swizzle: no lifting).
    grounded: RewTerm | None = RewTerm(
        func=microduck_mdp.grounded_reward,
        weight=1.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )

    # -- Backward locomotion: cmd_x < 0 means GO BACKWARD (not brake) -------
    # wheel_speed rewards wheel spin in the COMMANDED direction (fwd for +,
    # back for -); braking is dropped above (negative no longer means
    # "stop"); the command range is symmetrised (see microduck_commands_cfg.py)
    # so forward and backward get equal push range. To stop, command cmd_x ~
    # 0 (coast). grounded (above) uses |cmd_x| so it holds the blades down
    # both ways.
    wheel_speed: RewTerm | None = RewTerm(
        func=microduck_mdp.wheel_speed_reward,
        weight=10.0,  # inherited unchanged from velocity_rollers.
        params={"command_name": "twist", "vel_scale": 0.3, "bidirectional": True},
    )

    # -- Heading: go STRAIGHT first, then FOLLOW a commanded direction ------
    # velocity_rollers disables heading (ang_vel_z=(0,0), heading_hold only,
    # no heading_tracking). Here cmd[2] carries the heading error to a
    # sampled target (ang_vel_z range widened, see microduck_commands_cfg.py)
    # and heading_tracking is added (starts at 0, ramped by curriculum):
    #   phase 1 (straight): heading_hold dominant, heading_tracking off
    #   phase 2 (follow):   heading_hold -> 0, heading_tracking -> up
    # heading_hold itself is inherited unchanged (weight 1.0, matches the
    # heading_hold_weight curriculum's step-0 value).
    heading_tracking: RewTerm | None = RewTerm(
        func=microduck_mdp.heading_tracking_reward,
        weight=0.0,  # ramped up by the heading_tracking_weight curriculum (must match its step-0 value)
        params={"command_name": "twist", "std": 0.5},
    )

    # -- Head-pose control (Y button): the policy produces the head pose ----
    # Weight 0 here -- ramped in LATE by curriculum so it doesn't disturb
    # the swizzle before it's solid.
    head_pose_tracking: RewTerm | None = RewTerm(
        func=microduck_mdp.head_pose_tracking,
        weight=0.0,  # ramped up by the head_pose_tracking_weight curriculum (must match its step-0 value)
        params={"command_name": "head_pose", "std": 0.5},
    )

    # ``pose`` (inherited from velocity_rollers, kept) is RESCOPED to LEG
    # joints only -- reconciling the two HOME-pullers that would otherwise
    # fight head_pose_tracking: (1) neck_joint_pos_l2 pulled neck/head to
    # HOME, dropped above; (2) this pose reward's std dicts and asset_cfg
    # included neck/head/passive patterns, narrowed here to the leg-only
    # scope (same weight/func/other params as velocity_rollers's own).
    pose: RewTerm | None = RewTerm(
        func=mdp.variable_posture,
        weight=2.0,  # inherited unchanged from velocity_rollers.
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)),
            "command_name": "twist",
            "std_standing": _STD_STANDING,
            "std_walking": _STD_WALKING,
            "std_running": _STD_RUNNING,
            "walking_threshold": 0.01,  # inherited unchanged.
            "running_threshold": 0.5,  # inherited unchanged.
        },
    )
