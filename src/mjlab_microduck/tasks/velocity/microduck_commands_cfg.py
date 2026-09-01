"""Command specifications for the Microduck velocity task."""

import math

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.utils.configclass import configclass

from .cfg.commands_cfg import CommandsCfg
from .microduck_flags import BODY_POSE_CMD_RESAMPLE_S, HEAD_POSE_CMD_RESAMPLE_S, TURN_IN_PLACE_FRACTION


@configclass
class MicroduckCommandsCfg(CommandsCfg):
    """Command terms for the Microduck velocity task."""

    # Modest, FIXED command ranges (no widening curriculum): a ramp to
    # lin +-0.4 / ang +-2.0 outpaced the robot's capability and tracked a
    # post-iter-1000 reward/episode-length decline. ang +-1.0 is the big
    # change -- it makes turning learnable.
    twist: UniformVelocityCommandCfg | None = microduck_mdp.VelocityCommandCommandOnlyCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,  # small but non-zero from the start, ramped up by curriculum
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
            # heading_command=True requires a heading range even though this
            # task doesn't widen it (inherited unchanged from the mjlab base).
            heading=(-math.pi, math.pi),
        ),
        viz=UniformVelocityCommandCfg.VizCfg(z_offset=0.5),
        # Explicit turn-in-place bucket: lin=0, |ang| in [0.4*max, max].
        rel_turn_in_place_envs=TURN_IN_PLACE_FRACTION,
    )

    # Head pose command (4D deltas from HOME, in joint order: neck_pitch,
    # head_pitch, head_yaw, head_roll). Tracked as a primary reward -- see
    # head_pose_tracking in microduck_rewards_cfg.py. Initial ranges are
    # small non-zero so input neurons stay alive from step 0; the
    # head_pose_range curriculum (microduck_curriculum_cfg.py) widens them.
    # Per-joint final caps reflect each joint's mechanically reachable delta
    # from HOME (XML limits minus HOME offset, with ~10% safety margin):
    #   neck_pitch / head_pitch: +-1.10 rad (limit +-pi/2 with HOME=+-20deg)
    #   head_yaw                : +-1.40 rad (limit +-pi/2 with HOME=0)
    #   head_roll               : +-0.31 rad (limit +-20deg)
    head_pose: microduck_mdp.UniformPoseCommandCfg | None = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),  # neck_pitch
            (-0.05, 0.05),  # head_pitch
            (-0.07, 0.07),  # head_yaw
            (-0.015, 0.015),  # head_roll (tighter -- much smaller mechanical range)
        ),
    )

    # Body pose command (6D delta from nominal standing: [x, y, z, roll,
    # pitch, yaw]). Vel env carries this slot for runtime obs-shape parity;
    # tracked at a tiny weight to keep the input neurons alive but not steer
    # the policy. The standup env raises the weight + widens the ranges.
    body_pose: microduck_mdp.UniformPoseCommandCfg | None = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),  # roll (rad)
            (-0.05, 0.05),  # pitch (rad)
            (-0.05, 0.05),  # yaw (rad)
        ),
    )
