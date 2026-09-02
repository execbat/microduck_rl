"""Command specifications for the Microduck standup task.

``twist``: no locomotion command -- squashed to near-zero noise, kept only
for obs-shape parity. ``head_pose``: real commandable head control (like
velocity/sitstand). ``body_pose``: real commandable trunk delta
(z/roll/pitch tracked; see ``body_pose_tracking`` in
``microduck_rewards_cfg.py``) -- only present when
``ENABLE_BODY_CONTROL`` is on.
"""

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.standup.microduck_flags import (
    BODY_CMD_ALIVE_ANGLE,
    BODY_CMD_ALIVE_XY,
    BODY_CMD_ZERO_PROB,
    ENABLE_BODY_CONTROL,
    EPISODE_LENGTH_S,
)
from mjlab_microduck.tasks.velocity.cfg.commands_cfg import CommandsCfg
from mjlab_microduck.tasks.velocity.microduck_flags import BODY_POSE_CMD_RESAMPLE_S, HEAD_POSE_CMD_RESAMPLE_S
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckCommandsCfg(CommandsCfg):
    twist: microduck_mdp.VelocityCommandCommandOnlyCfg | None = microduck_mdp.VelocityCommandCommandOnlyCfg(
        entity_name="robot",
        resampling_time_range=(EPISODE_LENGTH_S, EPISODE_LENGTH_S * 2),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,  # inherited unchanged from the base velocity command.
        heading_command=False,
        heading_control_stiffness=0.5,  # inherited unchanged.
        debug_vis=False,
        ranges=microduck_mdp.VelocityCommandCommandOnlyCfg.Ranges(
            lin_vel_x=(-0.01, 0.01),
            lin_vel_y=(-0.01, 0.01),
            ang_vel_z=(-0.05, 0.05),
            heading=None,
        ),
    )
    head_pose: microduck_mdp.UniformPoseCommandCfg | None = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),  # neck_pitch
            (-0.05, 0.05),  # head_pitch
            (-0.07, 0.07),  # head_yaw
            (-0.015, 0.015),  # head_roll
        ),
    )
    body_pose: microduck_mdp.UniformPoseCommandCfg | None = (
        microduck_mdp.UniformPoseCommandCfg(
            resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
            zero_command_prob=BODY_CMD_ZERO_PROB,
            ranges=(
                (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY),  # x (m)
                (-BODY_CMD_ALIVE_XY, BODY_CMD_ALIVE_XY),  # y (m)
                (-0.005, 0.005),  # z (m) -- widened by the body_pose_range curriculum
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # roll
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # pitch
                (-BODY_CMD_ALIVE_ANGLE, BODY_CMD_ALIVE_ANGLE),  # yaw
            ),
        )
        if ENABLE_BODY_CONTROL
        else None
    )
