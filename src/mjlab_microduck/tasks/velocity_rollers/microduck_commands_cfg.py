"""Command specifications for the Microduck velocity_rollers task.

``cmd_x`` semantics: 0 = coast, >0 = push to accelerate, <0 = brake to stop.
``cmd[2]`` = heading error, handled internally by
``RelativeHeadingVelocityCommand`` (turning is currently disabled --
``ang_vel_z`` clipped to zero -- straight-line skating focus; see
``heading_hold`` in rewards for the yaw-drift corrective instead).
"""

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity.cfg.commands_cfg import CommandsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckCommandsCfg(CommandsCfg):
    twist: microduck_mdp.RelativeHeadingVelocityCommandCfg | None = (
        microduck_mdp.RelativeHeadingVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),  # inherited unchanged from the base velocity command.
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            rel_forward_envs=0.2,  # inherited unchanged from the base velocity command.
            heading_command=False,  # RelativeHeadingVelocityCommand handles heading internally.
            heading_control_stiffness=0.5,  # inherited unchanged.
            debug_vis=True,
            ranges=microduck_mdp.RelativeHeadingVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 0.6),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),  # clip limit for cmd[2] heading error; 0 = no turning demand.
                heading=None,  # must be None when heading_command=False.
            ),
            viz=microduck_mdp.RelativeHeadingVelocityCommandCfg.VizCfg(z_offset=0.5),
        )
    )
