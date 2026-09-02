"""Command specifications for the Microduck roller_slope task.

Pure balance -- no steering. All fields inherited unchanged from
``velocity_rollers``'s ``RelativeHeadingVelocityCommandCfg`` except
``rel_standing_envs`` (0.0 -> 1.0, every env forced to "standing" -- no
forward-push demand) and ``ranges.lin_vel_x`` (-0.5,0.6 -> 0,0).
"""

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.velocity.cfg.commands_cfg import CommandsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckCommandsCfg(CommandsCfg):
    twist: microduck_mdp.RelativeHeadingVelocityCommandCfg | None = (
        microduck_mdp.RelativeHeadingVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),  # inherited unchanged.
            rel_standing_envs=1.0,
            rel_heading_envs=0.0,  # inherited unchanged.
            rel_forward_envs=0.2,  # inherited unchanged.
            heading_command=False,  # inherited unchanged.
            heading_control_stiffness=0.5,  # inherited unchanged.
            debug_vis=True,  # inherited unchanged.
            ranges=microduck_mdp.RelativeHeadingVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 0.0),
                lin_vel_y=(0.0, 0.0),  # inherited unchanged.
                ang_vel_z=(0.0, 0.0),  # inherited unchanged.
                heading=None,  # inherited unchanged.
            ),
            viz=microduck_mdp.RelativeHeadingVelocityCommandCfg.VizCfg(z_offset=0.5),  # inherited unchanged.
        )
    )
