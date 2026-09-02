"""Command specifications for the Microduck roulade task.

The task has no locomotion command -- the robot rolls in place -- so
``twist`` is squashed to near-zero noise, kept only for obs-shape parity
with the other tasks.
"""

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.roulade.microduck_flags import EPISODE_LENGTH_S
from mjlab_microduck.tasks.locomotion.velocity.cfg.commands_cfg import CommandsCfg
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
