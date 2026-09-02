"""Command specifications for the Microduck roller_crouch task.

Unlike ``velocity_rollers`` (which this task shares a robot/obs layout
with), the original file built its command from mjlab's raw base
``make_velocity_env_cfg()`` directly, not from velocity_rollers's modified
one -- so the inherited defaults here are the STANDARD velocity command
ranges (``tasks/velocity/cfg/commands_cfg.py``), not velocity_rollers's
``RelativeHeadingVelocityCommandCfg`` ones. Only ``rel_standing_envs``/
``rel_heading_envs`` are overridden, plus the command TYPE, which is
replaced entirely by the same cyclic phase encoder ``ground_pick`` uses:

    command = [cos(2*pi*phase), sin(2*pi*phase), 0]

``randomize_phase=False``: every episode starts standing (phase 0),
matching deployment, where the button press starts the cycle at phase 0.
This avoids the policy learning to "stay low" from episodes that start
already crouched.
"""

import math

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.roller_crouch.microduck_flags import CROUCH_PERIOD
from mjlab_microduck.tasks.velocity.cfg.commands_cfg import CommandsCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckCommandsCfg(CommandsCfg):
    twist: microduck_mdp.GroundPickPhaseCommandCfg | None = microduck_mdp.GroundPickPhaseCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),  # inherited unchanged from the base velocity command.
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        rel_forward_envs=0.2,  # inherited unchanged.
        heading_command=True,  # inherited unchanged.
        heading_control_stiffness=0.5,  # inherited unchanged.
        debug_vis=True,
        ranges=microduck_mdp.GroundPickPhaseCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-math.pi, math.pi),
        ),
        period=CROUCH_PERIOD,
        randomize_phase=False,
    )
