"""Curriculum specifications for the Microduck roller_slope task.

Wholesale replacement: the original file deletes every curriculum term
``velocity_rollers`` has and adds a single new one. Subclasses the plain
base ``CurriculumCfg`` directly (not ``velocity_rollers``'s), since none of
that task's curriculum terms (action_rate_weight, wheel_friction, com_range,
head_com_range) survive here anyway.

Starts on the gentlest slope (2deg) and promotes to steeper (up to 20deg)
once the robot has descended far enough (``terrain_levels_slope``, based on
distance travelled). Only viable once ``descent_speed`` (well, here:
``wheel_glide``) actually makes it move forward -- previously it stayed
still and was never promoted. Learns balance progressively instead of being
thrown straight onto a 20deg slope (where it face-plants).
"""

from mjlab.managers.curriculum_manager import CurriculumTermCfg as CurrTerm

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.locomotion.velocity.cfg.curriculum_cfg import CurriculumCfg
from mjlab_microduck.utils.configclass import configclass


@configclass
class MicroduckCurriculumCfg(CurriculumCfg):
    command_vel: CurrTerm | None = None
    terrain_levels: CurrTerm | None = CurrTerm(func=microduck_mdp.terrain_levels_slope)
