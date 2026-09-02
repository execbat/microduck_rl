"""Microduck roller SWIZZLE task — clean classic swizzle.

A separate roller task producing a CLASSIC SWIZZLE: both blades stay on the
ground, the legs spread out and pull back in SYMMETRICALLY (hourglass
pattern), propelling the duck forward. Simpler / more stable alternative to
the alternating stride (``Mjlab-Velocity-Flat-MicroDuck-Rollers``), which
does not transfer well to the real robot. The stride env (``tasks/velocity_rollers``)
is left untouched.

Approach A (see docs/superpowers/specs/2026-07-23-swizzle-env-design.md):
the base roller recipe NATURALLY converges to a swizzle, so this task
reuses the stride env wholesale (robot, 61D obs, command, full DR,
curricula, sim2real -- deploys identically with ``--roller``) by
subclassing its env cfg class directly, and only swaps the reward recipe
-- see ``microduck_rewards_cfg.py``/``microduck_curriculum_cfg.py`` for the
full swizzle/backward-locomotion/heading/head-control design.
"""

from mjlab_microduck.tasks.velocity_rollers.microduck_velocity_rollers_env_cfg import MicroduckVelocityRollersEnvCfg
from mjlab_microduck.utils.configclass import configclass

from .microduck_commands_cfg import MicroduckCommandsCfg
from .microduck_curriculum_cfg import MicroduckCurriculumCfg
from .microduck_observations_cfg import MicroduckObservationsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg


@configclass
class MicroduckVelocitySwizzleEnvCfg(MicroduckVelocityRollersEnvCfg):
    """Microduck velocity_swizzle task -- symmetric hourglass gait."""

    observations: MicroduckObservationsCfg = MicroduckObservationsCfg()
    commands: MicroduckCommandsCfg = MicroduckCommandsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    # events/terminations/scene/sensors/DR/robot are all inherited unchanged
    # from velocity_rollers -- the original file never touched them.


def make_microduck_velocity_swizzle_env_cfg(play: bool = False):
    """Create the Microduck velocity_swizzle environment configuration.

    ``play`` is accepted (and unused, same as ``velocity_rollers``'s own
    function it wraps) purely for signature parity with the old function of
    the same name.

    Kept as a drop-in replacement: same signature, same return type (a real
    ``mjlab.envs.ManagerBasedRlEnvCfg``, via ``.to_mjlab_cfg()``), so gym
    registration keeps working unmodified.
    """
    return MicroduckVelocitySwizzleEnvCfg().to_mjlab_cfg()
