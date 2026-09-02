"""Microduck roulade (forward-roll) task — attempt 3, run 2.

Episodic policy: robot starts standing, rolls forward over the flat top of
its head, and lands back on its feet. Triggered at deployment like sit/
standup (policy switch = roll starts immediately; no phase clock, no
reference motion).

RUN-2 REWORK (run 1 learned a violent ballistic "breakdance" whip -- optimal
under the run-1 rewards: same 2*pi, sooner, no cost): rotation now only
counts while the robot touches the ground (support-gated accumulator -- a
roulade never leaves the floor), the landing annuity requires an
over-the-head contact latch, paid progress rate is capped at a measured
physical rate (faster forfeits the excess), an overspeed penalty taxes
genuine whips, and the impact/smoothness penalties are active from step 0
(discovery in this env is easy; style is the scarce resource, not
exploration).

Design (see ``microduck_rewards_cfg.py`` for the full reward-by-reward
history):
  - ONE dense progress signal -- paid increments of the max-so-far
    cumulative forward rotation (potential-based: a full roll pays 2*pi
    worth total, camping anywhere pays zero per step).
  - Landing rewards gated on ROLL COMPLETION (rotation frontier >= ~260deg),
    not on a clock -- "do nothing" earns nothing, the standing spawn cannot
    farm them, and no upright/height pressure ever opposes the flip.
  - Reverse curriculum via mid-roll spawns (the trick that fixed face-up
    recovery in standup): a slice of episodes starts 50-185deg into the
    roll, tucked, with forward angular momentum, accumulator pre-set to the
    spawn angle. The second half of a roulade IS the face-up recovery
    problem, which is known to be learnable.
  - Elan hook for later: ``set_roulade_state``'s ``forward_vel_range`` gives
    standing spawns an initial forward base velocity -- see
    ``ROULADE_FORWARD_VEL_RANGE`` in ``microduck_flags.py``. (0, 0) =
    standstill-only.

DR / obs / regularisers mirror the standup env (velocity sim2real parity),
with the motion-blockers kept near zero during discovery and introduced
late by curriculum -- the roll IS a large angular-velocity, large-impact
event; taxing attempts prevents discovery (proven twice on standup).

Flat terrain only, no play-specific behavior (``play`` is accepted and
ignored, matching the original file's dead parameter -- same situation as
``velocity_rollers``/``roller_crouch``).
"""

from mjlab.terrains import TerrainEntityCfg

from mjlab_microduck.tasks.velocity.locomotion_velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from mjlab_microduck.utils.configclass import configclass

from .microduck_commands_cfg import MicroduckCommandsCfg
from .microduck_curriculum_cfg import MicroduckCurriculumCfg
from .microduck_events_cfg import MicroduckEventsCfg
from .microduck_flags import EPISODE_LENGTH_S
from .microduck_observations_cfg import MicroduckObservationsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_scene_cfg import FEET_GROUND_CFG, HEAD_GROUND_CFG, MICRODUCK_STANDUP_ROBOT_CFG, ROBOT_GROUND_CFG, SELF_COLLISION_CFG
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckRouladeEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Microduck roulade task -- flat terrain, standing forward roll."""

    observations: MicroduckObservationsCfg = MicroduckObservationsCfg()
    commands: MicroduckCommandsCfg = MicroduckCommandsCfg()
    events: MicroduckEventsCfg = MicroduckEventsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    terminations: MicroduckTerminationsCfg = MicroduckTerminationsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    episode_length_s: float = EPISODE_LENGTH_S

    def __post_init__(self):
        self.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
        self.scene.sensors = (FEET_GROUND_CFG, SELF_COLLISION_CFG, HEAD_GROUND_CFG, ROBOT_GROUND_CFG)
        self.scene.terrain = TerrainEntityCfg(terrain_type="plane")
        self.viewer.body_name = "trunk_base"

        self.actions.joint_pos.scale = 1.0


def make_microduck_roulade_env_cfg(play: bool = False):
    """Create the Microduck roulade environment configuration.

    ``play`` is accepted (and unused) purely for signature parity with the
    old function of the same name -- there's no play-specific behavior in
    this task.

    Kept as a drop-in replacement: same signature, same return type (a real
    ``mjlab.envs.ManagerBasedRlEnvCfg``, via ``.to_mjlab_cfg()``), so gym
    registration keeps working unmodified.
    """
    return MicroduckRouladeEnvCfg().to_mjlab_cfg()
