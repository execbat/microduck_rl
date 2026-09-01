"""Microduck BallKick task — kick a ball forward with one foot.

Episodic policy: the robot starts STANDING (HOME pose + noise) with a 70mm /
15g ball sitting just in front of its kicking foot (``kick_foot`` below --
train a right-footed and a left-footed policy as two separate runs). The goal
is to kick the ball forward (robot's heading at reset) at ``BALL_TARGET_SPEED``
while keeping balance and staying robust to external pushes, then settle back
into a clean stand.

Key design decisions:
  - The policy is BLIND to the ball (no ball obs in the actor): the real
    robot has no ball sensing -- the operator aims the robot at the ball.
    Robustness to placement error comes from +-2cm ball-position DR at reset
    instead. The CRITIC does see ball pos/vel (asymmetric actor-critic) so
    the value function can anticipate the kick payoff.
  - No phase command: the kick reward is available from t=0 and an earlier
    kick collects more ball-rolling reward, so the policy kicks immediately.
    At deployment: hard ONNX swap to this policy (a la jump/ground-pick), it
    kicks, then auto-swaps back after ~2s.
  - Right-foot kick is enforced geometrically + economically: the ball
    spawns at the right toe, and an always-on LEFT-foot-grounded reward
    makes the left leg the support leg (lifting it costs reward every step;
    anti-hop).
  - Kick reward is LINEAR in ball forward speed (clamped at the target), not
    a saturating tanh -- "as hard as possible up to a target" needs gradient
    at high speeds.
  - Obs layout is the unified 61D actor layout (twist + zero-padded head/body
    command slots) so the runtime can hard-swap ONNX files with one buffer.

DR / noise / regularization: velocity-parity, copied from the standup env
(which is itself matched to velocity -- the recipe with proven transfer).

This is this task's flagship (and only) terrain variant -- flat only ("a
ball on rough terrain is a different task", per the original file). Unlike
``tasks/velocity``, there's no Flat/Rough split; instead ``kick_foot`` and
``play`` are plain dataclass fields on the env cfg itself, since they're
just parameter variations rather than structurally different tasks.
"""

from mjlab.terrains import TerrainEntityCfg

from mjlab_microduck.utils.configclass import configclass

from mjlab_microduck.tasks.velocity.locomotion_velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .microduck_commands_cfg import MicroduckCommandsCfg
from .microduck_curriculum_cfg import MicroduckCurriculumCfg
from .microduck_events_cfg import MicroduckEventsCfg
from .microduck_flags import EPISODE_LENGTH_S, KICK_FOOT, VELOCITY_PUSH_PLAY_INTERVAL_S, ball_offset_y_of
from .microduck_observations_cfg import MicroduckObservationsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_scene_cfg import MICRODUCK_BALL_CFG, MICRODUCK_STANDUP_ROBOT_CFG, build_sensors
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckBallKickEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Microduck BallKick task -- flat terrain, standing kick."""

    kick_foot: str = KICK_FOOT
    play: bool = False

    observations: MicroduckObservationsCfg = MicroduckObservationsCfg()
    commands: MicroduckCommandsCfg = MicroduckCommandsCfg()
    events: MicroduckEventsCfg = MicroduckEventsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    terminations: MicroduckTerminationsCfg = MicroduckTerminationsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    episode_length_s: float = EPISODE_LENGTH_S

    def __post_init__(self):
        assert self.kick_foot in ("right", "left")

        # Full-collision robot (same spec as standup/ground-pick): the ball
        # must be able to contact the whole leg, not just the foot pads of
        # the walk model. Robot MUST stay the first entity (set_random_
        # ground_state and the base reset events write robot root state at
        # qpos[:, 0:7]).
        self.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG, "ball": MICRODUCK_BALL_CFG}
        self.scene.sensors = build_sensors(self.kick_foot)
        self.scene.terrain = TerrainEntityCfg(terrain_type="plane")
        self.viewer.body_name = "trunk_base"

        # Extra contact headroom for the ball (ball-terrain + ball-robot
        # contacts on top of the full-collision robot's budget).
        self.sim.nconmax = 50

        self.actions.joint_pos.scale = 1.0

        # Ball spawn side follows the kicking foot -- fill in the y-offset
        # sign now that self.kick_foot is known (see microduck_events_cfg.py).
        x_offset, _ = self.events.reset_ball.params["offset"]
        self.events.reset_ball.params["offset"] = (x_offset, ball_offset_y_of(self.kick_foot))

        if self.play and self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = VELOCITY_PUSH_PLAY_INTERVAL_S


def make_microduck_ball_kick_env_cfg(
    play: bool = False,
    kick_foot: str | None = None,
):
    """Create the Microduck BallKick environment configuration.

    ``kick_foot`` overrides the module-level ``KICK_FOOT`` flag (used by
    tests); normal training just sets the flag in ``microduck_flags.py``.

    Kept as a drop-in replacement for the old function of the same name:
    same signature, same return type (a real ``mjlab.envs.ManagerBasedRlEnvCfg``,
    via ``.to_mjlab_cfg()``), so gym registration keeps working unmodified.
    """
    cfg = MicroduckBallKickEnvCfg(play=play, kick_foot=kick_foot or KICK_FOOT)
    return cfg.to_mjlab_cfg()
