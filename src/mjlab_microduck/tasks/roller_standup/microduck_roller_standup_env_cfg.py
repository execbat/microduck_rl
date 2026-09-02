"""Microduck roller_standup task — getting back up on rollers.

DEDICATED episodic policy: the robot starts on the ground (prone/belly,
supine/back) or already standing, and must get back up on its rollers and
then HOLD the stance. Port of the ``standup`` (walking duck) recipe to the
roller model.

Builds on the roller env (``tasks.velocity_rollers``) -- inherits the
roller robot, sensors, all DR, and the unified 61D observation as-is, so
it's hot-swappable at runtime (--new-cmd-obs). Same pattern as
``roller_slope``: subclasses ``MicroduckVelocityRollersEnvCfg`` directly
(the original called ``make_microduck_velocity_rollers_env_cfg(play=play)``).

See ``microduck_flags.py``/``microduck_rewards_cfg.py``/
``microduck_curriculum_cfg.py`` for the full design rationale (joint-index
remapping, dropped skating rewards, the inverted wheel-friction curriculum,
the sign-bug history on ``gentle_rise``).
"""

from mjlab_microduck.tasks.velocity_rollers.microduck_velocity_rollers_env_cfg import MicroduckVelocityRollersEnvCfg
from mjlab_microduck.utils.configclass import configclass

from .microduck_commands_cfg import MicroduckCommandsCfg
from .microduck_curriculum_cfg import _WHEEL_FRICTION_STAGE0, MicroduckCurriculumCfg
from .microduck_events_cfg import MicroduckEventsCfg
from .microduck_flags import EPISODE_LENGTH_S, play_face_up_overrides, resolve_play_face_up
from .microduck_observations_cfg import MicroduckObservationsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckRollerStandupEnvCfg(MicroduckVelocityRollersEnvCfg):
    """Microduck roller_standup task -- get back up on rollers."""

    observations: MicroduckObservationsCfg = MicroduckObservationsCfg()
    commands: MicroduckCommandsCfg = MicroduckCommandsCfg()
    events: MicroduckEventsCfg = MicroduckEventsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    terminations: MicroduckTerminationsCfg = MicroduckTerminationsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    episode_length_s: float = EPISODE_LENGTH_S

    # Runtime-selected: whether __post_init__ forces a supine-heavy start
    # mix (for eyeballing the hardest recovery case) -- see
    # make_microduck_roller_standup_env_cfg().
    play: bool = False

    def __post_init__(self):
        super().__post_init__()

        # Defensive redundancy (matches the original file's own comment):
        # the curriculum manager runs BEFORE reset events on every reset
        # (including the very first one), and wheel_friction_curriculum
        # itself defaults to its stage 0 -- so this line is never actually
        # load-bearing in practice. It just keeps the event's DEFAULT value
        # consistent with the curriculum's stage 0, in case the curriculum
        # is ever removed later while leaving the event in place.
        if self.events.randomize_wheel_friction is not None:
            self.events.randomize_wheel_friction.params["ranges"] = _WHEEL_FRICTION_STAGE0

        # Play override: force supine (on-the-back) starts so they can be
        # inspected. Writes the probabilities into the EVENT and removes the
        # curriculum: without that, event_param_curriculum (which runs
        # BEFORE reset events) would overwrite them with its stage 0 on the
        # very first reset. Play-only, so training and its easy->hard
        # curriculum are untouched.
        if self.play:
            play_face_up = resolve_play_face_up()
            if play_face_up is not None:
                self.events.set_ground_state.params.update(play_face_up_overrides(play_face_up))
                self.curriculum.ground_state_mix = None


def make_microduck_roller_standup_env_cfg(play: bool = False):
    """Create the Microduck roller_standup environment configuration.

    Kept as a drop-in replacement for the old function of the same name:
    same signature, same return type (a real ``mjlab.envs.ManagerBasedRlEnvCfg``,
    via ``.to_mjlab_cfg()``), so gym registration keeps working unmodified.
    """
    return MicroduckRollerStandupEnvCfg(play=play).to_mjlab_cfg()
