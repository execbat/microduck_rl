"""Microduck VelStand task — walking + fall recovery, one policy.

REBASED (2026-07, audit follow-up) on the velocity recipe — the proven
walker — instead of the abandoned older recipe the old velstand used. The
2026-07 audit found the old design starved the walk: only ~25% of
experience was clean commanded walking (2/3 prone resets + fallen envs
farming recovery reward for full 20s episodes), the recovery rewards taxed
the gait (always-on posture double-counting, a bounce incentive from
com_upward_velocity below walk height), and the prone init dropped the
robot from 0.20-0.25m (function defaults — a violent uncontrolled impact
opening most episodes).

Design:
  - Walk layer  = ``tasks.velocity``'s ``MicroduckVelocityFlatEnvCfg``/
    ``RoughEnvCfg``, subclassed directly. Everything the good walker has
    (tracking weights, air_time, turn-in-place bucket, fixed command
    ranges, DR/noise/obs) flows in by construction — this file only
    overrides the ``rewards``/``events``/``terminations``/``curriculum``
    manager fields (see ``microduck_*_cfg.py`` next to this file) and the
    robot entity.
  - Robot       = all-collision standup XML (body can physically lie down).
  - Recovery    = a small reward layer GATED on actually-being-fallen
    (trunk z < 0.10m OR tilt > 40deg): contributes exactly zero during
    clean walking, steers only when down. ``upright_progress`` gives an
    orientation gradient everywhere; ``com_upward_velocity`` pays for
    rising.
  - No impact penalties (trunk/head) — the standup specialist's recovery
    pushes off with head/trunk, and an impact penalty would tax exactly
    that strategy.
  - ``joint_torque_rate_l2`` (standup's proven anti-jitter) for transfer
    smoothness — penalises torque CHANGE, never blocks the recovery flip.

Phases (see the run-5/6/7 lessons in ``microduck_flags.py`` for why these
specific boundaries):
  Phase 1 (0 -> 500 iters): ``fell_over`` termination active (70deg) ->
    clean walking first.
  Phase 2 (500+): ``fell_over`` disabled (limit -> pi) so falls become
    recovery opportunities — but ``fallen_too_long`` (8s continuously down)
    recycles failed recoveries instead of letting them farm the full
    episode.
  Phase 3 (1500+): prone-init ramp: face-down first (easier), face-up mixed
    in later, capped at 45% prone so the walking data share stays
    >= ~55% (was 2/3 prone -> ~25% walking share).
"""

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks.velocity.microduck_flags import VELOCITY_PUSH_PLAY_INTERVAL_S
from mjlab_microduck.tasks.velocity.microduck_velocity_env_cfg import (
    MicroduckVelocityFlatEnvCfg,
    MicroduckVelocityRoughEnvCfg,
)
from mjlab_microduck.utils.configclass import configclass

from .microduck_curriculum_cfg import MicroduckCurriculumCfg
from .microduck_events_cfg import MicroduckEventsCfg
from .microduck_rewards_cfg import MicroduckRewardsCfg
from .microduck_terminations_cfg import MicroduckTerminationsCfg


@configclass
class MicroduckVelstandFlatEnvCfg(MicroduckVelocityFlatEnvCfg):
    """Microduck velstand task on a flat ground plane."""

    events: MicroduckEventsCfg = MicroduckEventsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    terminations: MicroduckTerminationsCfg = MicroduckTerminationsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # Full-collision standup XML: trunk/head shells keep their
        # contacts so the robot can physically lie on the ground and push
        # off it (velocity's walk-only robot has minimal non-foot
        # collision geometry).
        self.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}


@configclass
class MicroduckVelstandRoughEnvCfg(MicroduckVelocityRoughEnvCfg):
    """Microduck velstand task on gentle procedural rough terrain."""

    events: MicroduckEventsCfg = MicroduckEventsCfg()
    rewards: MicroduckRewardsCfg = MicroduckRewardsCfg()
    terminations: MicroduckTerminationsCfg = MicroduckTerminationsCfg()
    curriculum: MicroduckCurriculumCfg = MicroduckCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}


@configclass
class MicroduckVelstandFlatEnvCfg_PLAY(MicroduckVelstandFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = VELOCITY_PUSH_PLAY_INTERVAL_S
        # In play mode the curriculum doesn't run, so the fell_over-disable
        # curriculum below never fires -- just drop the termination and the
        # curriculum term outright (matches the original's
        # ``cfg.terminations.pop("fell_over", None)`` / ``if not play:``
        # guard around registering the curriculum term).
        self.terminations.fell_over = None
        self.curriculum.fell_over_disable = None


@configclass
class MicroduckVelstandRoughEnvCfg_PLAY(MicroduckVelstandRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        if self.events.push_robot is not None:
            self.events.push_robot.interval_range_s = VELOCITY_PUSH_PLAY_INTERVAL_S
        self.terminations.fell_over = None
        self.curriculum.fell_over_disable = None

        assert self.scene.terrain is not None
        assert self.scene.terrain.terrain_generator is not None
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5


def make_microduck_velstand_env_cfg(play: bool = False, rough: bool = False):
    """Create the Microduck velstand environment configuration.

    Kept as a drop-in replacement for the old function of the same name:
    same signature, same return type (a real ``mjlab.envs.ManagerBasedRlEnvCfg``,
    via ``.to_mjlab_cfg()``), so gym registration keeps working unmodified.
    """
    cfg_cls = {
        (False, False): MicroduckVelstandFlatEnvCfg,
        (True, False): MicroduckVelstandFlatEnvCfg_PLAY,
        (False, True): MicroduckVelstandRoughEnvCfg,
        (True, True): MicroduckVelstandRoughEnvCfg_PLAY,
    }[(play, rough)]
    return cfg_cls().to_mjlab_cfg()
