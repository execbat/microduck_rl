# Manager-based env-cfg restructure — `ball_kick` — status & notes

## What changed

`tasks/microduck_ball_kick_env_cfg.py` (~660 lines, one function that built
a base cfg via `make_velocity_env_cfg()` and then imperatively poked
`cfg.rewards["name"] = ...`) has been replaced by `tasks/ball_kick/`, the
same IsaacLab-style, one-file-per-manager package shape used for
`tasks/velocity/`. Backend is still 100% mjlab — nothing about the runtime
changed, only how the config is *authored*.

```
tasks/ball_kick/
  microduck_flags.py             # KICK_FOOT, every ENABLE_*/range, task constants
                                  # (support_foot_of / ball_offset_y_of helpers)
  microduck_scene_cfg.py         # robot+ball entities; build_sensors(kick_foot)
                                  # -- the support-foot sensor's pattern depends
                                  # on which foot is kicking, so unlike velocity's
                                  # static sensor objects this is a factory fn
  microduck_observations_cfg.py  # MicroduckObservationsCfg(ObservationsCfg)
  microduck_commands_cfg.py      # MicroduckCommandsCfg(CommandsCfg)
  microduck_events_cfg.py        # MicroduckEventsCfg(EventsCfg)
  microduck_rewards_cfg.py       # MicroduckRewardsCfg(RewardsCfg)
  microduck_terminations_cfg.py  # MicroduckTerminationsCfg(TerminationsCfg)
  microduck_curriculum_cfg.py    # MicroduckCurriculumCfg(CurriculumCfg)
  microduck_ball_kick_env_cfg.py # MicroduckBallKickEnvCfg(LocomotionVelocityRoughEnvCfg)
                                  # + make_microduck_ball_kick_env_cfg()
  microduck_rl_cfg.py            # MicroduckBallKickRlCfg (RSL-RL PPO)
  __init__.py
```

No new `cfg/` folder here — `ball_kick` reuses the *same* generic base
component files as `tasks/velocity/cfg/` (it's built on
`make_velocity_env_cfg()` too, just like the original file was), by
subclassing `tasks.velocity.locomotion_velocity_env_cfg.LocomotionVelocity
RoughEnvCfg` directly. This is the "several tasks build on top of
`make_microduck_velocity_env_cfg()`" case the velocity MIGRATION.md flagged
as the easy path.

**One structural difference from `velocity`:** this task has no Rough/Flat
split (flat terrain only — "a ball on rough terrain is a different task",
per the original docstring) and no separately-shaped `_PLAY` variant class.
`kick_foot` and `play` were plain function parameters in the old
`make_microduck_ball_kick_env_cfg(play, kick_foot)`, so they became plain
dataclass fields on `MicroduckBallKickEnvCfg` itself
(`kick_foot: str = KICK_FOOT`, `play: bool = False`) instead of separate
subclasses — there's no structural task variant to give its own class, just
parameter variation, so one class is the right shape.

**Public API is unchanged**: `make_microduck_ball_kick_env_cfg(play=,
kick_foot=)` still exists, still returns a real
`mjlab.envs.ManagerBasedRlEnvCfg`, same task IDs
(`Mjlab-BallKick-Flat-MicroDuck`, `Mjlab-BallKick-Flat-Backlash-MicroDuck`).
`tasks/__init__.py` needed exactly one import-path change
(`.microduck_ball_kick_env_cfg` → `.ball_kick`); no other file changes
(`scripts/infer_policy.py` only *mentions* the old path in a comment, not an
import — left as-is).

## One correctness-relevant detail preserved carefully: event ordering

The original code has a load-bearing comment: `reset_ball` **must** run
after `set_ground_state`, because the ball's spawn position is derived from
the robot's final reset pose, and mjlab runs same-mode events in dict
insertion order. In the old file that "dict insertion order" was implicit
in which line ran when. In the new `@configclass`-based `EventsCfg`, dict
insertion order is instead the dataclass **field declaration order**
(inherited fields first, in the base class's order; new fields after, in
subclass declaration order — see `manager_compat.group_to_dict`). So
`microduck_events_cfg.py` declares `set_ground_state` before `reset_ball` in
the class body, with a comment calling out why the order can't be
reshuffled. Same treatment applies to `reset_robot_joints`/`foot_friction`
(overridden in place, so they keep the base class's original position) vs.
the newly-added events (appended after, in original file order).

## Verified

- `pyflakes` clean on every new file (after two rounds of fixes: one unused
  import, one leftover parameter-shadowing cleanup).
- Isolated import of `mjlab_microduck.tasks.ball_kick` against the same stub
  `mjlab` used for `velocity` — imports cleanly,
  `make_microduck_ball_kick_env_cfg(play, kick_foot)` succeeds for all 4
  `play`×`kick_foot` combinations.
- Term counts per manager, cross-checked by hand against the original
  660-line file, match exactly in all 4 combinations: rewards=12,
  obs(actor)=8, obs(critic)=14, events=15, curriculum=4, commands=1,
  terminations=4.
- **Instance isolation check specific to this task**: `kick_foot="right"`
  vs. `kick_foot="left"` produce independently-correct, non-interfering
  results — opposite-signed ball spawn offset and mirrored support-foot
  sensor pattern (`^left_foot_collision$` vs. `^right_foot_collision$`) —
  confirming the per-instance `build_sensors(kick_foot)` factory and the
  `__post_init__` offset fixup don't leak state between two
  `MicroduckBallKickEnvCfg` instances built with different feet.
- Full `mjlab_microduck.tasks` package import (the same thing
  `tasks/__init__.py`'s registration does) reaches the identical,
  pre-existing, unrelated wall as before this change
  (`slope_terrain.py` / `roller_slope` env, a stub-completeness gap in the
  sandbox's lightweight `mjlab` stand-in, nothing to do with `ball_kick`) —
  confirms nothing downstream of `ball_kick`'s registration broke.
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step (needs the real heavy deps
  — torch, mujoco-warp, warp-lang — install them and run a short
  `uv run train Mjlab-BallKick-Flat-MicroDuck --env.scene.num-envs 16
  --max-iterations 5` smoke run before trusting this in production, the
  same recommendation as for `velocity`).

## Recipe reminder for the remaining environments

Same recipe as the velocity MIGRATION.md, refined by two data points now:

1. If the task builds on `make_microduck_velocity_env_cfg()` /
   `make_velocity_env_cfg()` and doesn't need its own rough-terrain
   sub-terrain config, **don't re-create a `cfg/` folder** — subclass
   `tasks.velocity.locomotion_velocity_env_cfg.LocomotionVelocityRoughEnvCfg`
   (or, if it explicitly calls `make_microduck_velocity_env_cfg()` itself
   like `microduck_velstand_env_cfg.py` does, subclass
   `tasks.velocity.MicroduckVelocityFlatEnvCfg`/`RoughEnvCfg` instead) and
   reuse `tasks/velocity/cfg/*` directly.
2. Before splitting rewards/events/curriculum into files, **grep the target
   file for any comment about ordering** ("must come after", "insertion
   order", "before the X curriculum removes it", etc.) — those are the
   spots where dataclass field declaration order in the new
   `@configclass`-based group is load-bearing, not just cosmetic. Get the
   field order right in one pass rather than discovering it via a runtime
   bug later.
3. If the old function took parameters beyond `play`/`rough` (like
   `kick_foot`), check whether each one selects between structurally
   different variants (→ separate subclasses, `velocity`-style) or is a
   plain value substitution (→ a dataclass field on the single env cfg
   class, `ball_kick`-style, consumed in `__post_init__`).
