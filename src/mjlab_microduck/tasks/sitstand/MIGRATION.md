# Manager-based env-cfg restructure — `sitstand` — status & notes

## What changed

`tasks/microduck_sitstand_env_cfg.py` (~930 lines, one function that built a
base cfg via mjlab's own `make_velocity_env_cfg()` and then imperatively
poked `cfg.rewards["name"] = ...`) has been replaced by `tasks/sitstand/`,
the same IsaacLab-style, one-file-per-manager package shape used for the
other tasks. Backend is still 100% mjlab — nothing about the runtime
changed, only how the config is *authored*.

```
tasks/sitstand/
  microduck_flags.py                 # ENABLE_*/ranges + task constants (SIT_Z, STAND_Z,
                                      # SITTING_TARGET_OVERRIDES, POSTURE_*, SIM_NCONMAX, ...)
  microduck_scene_cfg.py             # robot + feet/self-collision sensors
  microduck_observations_cfg.py      # MicroduckObservationsCfg(ObservationsCfg)
  microduck_commands_cfg.py          # MicroduckCommandsCfg(CommandsCfg) -- TWO active
                                      # commands: twist (SitStandCommandCfg) + head_pose
  microduck_rewards_cfg.py           # MicroduckRewardsCfg(RewardsCfg) -- the
                                      # posture-conditioned single-target stack
  microduck_events_cfg.py            # MicroduckEventsCfg(EventsCfg)
  microduck_terminations_cfg.py      # MicroduckTerminationsCfg(TerminationsCfg) -- fell_over dropped
  microduck_curriculum_cfg.py        # MicroduckCurriculumCfg(CurriculumCfg)
  microduck_sitstand_env_cfg.py      # MicroduckSitStandFlatEnvCfg / RoughEnvCfg /
                                      # *_PLAY, mirroring ground_pick's Flat/Rough/_PLAY
                                      # split, + make_microduck_sitstand_env_cfg()
  microduck_rl_cfg.py                # MicroduckSitStandRlCfg (RSL-RL PPO)
  __init__.py
```

Full 4-class `Flat`/`Flat_PLAY`/`Rough`/`Rough_PLAY` hierarchy (like
`ground_pick`, `velocity`) — real terrain variants and a `play` flag that
actually changes behavior (push interval).

## Task-specific things worth flagging clearly

- **Two active commands.** This is the first migrated task with more than
  one live command: `twist` (repurposed as the sit/stand posture flag via
  `SitStandCommandCfg`, not actual velocity tracking) *and* `head_pose`
  (real commandable head control, like `velocity`/`standup`). Unlike
  `ball_kick`/`roulade` (which squash `twist` to near-zero noise for obs
  parity only), `sitstand`'s `twist` command is functionally load-bearing —
  its value literally drives which posture the reward stack targets. The
  base `CommandsCfg` only declares `twist`, so `head_pose` is a genuinely
  *new* field on `MicroduckCommandsCfg`, not an override.
- **`fell_over` dropped, `nan_state` added** — same treatment as `roulade`:
  wobbles/tips during a sit/stand transition must play out (the policy
  needs to experience the impact/upright costs), not truncate the episode.
- **Contact-solver hardening (`nconmax=200`, `iterations=30`,
  `ls_iterations=50`) is applied UNCONDITIONALLY** — not just on rough
  terrain like `velocity`'s equivalent tuning. The seated pose alone
  (full-collision trunk + folded legs + head all in close contact)
  overflows the default solver regardless of what's underfoot, so this
  lives in the base `MicroduckSitStandFlatEnvCfg.__post_init__`, inherited
  unchanged by the Rough variant, rather than being Rough-only.
- **Reward-sign warning preserved verbatim as a file docstring**: the
  original file has an explicit, hard-won warning that
  `descent_speed`/`rise_speed`/`gentle_motion` need POSITIVE weights
  because the underlying functions already return negative values — a
  sign-convention bug here previously trained a "reward for violence"
  policy. Kept as the top-of-file docstring in
  `microduck_rewards_cfg.py`, not just an inline comment, so it's the first
  thing a future editor of that file sees.

**Public API is unchanged**: `make_microduck_sitstand_env_cfg(play=,
rough=)` still exists, still returns a real `mjlab.envs.ManagerBasedRlEnvCfg`,
same task IDs. `tasks/__init__.py` needed only the one import-path change;
no test file imports this task's factory function directly.

## Verified

- `pyflakes` clean on every new file.
- Isolated import of `mjlab_microduck.tasks.sitstand` against the same stub
  `mjlab` used for the other refactored tasks — imports cleanly,
  `make_microduck_sitstand_env_cfg(play, rough)` succeeds for all 4
  combinations.
- Term counts per manager, cross-checked by hand against the original
  ~930-line file, match exactly in all 4 combinations: rewards=20,
  obs(actor)=8, obs(critic)=12, events=14, curriculum=8 (flat)/9 (rough),
  commands=2, terminations=3 (`fell_over` confirmed genuinely absent).
- `nconmax=200` confirmed present in all 4 combinations (not just Rough).
- `commands.keys() == ["twist", "head_pose"]` confirmed for both active commands.
- Terrain-generator aliasing isolation check (same as `velocity`/
  `ground_pick`) — confirmed no leak into the shared
  `MICRODUCK_ROUGH_TERRAINS_CFG` module constant.
- Full `mjlab_microduck.tasks` package import reaches the same
  pre-existing, unrelated wall as before this change
  (`microduck_standup_env_cfg.py`, the sandbox's deliberate canary for
  not-yet-migrated files) — confirms nothing downstream of `sitstand` broke.
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step — install torch/
  mujoco-warp and run a short
  `uv run train Mjlab-SitStand-Flat-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 5` before trusting this in production. (No
  `tests/test_sitstand_*.py` file exists in this repo to run alongside it.)

## Recipe reminder for the remaining environments

Same recipe as the other MIGRATION.md files, refined by this task's data
points:

1. **A task can add a genuinely new command field**, not just override the
   base `twist` — check every `cfg.commands["X"] = ...` assignment (not
   just `cfg.commands["twist"] = ...`) when scanning a task's command
   section, and add the new field to `MicroduckCommandsCfg` (it won't be
   inherited from the base, since the base doesn't have it).
2. **Distinguish "unconditional tuning that happens to only matter on one
   terrain" from "Rough-only tuning"** before deciding whether a
   `self.sim.*` override belongs in the Flat base or the Rough subclass —
   `sitstand`'s contact-solver hardening is unconditional (needed on flat
   ground too, because of the seated pose) even though it looks
   superficially similar to `velocity`'s Rough-only sim tuning. Check what
   actually causes the problem the tuning fixes, not just which section of
   the original file it appears near.
3. When a comment in the original file reads like a **hard-won warning**
   (explicit sign-convention notes, "after any change check X stays <= 0",
   references to a specific bug that shipped once) — promote it to the new
   file's module docstring rather than leaving it as an inline comment
   easy to scroll past.
