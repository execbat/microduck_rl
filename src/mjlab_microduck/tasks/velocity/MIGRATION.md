# Manager-based env-cfg restructure — status & how to continue

## What changed

`tasks/microduck_velocity_env_cfg.py` (~950 lines, one function that built a
base cfg and imperatively poked `cfg.rewards["name"] = ...`) has been
replaced by `tasks/velocity/`, an IsaacLab-style, one-file-per-manager
package. Backend is still 100% mjlab — nothing about the runtime changed,
only how the config is *authored*.

```
tasks/velocity/
  cfg/                          # generic task defaults (mirrors mjlab's own
    observations_cfg.py         # make_velocity_env_cfg()), robot-agnostic
    actions_cfg.py
    commands_cfg.py
    events_cfg.py
    rewards_cfg.py
    terminations_cfg.py
    curriculum_cfg.py
  locomotion_velocity_env_cfg.py  # LocomotionVelocityRoughEnvCfg: assembles
                                   # the cfg/ package into one @configclass,
                                   # exactly the shape you sketched:
                                   #   scene, observations, rewards, commands,
                                   #   events, terminations, curriculum
                                   # + to_mjlab_cfg() -> mjlab.envs.ManagerBasedRlEnvCfg
  microduck_flags.py             # every ENABLE_*/tunable constant, single source of truth
  microduck_scene_cfg.py         # robot cfg, sensors, rough-terrain generator, spec_fn
  microduck_observations_cfg.py  # MicroduckObservationsCfg(ObservationsCfg)
  microduck_commands_cfg.py      # MicroduckCommandsCfg(CommandsCfg)
  microduck_events_cfg.py        # MicroduckEventsCfg(EventsCfg)
  microduck_rewards_cfg.py       # MicroduckRewardsCfg(RewardsCfg)
  microduck_terminations_cfg.py  # MicroduckTerminationsCfg(TerminationsCfg)
  microduck_curriculum_cfg.py    # MicroduckCurriculumCfg(CurriculumCfg)
  microduck_velocity_env_cfg.py  # MicroduckVelocityFlatEnvCfg / RoughEnvCfg /
                                  # *_PLAY, mirroring G1RoughEnvCfg/_PLAY/
                                  # G1FlatEnvCfg, + make_microduck_velocity_env_cfg()
  microduck_rl_cfg.py             # MicroduckRlCfg (RSL-RL PPO)
```

`mjlab_microduck/utils/configclass.py` and `.../utils/manager_compat.py` are
new, reusable infrastructure: a small vendored `@configclass` (no IsaacLab/
IsaacSim dependency) plus the glue that turns a tree of configclass instances
into the plain `dict[str, TermCfg]` mjlab's managers actually want. Every
other task can reuse these two files as-is.

**Public API is unchanged**: `make_microduck_velocity_env_cfg(play=, rough=)`
still exists, still returns a real `mjlab.envs.ManagerBasedRlEnvCfg`, same
task IDs. `tasks/__init__.py` and `tasks/backlash.py` needed exactly one
import-path change each; no other file changes.

## Two behavioral fixes that fell out of the restructure (not intentional
## scope, just consequences of per-instance config trees)

1. **Reward/command mutation no longer leaks across sibling env cfgs.** The
   old code built every task from the *same* `make_velocity_env_cfg()` call
   and then mutated the shared term objects in place (hence the scattered
   `deepcopy(...)` calls). Every `@configclass` instance now gets its own
   deep-copied terms automatically, so those deepcopy workarounds are gone
   and the underlying sharing bug can't come back.
2. **`MICRODUCK_ROUGH_TERRAINS_CFG` mutation no longer leaks into the shared
   module-level constant.** `MicroduckVelocityRoughEnvCfg_PLAY` shrinking
   `num_rows`/`num_cols` for visibility used to alias the same
   `TerrainGeneratorCfg` object across every env built from it; it's wrapped
   in `dataclasses.replace(...)` now (see `microduck_velocity_env_cfg.py`),
   the same pattern mjlab's own `velocity_env_cfg.py` uses for the same
   reason.

## Verified

- Standalone tests of `configclass`/`manager_compat` (instance isolation,
  nested-class auto-decoration, override/`None`-disable semantics,
  `__post_init__` chaining).
- Full import of `mjlab_microduck.tasks.velocity` against a stub `mjlab`
  (real `mjlab` needs torch/mujoco-warp, too heavy for this sandbox) —
  imports cleanly, `make_microduck_velocity_env_cfg(play, rough)` succeeds
  for all 4 combinations.
- Term counts per manager, cross-checked by hand against the original
  950-line file, match exactly in all 4 combinations: rewards=16,
  obs(actor)=8, obs(critic)=13, events=13, curriculum=7 (flat)/8 (rough),
  commands=3, terminations=4.
- `python -m py_compile` over the full `src/` tree.
- Not run: an actual mjlab/mujoco simulation step (needs the real heavy deps
  — torch, mujoco-warp, warp-lang — install them and run
  `pytest tests/test_head_pose_bias.py tests/test_obs_nan_guard.py` to check
  that too, since both were updated to import from the new location).

## Recipe for the other 10 environments

Every other `tasks/microduck_*_env_cfg.py` file follows the same shape as the
old velocity file (`make_microduck_X_env_cfg(play, rough) -> ManagerBasedRlEnvCfg`,
built by patching a base cfg). To migrate one:

1. `mkdir tasks/X/`, copy the `cfg/` folder from `tasks/velocity/cfg/` if the
   task is also a locomotion/velocity-family task (same generic defaults);
   otherwise write fresh generic defaults for that task's manager terms.
2. Split the old file's body by manager into `microduck_X_flags.py`,
   `microduck_X_scene_cfg.py`, `microduck_X_observations_cfg.py`, etc. —
   same mechanical transform as `microduck_velocity_env_cfg.py` → this
   package: each `cfg.rewards["name"] = RewTerm(...)` line becomes a class
   attribute `name: RewTerm | None = RewTerm(...)` on a `MicroduckXRewardsCfg`
   subclass; each `del cfg.rewards["name"]` / conditional becomes
   `name: RewTerm | None = None`.
3. Assemble in `microduck_X_env_cfg.py` following
   `MicroduckVelocityFlatEnvCfg` / `RoughEnvCfg` / `*_PLAY` as the template.
4. Update the one `from .microduck_X_env_cfg import ...` line in
   `tasks/__init__.py` to `from .X import ...`.

Several tasks (`microduck_velstand_env_cfg.py`, `microduck_standup_env_cfg.py`,
...) build on top of `make_microduck_velocity_env_cfg()` and patch further —
those become straightforward subclasses of `MicroduckVelocityFlatEnvCfg`/
`MicroduckVelocityRoughEnvCfg` instead, which should shrink them
considerably.
