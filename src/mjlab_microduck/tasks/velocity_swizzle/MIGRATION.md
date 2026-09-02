# Manager-based env-cfg restructure — `velocity_swizzle` — status & notes

## What changed

`tasks/microduck_velocity_swizzle_env_cfg.py` (~200 lines, one function
that called `make_microduck_velocity_rollers_env_cfg(play=play)` and
patched the reward/command/curriculum recipe) has been replaced by
`tasks/velocity_swizzle/`, the same IsaacLab-style, one-file-per-manager
package shape used for every other task. Backend is still 100% mjlab —
nothing about the runtime changed, only how the config is *authored*.

**This is genuinely the last of the 12 files** (the 11 task files from the
original folder screenshot, plus `mdp.py`) — `roller_standup`'s own
MIGRATION.md claimed to be last prematurely; `velocity_swizzle` had been
overlooked in that count.

```
tasks/velocity_swizzle/
  microduck_flags.py                     # ANTI_SWIZZLE_REWARDS, HEAD_POSE_INITIAL_RANGES
  microduck_observations_cfg.py          # MicroduckObservationsCfg -- head_command
                                          # becomes a REAL command on both groups
  microduck_commands_cfg.py              # MicroduckCommandsCfg -- symmetrised/widened
                                          # twist ranges + new head_pose command
  microduck_rewards_cfg.py               # MicroduckRewardsCfg(velocity_rollers.MicroduckRewardsCfg)
                                          # -- swizzle/backward-locomotion/heading/head recipe
  microduck_curriculum_cfg.py            # MicroduckCurriculumCfg -- heading swap + head-pose ramp
  microduck_velocity_swizzle_env_cfg.py  # MicroduckVelocitySwizzleEnvCfg(velocity_rollers.MicroduckVelocityRollersEnvCfg)
                                          # + make_microduck_velocity_swizzle_env_cfg()
  microduck_rl_cfg.py                    # MicroduckSwizzleRlCfg (dataclasses.replace of velocity_rollers's own)
  __init__.py
```

No `events_cfg.py`/`terminations_cfg.py` here — both are inherited
completely unchanged from `velocity_rollers` (the original file never
touched them at all), so the env cfg class doesn't even declare those
fields.

## Things worth flagging clearly

- **`pose` is kept but RESCOPED, not just weight-tweaked.** The original
  file's rescoping logic filters `velocity_rollers`'s own `std_standing`/
  `std_walking`/`std_running` dicts (dropping any pattern containing
  "neck"/"head"/"passive") and narrows `asset_cfg` to a leg-only regex.
  Rather than reproduce that filtering logic imperatively at
  configuration-build time, this file just pre-computed the filtered
  dicts once (`_STD_STANDING`/`_STD_WALKING`/`_STD_RUNNING`, leg patterns
  only) and declares the whole `pose` term statically — same end state,
  consistent with how every other reward override in this refactor is
  written. `tests/test_swizzle_head_cfg.py`'s own
  `test_swizzle_head_control_wired` explicitly checks that `pose`'s
  **function** is unchanged (only the params changed) — verified: both
  `velocity_swizzle` and `velocity_rollers`'s `pose` terms use the exact
  same `mdp.variable_posture` function reference.
- **`wheel_speed` gains a `bidirectional=True` param** (mutated on the
  otherwise-inherited term, same weight as `velocity_rollers`) rather than
  being dropped and re-added — this task repurposes negative `cmd_x` as
  "go backward" instead of "brake" (`braking` is dropped accordingly), so
  the wheel-speed reward needs to reward spin in *either* commanded
  direction, not just forward.
- **`heading_command=False` is inherited unchanged** even though the
  design intent ("re-enable heading command... cmd[2] carries the heading
  error to a sampled target") reads like it should flip to `True`. Checked
  the original file line by line: it genuinely never touches that field —
  `RelativeHeadingVelocityCommand` computes `cmd[2]` internally regardless
  of the flag, and only `ranges.ang_vel_z` (the *clip bound* on that
  internally-computed error) actually gets widened. Preserved exactly as
  written rather than "fixed" to match the docstring's phrasing.
- **`play` is a doubly-dead parameter here**: the original function passed
  it straight through to `make_microduck_velocity_rollers_env_cfg(play=play)`,
  which itself never references `play` in its body either (see that
  task's own MIGRATION.md). `make_microduck_velocity_swizzle_env_cfg(play=)`
  keeps accepting (and ignoring) it for calling-convention parity, same as
  every other roller-family task with a dead `play` param.

**Public API is unchanged**: `make_microduck_velocity_swizzle_env_cfg(play=)`
still exists, still returns a real `mjlab.envs.ManagerBasedRlEnvCfg`, same
task ID. `tasks/__init__.py` needed only the one import-path change;
`tests/test_swizzle_head_cfg.py` needed the same one-line fix.

## Verified

- `pyflakes` clean on every new file.
- **`tests/test_swizzle_head_cfg.py`'s own test runs via `pytest` against
  the new package** and passes — checks the head-command wiring, the
  `neck_joint_pos_l2` removal, the `pose` rescoping (both the joint-name
  regex shape and, critically, that the reward *function* itself wasn't
  swapped), and both late-curriculum terms' presence.
- Isolated import of `mjlab_microduck.tasks.velocity_swizzle` against the
  same stub `mjlab` used for the other refactored tasks — imports cleanly,
  `make_microduck_velocity_swizzle_env_cfg(play)` succeeds for both `play`
  values.
- Term counts per manager, cross-checked by hand against the original
  ~200-line file, match exactly: rewards=18 (velocity_rollers's 21, minus
  7 dropped, plus 4 new), events=13 (inherited unchanged), curriculum=8
  (velocity_rollers's 4, plus 4 new), commands=2 (`twist` + `head_pose`),
  terminations=4 (inherited unchanged).
- `bidirectional=True` on `wheel_speed`, symmetrised `lin_vel_x=(-0.6,0.6)`,
  and widened `ang_vel_z=(-0.5,0.5)` all confirmed present on the built
  config.
- Full `mjlab_microduck.tasks` package import succeeds completely, end to
  end — with `velocity_swizzle` migrated, every one of the 12 files
  (11 tasks + `mdp.py`) originally identified for restructuring is now
  done, and the whole package imports with zero errors in the sandbox's
  stub-`mjlab` harness.
- `python -m py_compile` / `ast.parse` over every new file.
- Not run: an actual mjlab/mujoco simulation step — install torch/
  mujoco-warp and run `uv run pytest tests/test_swizzle_head_cfg.py` for
  real, plus a short
  `uv run train Mjlab-Velocity-Swizzle-MicroDuck --env.scene.num-envs 16
  --agent.max-iterations 5` before trusting this in production. Given the
  `SERVO_LEG_JOINTS` bug a real GPU run surfaced in `roller_standup`
  (see that task's MIGRATION.md) — a bug this refactor's stub-import
  verification structurally cannot catch (it needs actual CUDA tensor
  indexing) — a real training smoke-test remains the one verification step
  no task in this whole restructure has had yet, and is worth doing before
  trusting any of the 11 migrated tasks in production, not just this one.
