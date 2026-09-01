"""MDP functions for microduck tasks.

This used to be a single ~7200-line ``mdp.py`` file. It's now a package, one
file per manager category (observations / actions / rewards / events /
terminations / curriculum / commands), plus ``_common.py`` for shared private
helpers and ``patches.py`` for the import-time monkey-patches this module has
always applied to mjlab/rsl_rl internals.

Public API is unchanged on purpose: every task/env-cfg file in this repo does

    from mjlab_microduck.tasks import mdp as microduck_mdp
    ...
    func=microduck_mdp.some_reward_or_event_or_whatever

and every name that used to be reachable as ``microduck_mdp.<name>`` still is
-- this file just re-exports everything from the split-out submodules into
the same flat namespace, so no env-cfg file needed to change.
"""

# Side-effecting only: patches RewardManager/PPO/the ONNX exporter as soon as
# this package is imported. Must run before anything else below (matches the
# original file's top-to-bottom execution order).
from . import patches as _patches  # noqa: F401

from .observations import *  # noqa: F401, F403
from .actions import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .events import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
from .curriculum import *  # noqa: F401, F403
from .commands import *  # noqa: F401, F403
