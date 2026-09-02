"""Custom action terms for the microduck MDP.

Empty on purpose: every microduck task currently uses mjlab's built-in
``mjlab.envs.mdp.actions.JointPositionActionCfg`` directly (see the
``actions_cfg.py`` / ``ActionsCfg`` files under ``tasks/locomotion/velocity/cfg`` and
each task's env cfg). This file is here so the package layout matches the
other manager categories, and as the natural home for a custom
``ActionTermCfg`` if one is ever added."""

# (no terms yet)
