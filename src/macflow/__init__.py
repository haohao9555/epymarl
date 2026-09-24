# MAC-Flow (online, off-policy) — self-contained package.
#
# Lives entirely outside controllers/learners/modules/agents so that the
# existing MAFPO code (fpo_actor.py, fpo_mac.py, fpo_critic.py,
# fpo_continuous_learner.py, mafpo*.yaml) is never touched. See README.md in
# this folder for the design writeup this implements.
#
# The only edits made to shared framework files are three additive registry
# lines (one import + one dict entry each) in:
#   controllers/__init__.py   -> "mac_flow_mac"
#   learners/__init__.py      -> "mac_flow_learner"
#   modules/agents/__init__.py -> "mac_flow_one_step_actor"
# The critic is reused as-is (critic_type: "maddpg_critic"), so
# modules/critics/__init__.py is untouched.
