from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent

# ------ 新增：注册连续动作 Agent ----------
# -----------------------------------------------------------------------------
from .rnn_continuous_agent import RNNContinuousAgent
from .policyflow_actor import PolicyFlowActor
# -----------------------------------------------------------------------------

# ------ 新增：MAFPO（GitHub 独立演化线，individual actors，见 mafpo_actor.py）----------
# -----------------------------------------------------------------------------
from .mafpo_actor import MAFPOActor
# -----------------------------------------------------------------------------

REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent

# ------ 新增 ----------
# -----------------------------------------------------------------------------
REGISTRY["rnn_continuous"] = RNNContinuousAgent
REGISTRY["policyflow_actor"] = PolicyFlowActor
REGISTRY["mafpo_actor"] = MAFPOActor
# -----------------------------------------------------------------------------
