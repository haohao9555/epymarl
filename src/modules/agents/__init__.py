from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .fpo_agent import RNNAgent as FPOAgent

# ------ 新增：注册连续动作 Agent ----------
# -----------------------------------------------------------------------------
from .rnn_continuous_agent import RNNContinuousAgent
# -----------------------------------------------------------------------------

REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["fpo_agent"] = FPOAgent

# ------ 新增 ----------
# -----------------------------------------------------------------------------
REGISTRY["rnn_continuous"] = RNNContinuousAgent
# -----------------------------------------------------------------------------
