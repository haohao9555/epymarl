from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent

# ------ 新增：注册连续动作 Agent ----------
# -----------------------------------------------------------------------------
from .rnn_continuous_agent import RNNContinuousAgent
from .fpo_actor import FPOActor
from .fpo_discrete_agent import FPODiscreteAgent
# -----------------------------------------------------------------------------

REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent

# ------ 新增 ----------
# -----------------------------------------------------------------------------
REGISTRY["rnn_continuous"] = RNNContinuousAgent
REGISTRY["fpo_actor"] = FPOActor
REGISTRY["fpo_discrete_agent"] = FPODiscreteAgent
# -----------------------------------------------------------------------------
