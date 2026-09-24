from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner

# ------ 新增：导入连续动作 Learner ----------
# -----------------------------------------------------------------------------
from .ppo_continuous_learner import PPOContinuousLearner
from .policyflow_continuous_learner import PolicyFlowContinuousLearner
# -----------------------------------------------------------------------------

# ------ 新增：MAC-Flow (online, off-policy) learner，独立文件夹 src/macflow/ ----------
# -----------------------------------------------------------------------------
from macflow.mac_flow_learner import MACFlowLearner
from macflow.paper_learner import MACFlowPaperLearner
# -----------------------------------------------------------------------------

# ------ 新增：FPO++（基于 MAFPO fork，逐点 ratio + A<0 时 SPO 平滑目标，见
# fpopp_learner.py）。2026-09-02 起 individual-actors 和 shared-network 两种配
# 置合并进同一个类，由 fpo_individual_agents/fpo_actor_optim_per_agent/
# fpo_anchor_per_agent/critic_type 这几个独立、显式的 config 项分别控制（不再
# 是两个几乎一样的文件），fpopp_shared.yaml 现在也指向这同一个 learner，见该
# 类的 docstring。连续动作是这条线唯一支持的动作空间，文件/类名不再带
# continuous 后缀；离散版本（fpo_discrete_learner.py 等）已删除。这个类当初是
# 从 MAFPOContinuousLearner（mafpo_continuous_learner.py，GitHub 拉取的
# individual-actors 原始线）fork 出来改进 ratio/surrogate 数学的，本质上是它的
# 严格改进版——原文件也已经一并删除，不再单独维护两条几乎重复、只有 ratio 计
# 算方式不同的线。----------
# -----------------------------------------------------------------------------
from .fpopp_learner import FPOPPLearner
from .mafpo_gauss_learner import MAFPOGaussLearner
# -----------------------------------------------------------------------------

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner

# ------ 新增 ----------
# -----------------------------------------------------------------------------
REGISTRY["ppo_continuous_learner"] = PPOContinuousLearner
REGISTRY["policyflow_continuous_learner"] = PolicyFlowContinuousLearner
REGISTRY["mac_flow_learner"] = MACFlowLearner
REGISTRY["macflow_paper_learner"] = MACFlowPaperLearner
REGISTRY["fpopp_learner"] = FPOPPLearner
REGISTRY["mafpo_gauss_learner"] = MAFPOGaussLearner
# -----------------------------------------------------------------------------
