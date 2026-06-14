from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
#---------------新增：导入 FPO learner------------------------------
from .fpo_learner import PPOLearner as FPOLearner
#----------------------
from .ppo_learner import PPOLearner

# ------ 新增：导入连续动作 Learner ----------
# -----------------------------------------------------------------------------
from .ppo_continuous_learner import PPOContinuousLearner
# -----------------------------------------------------------------------------

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
#---------------新增：注册 FPO learner------------------------------
REGISTRY["fpo_learner"] = FPOLearner
#----------------------
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner

# ------ 新增 ----------
# -----------------------------------------------------------------------------
REGISTRY["ppo_continuous_learner"] = PPOContinuousLearner
# -----------------------------------------------------------------------------
