REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC

# ------ 新增：注册连续动作 MAC ----------
# -----------------------------------------------------------------------------
from .continuous_mac import ContinuousMAC
from .policyflow_mac import PolicyFlowMAC
# -----------------------------------------------------------------------------

# ------ 新增：MAC-Flow (online, off-policy) 控制器，独立文件夹 src/macflow/ ----------
# -----------------------------------------------------------------------------
from macflow.mac_flow_mac import MACFlowMAC
from macflow.paper_mac import MACFlowPaperMAC
# -----------------------------------------------------------------------------

# ------ 新增：MAFPO（GitHub 独立演化线，individual actors，见 mafpo_mac.py）----------
# -----------------------------------------------------------------------------
from .mafpo_mac import MAFPOMAC
from .mafpo_gauss_mac import MAFPOGaussMAC
# -----------------------------------------------------------------------------

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC

# ------ 新增 ----------
# -----------------------------------------------------------------------------
REGISTRY["continuous_mac"] = ContinuousMAC
REGISTRY["policyflow_mac"] = PolicyFlowMAC
REGISTRY["mac_flow_mac"] = MACFlowMAC
REGISTRY["macflow_paper_mac"] = MACFlowPaperMAC
REGISTRY["mafpo_mac"] = MAFPOMAC
REGISTRY["mafpo_gauss_mac"] = MAFPOGaussMAC
# -----------------------------------------------------------------------------