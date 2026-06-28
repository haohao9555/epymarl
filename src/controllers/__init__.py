REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC

# ------ 新增：注册连续动作 MAC ----------
# -----------------------------------------------------------------------------
from .continuous_mac import ContinuousMAC
from .fpo_mac import FPOMAC
from .fpo_discrete_mac import FPODiscreteMAC
# -----------------------------------------------------------------------------

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC

# ------ 新增 ----------
# -----------------------------------------------------------------------------
REGISTRY["continuous_mac"] = ContinuousMAC
REGISTRY["fpo_mac"] = FPOMAC
REGISTRY["fpo_discrete_mac"] = FPODiscreteMAC
# -----------------------------------------------------------------------------