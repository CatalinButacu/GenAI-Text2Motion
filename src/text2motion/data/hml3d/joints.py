import numpy as np
import torch

from text2motion.data.hml3d.feature import recover_from_ric
from text2motion.data.hml3d.param_util import joints_num, t2m_kinematic_chain


def recover_skeleton(feat263: np.ndarray | torch.Tensor) -> np.ndarray:
    data = (
        feat263
        if isinstance(feat263, torch.Tensor)
        else torch.as_tensor(feat263, dtype=torch.float32)
    )
    return recover_from_ric(data.float(), joints_num).cpu().numpy()


def kinematic_bones() -> np.ndarray:
    bones = [
        [chain[i], chain[i + 1]] for chain in t2m_kinematic_chain for i in range(len(chain) - 1)
    ]
    return np.array(bones, dtype=np.int64)
