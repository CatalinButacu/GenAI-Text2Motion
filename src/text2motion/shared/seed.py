import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
            ":4096:8"  # deterministic cuBLAS matmul (CUDA >=10.2)
        )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # benchmark picks fastest (nondeterministic) kernels
        torch.use_deterministic_algorithms(True)  # raises if an op has no deterministic impl

    return seed
