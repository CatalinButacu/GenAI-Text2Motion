"""Reproducible seeding. Log the seed on every run -- the controlled twin (ADR 0001) must be
comparable across architectures, which requires fixed, recorded seeds."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return seed
