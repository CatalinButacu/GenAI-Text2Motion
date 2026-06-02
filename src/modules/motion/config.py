"""Compatibility shim -- all names now live in src.shared.config / src.shared.constants."""
from src.shared.config import (  # noqa: F401
    DataConfig,
    ModelConfig,
    MotionConfig,
    TrainingConfig,
    load_yaml_config,
)
from src.shared.constants import MAMBA, RVQ, SSM, MambaSpec, RvqSpec, SsmSpec  # noqa: F401
