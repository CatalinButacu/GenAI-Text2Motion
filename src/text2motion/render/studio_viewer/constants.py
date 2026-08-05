from pathlib import Path

SKIN_COLOR = (0.86, 0.72, 0.61, 1.0)
SKY_COLOR = (240 / 255, 182 / 255, 182 / 255, 1.0)
GENDERS = ("neutral", "male", "female")
BACKBONES = ("transformer", "mamba")
N_BETAS = 10
LIVE_STEPS = 20
LIVE_DEBOUNCE_S = 0.35
ROOT = Path(__file__).resolve().parents[4]
MODEL_REGISTRY = ROOT / "configs" / "demo_models.yaml"
