import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from text2motion.motion import representation
from text2motion.motion.kinematics import Skeleton
from text2motion.tokenization.contracts import (
    FiniteScalarStackLayout,
    ResidualVectorQuantizationConfig,
    TokenizerConfig,
)
from text2motion.tokenization.network.finite_scalar import FiniteScalarQuantizationTokenizer
from text2motion.tokenization.network.residual import ResidualVectorQuantizationTokenizer

GUARD_FILE = Path(__file__).with_name("refactor_guard.json")
SEED = 2026
DECIMALS = 5
TOKENIZERS = {
    "fsq_grouped": lambda: FiniteScalarQuantizationTokenizer(
        TokenizerConfig(width=32, num_quantizers=2, fsq_levels=(4, 4), n_resblocks=1)
    ),
    "fsq_residual": lambda: FiniteScalarQuantizationTokenizer(
        TokenizerConfig(
            width=32,
            num_quantizers=2,
            fsq_levels=(4, 4),
            n_resblocks=1,
            quantizer=FiniteScalarStackLayout.RESIDUAL,
        )
    ),
    "fsq_causal_heading": lambda: FiniteScalarQuantizationTokenizer(
        TokenizerConfig(
            width=32,
            num_quantizers=2,
            fsq_levels=(4, 4),
            n_resblocks=1,
            causal=True,
            heading_correction=True,
        )
    ),
    "rvq": lambda: ResidualVectorQuantizationTokenizer(
        ResidualVectorQuantizationConfig(
            width=32, n_resblocks=1, num_quantizers=2, codebook_size=16, code_dim=32
        )
    ),
}


def digest(values: np.ndarray) -> str:
    rounded = np.round(np.asarray(values, dtype=np.float64), DECIMALS) + 0.0
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def synthetic_joints() -> np.ndarray:
    frames = 30
    generator = np.random.default_rng(0)
    joints = np.tile(representation.RAW_OFFSETS.astype(np.float64)[None] * 0.3, (frames, 1, 1))
    for chain in representation.KINEMATIC_CHAINS:
        for index in range(1, len(chain)):
            joints[:, chain[index]] += joints[:, chain[index - 1]]
    time = np.linspace(0, 1, frames)[:, None, None]
    return joints + 0.05 * np.sin(2 * np.pi * time) + 0.02 * generator.standard_normal(joints.shape)


def motion_record() -> dict:
    joints = synthetic_joints()
    params = representation.feature_params(representation.target_offsets(joints))
    features = representation.encode_joints(joints, params)
    offsets = torch.from_numpy(representation.RAW_OFFSETS.astype(np.float32))
    skeleton = Skeleton(offsets, representation.KINEMATIC_CHAINS, "cpu")
    return {
        "encode_joints": [digest(part) for part in features],
        "inverse_kinematics": digest(skeleton.inverse_kinematics_np(joints, (2, 1, 17, 16))),
    }


def tokenizer_record(name: str) -> dict:
    torch.manual_seed(SEED)
    network = TOKENIZERS[name]().eval()
    motion = torch.randn(2, 32, 263, generator=torch.Generator().manual_seed(SEED))
    with torch.no_grad():
        output = network(motion)
    return {
        "parameter_order": [parameter for parameter, _ in network.named_parameters()],
        "state_dict": {key: list(value.shape) for key, value in network.state_dict().items()},
        "token_indices": output.token_indices.tolist(),
        "reconstruction": digest(output.reconstruction.numpy()),
        "commitment_loss": digest(output.commitment_loss.numpy()),
    }


def measure() -> dict:
    return {"motion": motion_record()} | {name: tokenizer_record(name) for name in TOKENIZERS}


def test_motion_and_tokenizer_outputs_match_the_recorded_guard():
    recorded = json.loads(GUARD_FILE.read_text(encoding="utf-8"))
    measured = measure()

    assert set(measured) == set(recorded)
    for name, record in recorded.items():
        assert set(measured[name]) == set(record), f"{name}: parts changed"
        for part, expected in record.items():
            assert measured[name][part] == expected, f"{name}: {part} changed"
