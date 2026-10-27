import numpy as np
import pytest
import torch

from text2motion.motion.contracts import Split
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM
from text2motion.motion.storage import FEATURES_DIR, SPLIT_FILES
from text2motion.tokenization.contracts import TokenizerReconstruction
from text2motion.tokenization.metrics import evaluate_tokenizer_reconstruction
from text2motion.tokenization.nn_model import TokenizerNetwork


############################################################
#       IdentityTokenizer
############################################################
class IdentityTokenizer(TokenizerNetwork):
    def forward(self, motion: torch.Tensor) -> TokenizerReconstruction:
        empty = motion.new_zeros(())
        return TokenizerReconstruction(motion, empty.long(), empty)


def __dataset(root, clips: dict[str, int], listed: str):
    features_dir = root / FEATURES_DIR
    features_dir.mkdir()
    generator = np.random.default_rng(0)
    for clip_id, frames in clips.items():
        features = generator.standard_normal((frames, DIM)).astype(np.float32) * 0.1
        np.save(features_dir / f"{clip_id}.npy", features)
    (root / SPLIT_FILES[Split.VALIDATION]).write_text(listed)
    return MotionScaler(mean=np.zeros(DIM, np.float32), std=np.ones(DIM, np.float32))


def test_a_perfect_reconstruction_scores_zero_and_skips_short_clips(tmp_path):
    scaler = __dataset(
        tmp_path, {"000001": 40, "000002": 50, "000003": 6}, "000001\nM000001\n000002\n000003\n"
    )
    scored: list[int] = []

    def count_clips(reference, reconstructed):
        scored.append(len(reference))
        return 0.5

    metrics = evaluate_tokenizer_reconstruction(
        IdentityTokenizer(),
        tmp_path,
        scaler,
        max_frames=30,
        split=Split.VALIDATION,
        reconstruction_fid_scorer=count_clips,
    )
    assert metrics["clips"] == 2
    assert metrics["mpjpe_mm"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["feature_l2"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["recon_fid"] == 0.5
    assert scored == [2]


def test_a_listed_clip_without_a_feature_file_raises(tmp_path):
    scaler = __dataset(tmp_path, {"000001": 40}, "000001\n000009\n")
    with pytest.raises(FileNotFoundError, match="000009"):
        evaluate_tokenizer_reconstruction(
            IdentityTokenizer(), tmp_path, scaler, max_frames=30, split=Split.VALIDATION
        )
