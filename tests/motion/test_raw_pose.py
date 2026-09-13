import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from text2motion.app.config import load_config
from text2motion.motion.amass import AmassPoseExtractor


class RecordingSmplx:
    def __init__(self) -> None:
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, **inputs: torch.Tensor) -> SimpleNamespace:
        self.calls.append(inputs)
        joints = inputs["transl"][:, None, :].expand(-1, 22, -1)
        return SimpleNamespace(joints=joints)


def test_amass_extractor_crosses_backend_boundaries_once_per_clip(tmp_path):
    frames = 5
    trans = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
    source = tmp_path / "clip.npz"
    np.savez(
        source,
        mocap_frame_rate=np.float32(20),
        root_orient=np.zeros((frames, 3), dtype=np.float32),
        pose_body=np.zeros((frames, 63), dtype=np.float32),
        pose_hand=np.zeros((frames, 90), dtype=np.float32),
        trans=trans,
        betas=np.zeros(16, dtype=np.float32),
        gender="neutral",
    )

    model = RecordingSmplx()
    extractor = AmassPoseExtractor.__new__(AmassPoseExtractor)
    extractor.device = torch.device("cpu")
    extractor._beta_count = 16
    extractor._joint_count = 22
    extractor._fps = 20
    extractor._chunk_size = 2
    extractor._models = {gender: model for gender in ("neutral", "male", "female")}
    extractor._expression_count = 10
    original_tensor = extractor._tensor
    transferred_shapes: list[tuple[int, ...]] = []

    def record_transfer(array: np.ndarray) -> torch.Tensor:
        transferred_shapes.append(array.shape)
        return original_tensor(array)

    extractor._tensor = record_transfer

    joints = extractor.extract(source)

    assert joints is not None
    assert joints.dtype == np.float32
    assert joints.shape == (frames, 22, 3)
    np.testing.assert_array_equal(joints[:, 0], trans[:, [0, 2, 1]])
    assert transferred_shapes == [(5, 3), (5, 63), (5, 90), (5, 3), (16,)]
    assert [call["transl"].shape[0] for call in model.calls] == [2, 2, 1]
    assert all(
        isinstance(value, torch.Tensor) and value.device.type == "cpu"
        for call in model.calls
        for value in call.values()
    )


@pytest.mark.slow
def test_smplx_amass_extractor_real_file():
    cfg = load_config("configs/default.yaml")
    smplx_dir = cfg.paths.smplx_models
    amass_dir = cfg.paths.amass_dir
    if not smplx_dir or not Path(smplx_dir).exists():
        pytest.skip("SMPL-X models unavailable")
    if not amass_dir or not Path(amass_dir).exists():
        pytest.skip("AMASS unavailable")

    ext = AmassPoseExtractor(Path(smplx_dir))

    src = None
    for root, _, files in os.walk(amass_dir):
        npz = [f for f in files if f.endswith(".npz")]
        if npz:
            src = Path(root) / npz[0]
            break
    if src is None:
        pytest.skip("no AMASS npz found")

    joints = ext.extract(src)
    assert joints is not None
    assert joints.ndim == 3 and joints.shape[1:] == (22, 3)
    assert np.isfinite(joints).all()
