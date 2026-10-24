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


def __write_source(path, frames: int, fields: dict) -> np.ndarray:
    trans = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
    np.savez(
        path,
        root_orient=np.zeros((frames, 3), dtype=np.float32),
        pose_body=np.zeros((frames, 63), dtype=np.float32),
        pose_hand=np.zeros((frames, 90), dtype=np.float32),
        trans=trans,
        betas=np.zeros(16, dtype=np.float32),
        gender="neutral",
        **fields,
    )
    return trans


def __recording_extractor(model: RecordingSmplx) -> AmassPoseExtractor:
    extractor = AmassPoseExtractor.__new__(AmassPoseExtractor)
    extractor.device = torch.device("cpu")
    extractor._beta_count = 16
    extractor._joint_count = 22
    extractor._fps = 20
    extractor._chunk_size = 2
    extractor._models = {gender: model for gender in ("neutral", "male", "female")}
    extractor._expression_count = 10
    return extractor


def test_a_120_fps_source_keeps_every_sixth_frame(tmp_path):
    source = tmp_path / "clip.npz"
    trans = __write_source(source, 12, {"mocap_frame_rate": np.float32(120)})

    joints = __recording_extractor(RecordingSmplx()).extract(source)

    assert joints is not None
    np.testing.assert_array_equal(joints[:, 0], trans[::6][:, [0, 2, 1]])


@pytest.mark.parametrize("rate", [59.94, 50.0, 30.0, 10.0])
def test_a_rate_that_is_not_a_multiple_of_20_stops_the_extraction(tmp_path, rate):
    source = tmp_path / "clip.npz"
    __write_source(source, 12, {"mocap_framerate": np.float32(rate)})

    with pytest.raises(ValueError, match="not an integer multiple"):
        __recording_extractor(RecordingSmplx()).extract(source)


def test_motion_without_a_rate_stops_and_a_shape_file_is_skipped(tmp_path):
    motion = tmp_path / "clip.npz"
    __write_source(motion, 12, {})
    shape = tmp_path / "female_stagei.npz"
    np.savez(shape, betas=np.zeros(16, dtype=np.float32), gender="female")
    extractor = __recording_extractor(RecordingSmplx())

    with pytest.raises(ValueError, match="no frame-rate field"):
        extractor.extract(motion)
    assert extractor.extract(shape) is None


def test_amass_extractor_crosses_backend_boundaries_once_per_clip(tmp_path):
    frames = 5
    source = tmp_path / "clip.npz"
    trans = __write_source(source, frames, {"mocap_frame_rate": np.float32(20)})

    model = RecordingSmplx()
    extractor = __recording_extractor(model)
    original_tensor = extractor.to_float_tensor
    transferred_shapes: list[tuple[int, ...]] = []

    def record_transfer(array: np.ndarray) -> torch.Tensor:
        transferred_shapes.append(array.shape)
        return original_tensor(array)

    extractor.to_float_tensor = record_transfer

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
