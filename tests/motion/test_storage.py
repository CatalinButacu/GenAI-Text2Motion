from pathlib import Path

import pytest
import torch

from text2motion.motion.storage import atomic_torch_save


def test_a_failed_save_keeps_the_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "state.pt"
    atomic_torch_save({"epoch": 1}, path)

    def failing_save(state, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError, match="disk full"):
        atomic_torch_save({"epoch": 2}, path)
    monkeypatch.undo()

    assert torch.load(path)["epoch"] == 1
    assert list(tmp_path.iterdir()) == [path]
