from pathlib import Path

import pytest
import torch

from text2motion.motion.contracts import Split
from text2motion.motion.storage import (
    FAILURE_DUMP_FILE,
    SPLIT_FILES,
    atomic_torch_save,
    dump_on_failure,
    split_clip_ids,
)


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


def test_split_clip_ids_strip_the_mirror_prefix_and_keep_the_first_order(tmp_path):
    (tmp_path / SPLIT_FILES[Split.VALIDATION]).write_text("000002\nM000002\n\n 000001 \nM000003\n")
    assert split_clip_ids(tmp_path, Split.VALIDATION) == ["000002", "000001", "000003"]


def test_split_clip_ids_without_the_split_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="split file not found"):
        split_clip_ids(tmp_path, Split.TEST)


@pytest.mark.parametrize("enabled", [True, False])
def test_a_failure_in_the_block_writes_a_dump_and_raises_again(tmp_path, enabled):
    module = torch.nn.Linear(2, 1)
    module.weight.grad = torch.full_like(module.weight, float("nan"))
    path = tmp_path / FAILURE_DUMP_FILE

    with (
        pytest.raises(RuntimeError, match="boom"),
        dump_on_failure(
            path if enabled else None,
            lambda: ({"epoch": 3}, {"motion": torch.ones(1)}, 7, module),
        ),
    ):
        raise RuntimeError("boom")

    if not enabled:
        assert not path.exists()
        return
    dump = torch.load(path, weights_only=True)
    assert dump["epoch"] == 3
    assert dump["epoch_step"] == 7
    assert dump["non_finite"] == ["weight.grad"]
    assert "boom" in dump["error"]
    torch.testing.assert_close(dump["inputs"]["motion"], torch.ones(1))
