import pytest

from text2motion.app.stages.data import (
    PROTOCOL_SEED,
    freeze_protocol_splits,
    require_protocol_run,
    verify_protocol_splits,
)


def write_splits(root):
    for name, text in (("train.txt", "A\nB\n"), ("val.txt", "C\n"), ("test.txt", "D\n")):
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_a_protocol_run_uses_the_protocol_seed():
    require_protocol_run(PROTOCOL_SEED, False, "train-tokenizer")


def test_a_run_with_another_seed_is_refused_unless_exploratory():
    with pytest.raises(ValueError, match="would use seed 2027"):
        require_protocol_run(2027, False, "train-tokenizer")
    require_protocol_run(2027, True, "train-tokenizer")


def test_the_split_hashes_pass_when_the_files_are_unchanged(tmp_path):
    write_splits(tmp_path)
    freeze_protocol_splits(tmp_path)
    verify_protocol_splits(tmp_path, False)


def test_a_changed_split_file_is_refused(tmp_path):
    write_splits(tmp_path)
    freeze_protocol_splits(tmp_path)
    (tmp_path / "test.txt").write_text("D\nE\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test.txt changed since it was frozen"):
        verify_protocol_splits(tmp_path, False)


def test_a_missing_split_record_is_refused(tmp_path):
    write_splits(tmp_path)
    with pytest.raises(FileNotFoundError, match="freeze the splits"):
        verify_protocol_splits(tmp_path, False)


def test_an_exploratory_run_skips_the_split_check(tmp_path):
    write_splits(tmp_path)
    verify_protocol_splits(tmp_path, True)
