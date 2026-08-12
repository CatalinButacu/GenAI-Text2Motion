import pickle

import numpy as np
import pytest

from text2motion.train.train_pretrain import TokenPack, collate, split_keys_by_clip


def make_pack(tmp_path, clips=("clipA", "clipB", "clipC", "clipD"), windows=3):
    path = tmp_path / "pack.npz"
    arrays = {}
    for c_index, clip in enumerate(clips):
        for w in range(windows):
            arrays[f"{clip}__{w}"] = np.full((4 + w, 2), c_index, dtype=np.int64)
    np.savez(path, **arrays)
    return path


def test_windows_of_one_clip_never_straddle_the_split(tmp_path):
    pack = make_pack(tmp_path)
    keys = list(np.load(pack).keys())

    train, val = split_keys_by_clip(keys, 0.5, seed=2026)

    assert train and val
    train_clips = {k.rsplit("__", 1)[0] for k in train}
    val_clips = {k.rsplit("__", 1)[0] for k in val}
    assert not (train_clips & val_clips)
    assert sorted(train + val) == sorted(keys)


def test_split_is_deterministic_for_a_seed(tmp_path):
    keys = list(np.load(make_pack(tmp_path)).keys())

    assert split_keys_by_clip(keys, 0.25, 2026) == split_keys_by_clip(keys, 0.25, 2026)
    assert split_keys_by_clip(keys, 0.25, 7) != split_keys_by_clip(keys, 0.25, 2026)


def test_token_pack_is_picklable_so_dataloader_workers_are_safe(tmp_path):
    pack = TokenPack(str(make_pack(tmp_path)))
    first = pack[0]

    revived = pickle.loads(pickle.dumps(pack))

    assert len(revived) == len(pack)
    assert np.array_equal(revived[0].numpy(), first.numpy())


def test_empty_pack_fails_loud(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez(path)
    with pytest.raises(RuntimeError, match="empty token pack"):
        TokenPack(str(path))


def test_collate_pads_and_reports_true_lengths(tmp_path):
    pack = TokenPack(str(make_pack(tmp_path)))
    batch = [pack[0], pack[1], pack[2]]

    padded, lengths = collate(batch)

    assert padded.shape[0] == 3
    assert padded.shape[1] == max(int(b.shape[0]) for b in batch)
    assert lengths.tolist() == [int(b.shape[0]) for b in batch]
    for row, length in enumerate(lengths.tolist()):
        assert padded[row, length:].sum() == 0
