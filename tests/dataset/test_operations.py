import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from text2motion.dataset.core.operations import (
    copy_sample,
    extract_tar_bz2,
    remove_tree,
    sha256_file,
)


def make_archive(path: Path) -> None:
    with tarfile.open(path, "w:bz2") as archive:
        for name, payload in (("KIT/4/a.npz", b"alpha"), ("KIT/4/b.npz", b"beta")):
            data = io.BytesIO(payload)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, data)


def test_hash_matches_the_standard_library(tmp_path: Path) -> None:
    target = tmp_path / "x.bin"
    target.write_bytes(b"motion")
    assert sha256_file(target) == hashlib.sha256(b"motion").hexdigest()


def test_extract_counts_the_files_it_wrote(tmp_path: Path) -> None:
    archive = tmp_path / "KIT.tar.bz2"
    make_archive(archive)
    dest = tmp_path / "work"
    assert extract_tar_bz2(archive, dest) == 2
    assert (dest / "KIT" / "4" / "a.npz").read_bytes() == b"alpha"


def test_remove_refuses_paths_outside_the_work_root(tmp_path: Path) -> None:
    work = tmp_path / "work"
    outside = tmp_path / "raw"
    outside.mkdir()
    (outside / "keep.npz").write_bytes(b"keep")
    with pytest.raises(PermissionError, match="outside"):
        remove_tree(outside, work)
    assert (outside / "keep.npz").is_file()


def test_remove_deletes_inside_the_work_root(tmp_path: Path) -> None:
    work = tmp_path / "work"
    target = work / "KIT_raw"
    target.mkdir(parents=True)
    (target / "a.npz").write_bytes(b"a")
    remove_tree(target, work)
    assert not target.exists()


def test_copy_sample_checks_the_size(tmp_path: Path) -> None:
    source = tmp_path / "a.npz"
    source.write_bytes(b"12345")
    target = tmp_path / "samples" / "a.npz"
    assert copy_sample(source, target) == 5
    assert target.read_bytes() == b"12345"
