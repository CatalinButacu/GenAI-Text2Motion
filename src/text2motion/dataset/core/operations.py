from __future__ import annotations

import hashlib
import shutil
import tarfile
from pathlib import Path

CHUNK = 1 << 20
NPY_FILES = "*.npy"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_tar_bz2(archive: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(archive, mode="r:bz2") as handle:
        for member in handle:
            if member.isfile():
                written += 1
        handle.extractall(dest, filter="data")
    return written


def remove_tree(target: Path, work_root: Path) -> None:
    resolved = target.resolve()
    if not resolved.is_relative_to(work_root.resolve()):
        raise PermissionError(f"{target} is outside the work root {work_root}; refusing to delete")
    if resolved == work_root.resolve():
        raise PermissionError("refusing to delete the work root itself")
    if resolved.is_dir():
        shutil.rmtree(resolved)
    elif resolved.exists():
        resolved.unlink()


def copy_sample(source: Path, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    copied = target.stat().st_size
    if copied != source.stat().st_size:
        raise OSError(f"copy of {source} is incomplete: {copied} of {source.stat().st_size} bytes")
    return copied


DATA_CLIP_ERRORS = (ValueError, OSError, RuntimeError)
