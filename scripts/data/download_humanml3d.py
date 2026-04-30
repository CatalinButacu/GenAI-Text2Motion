#!/usr/bin/env python
"""Download HumanML3D text annotations and clip-ID splits from HuggingFace.

Source: https://huggingface.co/datasets/lxxiao/272-dim-HumanML3D
  - texts.zip   12.2 MB  -- natural-language motion descriptions (~14 k clips)
  - split/*.txt  216 kB  -- train/val/test clip-ID lists

These files are used solely as a text-annotation layer on top of AMASS.
Motion training loads native SMPL-X clips from AMASS using index.csv;
the 272-dim feature tensors from the HuggingFace repo are NOT downloaded
or used anywhere in the pipeline.

Usage:
    python scripts/data/download_humanml3d.py
    python scripts/data/download_humanml3d.py --dest data/humanml3d
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download as hfHubDownload
    HF_AVAILABLE = True
except ImportError:
    hfHubDownload = None
    HF_AVAILABLE = False

log = logging.getLogger(__name__)

HF_REPO_ID  = "lxxiao/272-dim-HumanML3D"
REPO_TYPE   = "dataset"

# Files downloaded
SMALL_FILES = [
    "texts.zip",
    "split/train.txt",
    "split/val.txt",
    "split/test.txt",
]


def hfDownload(repoId: str, filename: str, destDir: Path, token: str | None) -> Path | None:
    """Download a single file from a HuggingFace dataset repo."""
    if not HF_AVAILABLE or hfHubDownload is None:
        log.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return None

    localDir = destDir / "_hf_cache"
    localDir.mkdir(parents=True, exist_ok=True)

    try:
        log.info("Downloading %s ...", filename)
        path = hfHubDownload(
            repo_id=repoId,
            filename=filename,
            repo_type=REPO_TYPE,
            local_dir=str(localDir),
            token=token,
        )
        return Path(path)
    except Exception as e:
        log.error("Failed to download %s: %s", filename, e)
        return None


def extractZip(zipPath: Path, destDir: Path, label: str) -> bool:
    """Extract a zip archive to destDir, returning True on success."""
    log.info("Extracting %s ...", label)
    try:
        with zipfile.ZipFile(zipPath, "r") as zf:
            zf.extractall(str(destDir))
        log.info("Extracted %s -> %s", label, destDir)
        return True
    except zipfile.BadZipFile as e:
        log.error("Bad zip file %s: %s", zipPath, e)
        return False


def moveFile(src: Path, dest: Path) -> None:
    """Copy src to dest (keeps cache copy)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dest))


def integrityCheck(dest: Path) -> dict:
    """Check downloaded artifacts and return a status dict."""
    status = {}

    textsDir = dest / "texts"
    if textsDir.exists():
        n = len(list(textsDir.glob("*.txt")))
        status["texts"] = n
        if n < 1000:
            log.warning("texts/ has only %d files -- expected ~14,000", n)
        else:
            log.info("texts/: %d annotation files OK", n)
    else:
        status["texts"] = 0
        log.warning("texts/ directory not found -- texts.zip may not have extracted correctly")

    for fname in ["split/train.txt", "split/val.txt", "split/test.txt"]:
        fpath = dest / fname
        if fpath.exists():
            n = sum(1 for _ in fpath.read_text().splitlines() if _.strip())
            status[fname] = n
            log.info("%s: %d clip IDs", fname, n)
        else:
            status[fname] = 0

    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download HumanML3D annotations from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dest", default="data/humanml3d",
                        help="Target directory (default: data/humanml3d)")
    parser.add_argument("--token", default=None,

                        help="HuggingFace token (not required for public datasets)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    log.info("Target: %s", dest.resolve())

    filesToDownload = list(SMALL_FILES)

    #  Download
    downloaded: dict[str, Path] = {}
    for filename in filesToDownload:
        path = hfDownload(HF_REPO_ID, filename, dest, args.token)
        if path:
            downloaded[filename] = path

    if not downloaded:
        log.error("Nothing was downloaded. Check your internet connection.")
        return 1

    #  Extract zips

    # texts.zip -> dest/texts/
    if "texts.zip" in downloaded:
        # The zip may have the texts directly or inside a "texts/" subfolder
        textsDest = dest
        extractZip(downloaded["texts.zip"], textsDest, "texts.zip")
        # Normalise: some zips extract to texts/, others to HumanML3D/texts/
        hml3dTexts = dest / "HumanML3D" / "texts"
        if hml3dTexts.exists() and not (dest / "texts").exists():
            shutil.move(str(hml3dTexts), str(dest / "texts"))
            log.info("Moved HumanML3D/texts -> texts/")

    #  Copy flat files (splits)
    for filename in ["split/train.txt", "split/val.txt", "split/test.txt"]:
        if filename in downloaded:
            moveFile(downloaded[filename], dest / filename)

    #  Integrity check
    status = integrityCheck(dest)

    log.info("" * 60)
    log.info("Download complete -> %s", dest.resolve())
    log.info("  texts/          : %d annotation files", status.get("texts", 0))
    log.info("  split/train.txt : %d clips", status.get("split/train.txt", 0))
    log.info("  split/val.txt   : %d clips", status.get("split/val.txt", 0))

    if status.get("texts", 0) > 0:
        log.info("")
        log.info("Next steps:")
        log.info("  Text annotations will enrich AMASS training automatically.")
        log.info("  python scripts/training/train_motion_ssm.py --data-dir data/AMASS")
        log.info("  For HumanML3D-split training: python scripts/training/train_motion_ssm.py "
                 "--data-source humanml3d")
    return 0


if __name__ == "__main__":
    sys.exit(main())
