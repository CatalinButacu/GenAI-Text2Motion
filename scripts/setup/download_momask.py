"""One-shot setup for the MoMask text-to-motion generator.

Usage:
    python scripts/setup/download_momask.py

Steps:
    1. Clone EricGuo5513/momask-codes into vendor/momask (pinned to a commit).
    2. Install MoMask's Python dependencies into the current environment.
    3. Download the HumanML3D pretrained checkpoints (Google Drive, ~1 GB).
    4. Unpack checkpoints into vendor/momask/checkpoints/.

After this runs successfully, src.modules.motion.momask.MoMaskGenerator
can generate motion from text without further setup.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/EricGuo5513/momask-codes.git"
REPO_COMMIT = "main"
VENDOR_DIR = Path("vendor/momask")
CHECKPOINT_ZIP_ID = "1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0"
CHECKPOINT_ZIP_NAME = "humanml3d_models.zip"

MOMASK_DEPS = [
    "gdown",
    "einops",
    "vector-quantize-pytorch",
    "ftfy",
    "regex",
    "git+https://github.com/openai/CLIP.git",
]


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[setup] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def cloneRepo() -> None:
    if VENDOR_DIR.exists():
        print(f"[setup] vendor already present at {VENDOR_DIR}, skipping clone")
        return
    VENDOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", REPO_URL, str(VENDOR_DIR)])
    if REPO_COMMIT != "main":
        run(["git", "checkout", REPO_COMMIT], cwd=VENDOR_DIR)


def installDeps() -> None:
    run([sys.executable, "-m", "pip", "install", *MOMASK_DEPS])


def downloadCheckpoints() -> None:
    ckptRoot = VENDOR_DIR / "checkpoints"
    if (ckptRoot / "t2m").exists():
        print(f"[setup] checkpoints already present at {ckptRoot / 't2m'}, skipping download")
        return
    ckptRoot.mkdir(parents=True, exist_ok=True)
    zipPath = ckptRoot / CHECKPOINT_ZIP_NAME
    run(
        [
            sys.executable,
            "-m",
            "gdown",
            f"https://drive.google.com/uc?id={CHECKPOINT_ZIP_ID}",
            "-O",
            str(zipPath),
        ]
    )
    print(f"[setup] unzipping {zipPath} -> {ckptRoot}")
    with zipfile.ZipFile(zipPath) as z:
        z.extractall(ckptRoot)
    zipPath.unlink()


def verify() -> None:
    genScript = VENDOR_DIR / "gen_t2m.py"
    t2mDir = VENDOR_DIR / "checkpoints" / "t2m"
    if not genScript.exists():
        raise RuntimeError(f"[setup] gen_t2m.py missing at {genScript}")
    if not t2mDir.exists():
        raise RuntimeError(f"[setup] checkpoints missing at {t2mDir}")
    print(f"[setup] OK  gen_t2m.py -> {genScript}")
    print(f"[setup] OK  checkpoints -> {t2mDir}")


def main() -> None:
    cloneRepo()
    installDeps()
    downloadCheckpoints()
    verify()
    print("[setup] MoMask is ready. Run the pipeline with the default motion backend.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        print(f"[setup] command failed: {e}", file=sys.stderr)
        sys.exit(1)
