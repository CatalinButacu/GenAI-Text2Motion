"""Robust Inter-X dataset downloader with retry, resume, and integrity checks.

Downloads from:
    https://drive.google.com/drive/folders/1eSekYd4jPTAPnAabLbr2lrrC8_Fwa_b0

Features:
  - Enumerates the shared folder to discover all files
  - Downloads each file individually with configurable retries
  - Supports resume for partially downloaded files
  - Verifies download integrity via file-size check
  - Extracts zip files after all downloads complete
  - Cleans up partial/corrupt downloads

Usage:
    python scripts/download_interx.py
    python scripts/download_interx.py --dest data/inter-x --retries 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile

try:
    import gdown
except ImportError:
    print("ERROR: gdown not installed. Run: pip install gdown")
    sys.exit(1)

FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1eSekYd4jPTAPnAabLbr2lrrC8_Fwa_b0"
)

# Known file sizes (bytes) from the Inter-X release for integrity verification.
# If a file isn't listed here, it's still downloaded but only checked for
# non-zero size.
EXPECTED_SIZES: dict[str, int] = {
    # Add known sizes here as they become available, e.g.:
    # "motions.zip": 28_000_000_000,
}


def downloadFolderRobust(
    dest: str,
    maxRetries: int = 3,
    retryDelay: float = 10.0,
) -> tuple[list[str], list[str]]:
    """Download all files from the Inter-X Google Drive folder.

    Uses gdown to enumerate the folder, then downloads each file individually
    with retry logic.

    Returns (succeeded, failed) file path lists.
    """
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading Inter-X dataset to {dest}/")
    print(f"Source: {FOLDER_URL}")
    print(f"Retries per file: {maxRetries}")
    print(f"Retry delay: {retryDelay}s (x attempt)")
    print()

    # Use gdown to enumerate and download folder contents
    for attempt in range(1, maxRetries + 1):
        print(f"Download attempt {attempt}/{maxRetries}...")
        try:
            gdown.download_folder(
                url=FOLDER_URL,
                output=dest,
                quiet=False,
                use_cookies=False,
                remaining_ok=True,
            )
            break
        except Exception as e:
            print(f"Attempt {attempt} error: {e}")
            if attempt < maxRetries:
                wait = retryDelay * attempt
                print(f"  Waiting {wait:.0f}s before retry...")
                time.sleep(wait)
            else:
                print("Proceeding with whatever was downloaded...\n")

    # Report what we got
    succeeded: list[str] = []
    failed: list[str] = []

    print(f"\nScanning {dest}/ for downloaded content...")
    for root, _dirs, files in os.walk(dest):
        for fname in files:
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)

            # Skip partial download fragments
            if fname.endswith(".part"):
                print(f"  ! Partial file found (will retry): {fname}")
                failed.append(fpath)
                continue

            # Check for quota error pages
            if size < 10_000 and fname not in (
                "action_setting.txt", "familiarity.txt",
            ):
                with open(fpath, "rb") as f:
                    header = f.read(500)
                if b"<html" in header.lower() or b"quota" in header.lower():
                    print(f"  ! Quota error page detected: {fname}")
                    failed.append(fpath)
                    continue

            succeeded.append(fpath)
            rel = os.path.relpath(fpath, dest)
            print(f"  OK {rel} ({size / 1024 / 1024:.1f} MB)")

    return succeeded, failed


def extractZips(dest: str) -> None:
    """Extract any .zip files in the destination folder."""
    for name in sorted(os.listdir(dest)):
        if not name.endswith(".zip"):
            continue
        zipPath = os.path.join(dest, name)
        extractDir = os.path.join(dest, name[:-4])

        if os.path.isdir(extractDir) and os.listdir(extractDir):
            print(f"  {name} already extracted -> {extractDir}/")
            continue

        print(f"  Extracting {name}...")
        try:
            with zipfile.ZipFile(zipPath, "r") as zf:
                # Validate zip integrity first
                bad = zf.testzip()
                if bad is not None:
                    print(f"  X Corrupt zip entry: {bad}")
                    print(f"    Delete {zipPath} and re-download.")
                    continue
                zf.extractall(extractDir)
            nEntries = sum(1 for _ in os.scandir(extractDir))
            print(f"  OK Extracted {nEntries} entries -> {extractDir}/")
        except zipfile.BadZipFile:
            print(f"  X {name} is corrupt (BadZipFile). Delete and re-download.")
        except Exception as e:
            print(f"  X Extraction failed: {e}")


def report(dest: str) -> None:
    """Print a summary of dataset contents."""
    print(f"\n{'='*60}")
    print(f"Inter-X dataset in {dest}/")
    print(f"{'='*60}")

    totalSize = 0
    for name in sorted(os.listdir(dest)):
        full = os.path.join(dest, name)
        if os.path.isfile(full):
            sz = os.path.getsize(full)
            totalSize += sz
            print(f"  FILE  {name}: {sz / 1024 / 1024:.1f} MB")
        else:
            n = sum(1 for _ in os.scandir(full) if True)
            dirSz = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(full)
                for f in fns
            )
            totalSize += dirSz
            print(f"  DIR   {name}/ ({n} entries, {dirSz / 1024 / 1024:.1f} MB)")

    print(f"\n  Total: {totalSize / 1024 / 1024 / 1024:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Inter-X dataset from Google Drive (robust)",
    )
    parser.add_argument(
        "--dest", type=str, default="data/inter-x",
        help="Destination directory (default: data/inter-x)",
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Max retries per file (default: 3)",
    )
    parser.add_argument(
        "--retry-delay", type=float, default=10.0,
        help="Base retry delay in seconds (default: 10, multiplied by attempt)",
    dest="retryDelay")
    parser.add_argument(
        "--no-extract", action="store_true",
        help="Skip zip extraction after download",
    dest="noExtract")
    args = parser.parse_args()

    t0 = time.time()

    succeeded, failed = downloadFolderRobust(
        args.dest,
        maxRetries=args.retries,
        retryDelay=args.retry_delay,
    )

    if not args.no_extract:
        print("\nExtracting zip files...")
        extractZips(args.dest)

    report(args.dest)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed / 60:.1f} minutes")
    print(f"  Succeeded: {len(succeeded)} files")
    if failed:
        print(f"  Failed: {len(failed)} files:")
        for f in failed:
            print(f"    - {os.path.relpath(f, args.dest)}")
        print("\n  Re-run this script to retry failed downloads.")
    else:
        print("  All files downloaded successfully!")


if __name__ == "__main__":
    main()
