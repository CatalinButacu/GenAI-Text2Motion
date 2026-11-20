#!/usr/bin/env python
# run: .venv\Scripts\python.exe scripts\data\extract_inbox.py
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INBOX = ROOT.parent / "inbox" / "amass"
RAW = ROOT / "data" / "raw"
LOG = RAW / "EXTRACT_LOG.json"
SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
SPACE_FACTOR = 4.0
SPACE_MARGIN = 5 * 2**30
SKIP_WORDS = ("render", "mirror", "crdownload")


def gib(n):
    return f"{n / 2**30:.1f} GB"


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def is_data_archive(path):
    name = path.name.lower()
    if any(word in name for word in SKIP_WORDS):
        return False
    return name.endswith(".tar.bz2") or name.endswith(".tar")


def destination(path):
    stem = path.name.split(".")[0]
    if stem == "GRAB":
        return RAW / "grab"
    return RAW / "amass" / stem


def queue():
    return [p for p in sorted(INBOX.iterdir()) if p.is_file() and is_data_archive(p)]


def load_log():
    if LOG.exists():
        return json.loads(LOG.read_text(encoding="utf-8"))
    return {}


def save_log(log):
    RAW.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(log, indent=2), encoding="utf-8")


def free_bytes():
    return shutil.disk_usage(ROOT).free


def run_seven_zip(archive, dest):
    proc = subprocess.run(
        [str(SEVEN_ZIP), "x", str(archive), f"-o{dest}", "-y"], capture_output=True, text=True
    )
    return proc.returncode, proc.stdout[-400:] + proc.stderr[-400:]


def extract(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    code, tail = run_seven_zip(archive, dest)
    if code != 0:
        return code, tail
    for inner in sorted(dest.glob("*.tar")):
        code, tail = run_seven_zip(inner, dest)
        if code != 0:
            return code, tail
        inner.unlink()
    return 0, ""


def count_files(folder):
    return sum(1 for p in folder.rglob("*") if p.is_file())


def main():
    log = load_log()
    for archive in queue():
        entry = log.get(archive.name, {})
        if entry.get("status") == "ok":
            print(f"skip (done): {archive.name}", flush=True)
            continue
        dest = destination(archive)
        need = archive.stat().st_size * SPACE_FACTOR + SPACE_MARGIN
        free = free_bytes()
        if free < need:
            log[archive.name] = {
                "status": "waiting_space",
                "need": gib(need),
                "free": gib(free),
                "checked": utc_now(),
            }
            save_log(log)
            print(
                f"STOP before {archive.name}: needs {gib(need)} free, have {gib(free)}. Remaining archives kept.",
                flush=True,
            )
            return 2
        print(f"extracting {archive.name} ({gib(archive.stat().st_size)}) -> {dest}", flush=True)
        code, tail = extract(archive, dest)
        if code != 0:
            log[archive.name] = {
                "status": "error",
                "exit_code": code,
                "tail": tail,
                "checked": utc_now(),
            }
            save_log(log)
            print(f"ERROR {archive.name}: 7-Zip exit {code}. Archive kept.", flush=True)
            return 1
        files = count_files(dest)
        if files == 0:
            log[archive.name] = {
                "status": "error",
                "exit_code": 0,
                "tail": "no files extracted",
                "checked": utc_now(),
            }
            save_log(log)
            print(f"ERROR {archive.name}: nothing extracted. Archive kept.", flush=True)
            return 1
        archive.unlink()
        log[archive.name] = {
            "status": "ok",
            "dest": str(dest),
            "files": files,
            "checked": utc_now(),
        }
        save_log(log)
        print(f"done {archive.name}: {files} files, original deleted", flush=True)
    print("all archives processed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
