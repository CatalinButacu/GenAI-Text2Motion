#!/usr/bin/env python
# run: .venv\Scripts\python.exe scripts\data\fetch_sources.py list
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
CHUNK = 1 << 20

SOURCES = MappingProxyType(
    {
        "motionhub": MappingProxyType(
            {
                "access": "hf",
                "repo": "ZeyuLing/MotionHub",
                "home": "https://hf.co/datasets/ZeyuLing/MotionHub",
                "licence": "other: read the dataset card before using it",
            }
        ),
        "amass": MappingProxyType(
            {
                "access": "login",
                "home": "https://amass.is.tue.mpg.de",
                "licence": "AMASS licence (MPI); read the site",
            }
        ),
        "smplx": MappingProxyType(
            {
                "access": "login",
                "home": "https://smpl-x.is.tue.mpg.de",
                "licence": "SMPL-X licence (MPI); read the site",
            }
        ),
        "grab": MappingProxyType(
            {
                "access": "login",
                "home": "https://grab.is.tue.mpg.de",
                "licence": "GRAB licence (MPI); read the site",
            }
        ),
        "babel": MappingProxyType(
            {
                "access": "login",
                "home": "https://babel.is.tue.mpg.de",
                "licence": "BABEL licence (MPI); read the site",
            }
        ),
    }
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def source(name):
    if name not in SOURCES:
        sys.exit(f"unknown source '{name}'. Run: list")
    return SOURCES[name]


def dest(name):
    return RAW / name


def manifest_entries(folder):
    files = sorted(p for p in folder.rglob("*") if p.is_file() and p.name != "MANIFEST.json")
    return [
        {"path": p.relative_to(folder).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)}
        for p in files
    ]


def write_manifest(name, licence):
    folder = dest(name)
    entries = manifest_entries(folder)
    data = {
        "source": name,
        "licence": licence,
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "files": entries,
        "total_bytes": sum(e["size"] for e in entries),
    }
    (folder / "MANIFEST.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(
        f"{name}: {len(entries)} files, {data['total_bytes'] / 2**30:.2f} GB -> {folder / 'MANIFEST.json'}"
    )


def cmd_list(_args):
    print(f"{'source':<11}{'access':<8}{'where':<46}licence")
    for name, meta in SOURCES.items():
        print(f"{name:<11}{meta['access']:<8}{meta['home']:<46}{meta['licence']}")


def cmd_fetch(args):
    meta = source(args.name)
    if meta["access"] != "hf":
        sys.exit(
            f"{args.name} needs a login on its site. Download it there, then run: place {args.name} <folder>"
        )
    if not args.accept_licence:
        sys.exit("Read the licence first, then re-run with --accept-licence")
    from huggingface_hub import snapshot_download

    folder = dest(args.name)
    folder.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=meta["repo"],
        repo_type="dataset",
        local_dir=str(folder),
        allow_patterns=args.allow or None,
    )
    write_manifest(args.name, meta["licence"])


def cmd_place(args):
    source(args.name)
    src = Path(args.folder)
    if not src.is_dir():
        sys.exit(f"not a folder: {src}")
    folder = dest(args.name)
    folder.mkdir(parents=True, exist_ok=True)
    import shutil

    for item in src.iterdir():
        target = folder / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
    write_manifest(args.name, source(args.name)["licence"])


def cmd_verify(args):
    source(args.name)
    manifest_path = dest(args.name) / "MANIFEST.json"
    if not manifest_path.exists():
        sys.exit("no MANIFEST.json: run fetch or place first")
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad = []
    for e in recorded["files"]:
        p = dest(args.name) / e["path"]
        if not p.exists() or sha256(p) != e["sha256"]:
            bad.append(e["path"])
    print(f"{args.name}: {len(recorded['files']) - len(bad)} ok, {len(bad)} bad")
    for b in bad:
        print("  BAD", b)
    if bad:
        sys.exit(1)


def build_parser():
    parser = argparse.ArgumentParser(prog="fetch_sources")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    f = sub.add_parser("fetch")
    f.add_argument("name")
    f.add_argument("--accept-licence", action="store_true")
    f.add_argument("--allow", nargs="*", help="file patterns, e.g. --allow '*.parquet'")
    p = sub.add_parser("place")
    p.add_argument("name")
    p.add_argument("folder")
    v = sub.add_parser("verify")
    v.add_argument("name")
    return parser


def main():
    args = build_parser().parse_args()
    handlers = {"list": cmd_list, "fetch": cmd_fetch, "place": cmd_place, "verify": cmd_verify}
    handlers[args.command](args)


if __name__ == "__main__":
    main()
