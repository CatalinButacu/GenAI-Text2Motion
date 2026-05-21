"""Rename stale state_dict keys in saved PyTorch checkpoints after the snake->camel pass.

Existing checkpoints under checkpoints/ were saved before the renames. The saved
state_dict still has keys like `text_encoder._sbert.*` while the live model now
expects `text_encoder.sbert.*`. This script:

  1. Loads each .pt checkpoint
  2. Walks the model_state_dict keys
  3. Renames any key segment matching a known old->new mapping
  4. Writes back to a `.migrated.pt` file (does not overwrite original)

Usage
-----
  python scripts/refactor/migrate_checkpoint_keys.py            # dry-run, list rewrites
  python scripts/refactor/migrate_checkpoint_keys.py --apply    # write migrated copies
  python scripts/refactor/migrate_checkpoint_keys.py --apply --in-place  # overwrite

Safe by default: nothing is written unless --apply is passed. With --in-place, a
.bak file is created next to each modified checkpoint first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

CKPT_DIR = ROOT / "checkpoints"

# Old underscore-prefixed name -> new public name (Rule 1 fixes).
# Keep this list in sync with the project naming rules.
KEY_PREFIX_MAP = {
    "_sbert": "sbert",
    "_GENERATOR": "generator",
}


def rename_key(key: str) -> str:
    parts = key.split(".")
    out = []
    for p in parts:
        out.append(KEY_PREFIX_MAP.get(p, p))

    return ".".join(out)


def migrate_one(ckpt: Path, apply: bool, inPlace: bool) -> int:
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        print(f"  SKIP  {ckpt.name}: not a dict checkpoint")

        return 0

    sd_key = next((k for k in ("model_state_dict", "state_dict", "model") if k in obj), None)
    if sd_key is None:
        print(f"  SKIP  {ckpt.name}: no state_dict-like key")

        return 0

    sd = obj[sd_key]
    rewrites = {}
    for k in list(sd.keys()):
        new = rename_key(k)
        if new != k:
            rewrites[k] = new

    if not rewrites:
        print(f"  OK    {ckpt.name}: no stale keys")

        return 0

    print(f"  {ckpt.name}: {len(rewrites)} keys need renaming")
    sample = list(rewrites.items())[:3]
    for o, n in sample:
        print(f"      {o}  ->  {n}")

    if not apply:
        return len(rewrites)

    new_sd = {rewrites.get(k, k): v for k, v in sd.items()}
    obj[sd_key] = new_sd

    if inPlace:
        bak = ckpt.with_suffix(ckpt.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(ckpt, bak)
        target = ckpt
    else:
        target = ckpt.with_name(ckpt.stem + ".migrated.pt")

    torch.save(obj, target)
    print(f"  WROTE {target.name}")

    return len(rewrites)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--in-place", action="store_true", dest="inPlace")
    args = ap.parse_args()

    if not CKPT_DIR.exists():
        print(f"No checkpoints/ at {CKPT_DIR}")

        return 0

    total = 0
    for ckpt in sorted(CKPT_DIR.rglob("*.pt")):
        if ckpt.name.endswith(".bak.pt") or ckpt.name.endswith(".migrated.pt"):
            continue
        total += migrate_one(ckpt, args.apply, args.inPlace)

    if args.apply:
        print(f"\nMigrated {total} keys total")
    else:
        print(f"\n{total} keys would be renamed. Pass --apply to write.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
