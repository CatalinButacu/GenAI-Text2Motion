"""Rename text_encoder.sbert.0.model.* -> .auto_model.* in saved checkpoints."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

OLD = "text_encoder.sbert.0.model."
NEW = "text_encoder.sbert.0.auto_model."

log = logging.getLogger("fix_sbert_keys")


def remapKeys(stateDict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    remapped = {}
    nChanged = 0

    for k, v in stateDict.items():

        if k.startswith(OLD):
            newK = NEW + k[len(OLD):]
            remapped[newK] = v
            nChanged += 1
        else:
            remapped[k] = v

    return remapped, nChanged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="Path to .pt to migrate")
    parser.add_argument("--inplace", action="store_true",
                        help="Overwrite the input file (otherwise writes _sbertfix.pt)")
    parser.add_argument("--out", type=Path, default=None, help="Explicit output path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.checkpoint.exists():
        log.error("Checkpoint not found: %s", args.checkpoint)
        return 1

    log.info("Loading %s ...", args.checkpoint)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if "model_state_dict" not in ck:
        log.error("Checkpoint missing 'model_state_dict' key. Got: %s", sorted(ck.keys()))
        return 1

    fixed, n = remapKeys(ck["model_state_dict"])
    log.info("Renamed %d SBERT keys (%s -> %s)", n, OLD, NEW)

    if n == 0:
        log.warning("No keys matched the OLD prefix. Maybe already migrated?")

    ck["model_state_dict"] = fixed

    if args.out:
        outPath = args.out
    elif args.inplace:
        outPath = args.checkpoint
    else:
        outPath = args.checkpoint.with_name(args.checkpoint.stem + "_sbertfix.pt")

    log.info("Saving -> %s", outPath)
    torch.save(ck, outPath)
    log.info("Done. Verify by running: python main.py 'a person walks forward'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
