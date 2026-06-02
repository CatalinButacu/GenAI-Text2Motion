"""Pre-build the SSM trainer's unified buffer cache without launching training.

Used to populate data/.cache/unified_buf_<hash>.joblib so cloud (which has no
raw AMASS) can train via cache reuse only.

Usage:
    python scripts/data/prebuild_unified_cache.py --sources humanml3d
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.architecture.training.trainer import build_unified_buf
from src.data.unified_dataset import SourceConfig, UnifiedConfig

log = logging.getLogger(__name__)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+",
                        choices=["amass", "arctic", "humanml3d", "interx"],
                        default=["humanml3d"])
    parser.add_argument("--amass-dir", default="data/AMASS", dest="amass_dir")
    parser.add_argument("--arctic-dir", default="data/arctic/unpack", dest="arctic_dir")
    parser.add_argument("--humanml3d-dir", default="data/humanml3d", dest="humanml3d_dir")
    parser.add_argument("--interx-dir", default="data/inter-x", dest="interx_dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    cfg = UnifiedConfig(
        amass=SourceConfig(enabled="amass" in args.sources, data_dir=args.amass_dir),
        arctic=SourceConfig(enabled="arctic" in args.sources, data_dir=args.arctic_dir),
        humanml3d=SourceConfig(enabled="humanml3d" in args.sources,
                                data_dir=args.humanml3d_dir, amass_dir=args.amass_dir),
        interx=SourceConfig(enabled="interx" in args.sources, data_dir=args.interx_dir),
    )

    log.info("[prebuild] sources=%s", args.sources)
    buf = build_unified_buf(cfg)
    log.info("[prebuild] DONE: %d samples", len(buf))

    return 0

if __name__ == "__main__":
    sys.exit(main())
