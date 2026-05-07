"""Pre-build the SSM trainer's unified buffer cache without launching training.

Used to populate data/.cache/unified_buf_<hash>.joblib so cloud (which has no
raw AMASS) can train via cache reuse only.

Usage:
    python scripts/data/prebuild_unified_cache.py --sources humanml3d
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.unified_dataset import SourceConfig, UnifiedConfig
from src.modules.motion.training.trainer import buildUnifiedBuf

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+",
                        choices=["amass", "arctic", "humanml3d", "interx"],
                        default=["humanml3d"])
    parser.add_argument("--amass-dir", default="data/AMASS", dest="amassDir")
    parser.add_argument("--arctic-dir", default="data/arctic/unpack", dest="arcticDir")
    parser.add_argument("--humanml3d-dir", default="data/humanml3d", dest="humanml3dDir")
    parser.add_argument("--interx-dir", default="data/inter-x", dest="interxDir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    cfg = UnifiedConfig(
        amass=SourceConfig(enabled="amass" in args.sources, dataDir=args.amassDir),
        arctic=SourceConfig(enabled="arctic" in args.sources, dataDir=args.arcticDir),
        humanml3d=SourceConfig(enabled="humanml3d" in args.sources,
                                dataDir=args.humanml3dDir, amassDir=args.amassDir),
        interx=SourceConfig(enabled="interx" in args.sources, dataDir=args.interxDir),
    )

    log.info("[prebuild] sources=%s", args.sources)
    buf = buildUnifiedBuf(cfg)
    log.info("[prebuild] DONE: %d samples", len(buf))

    return 0


if __name__ == "__main__":
    sys.exit(main())
