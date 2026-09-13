from __future__ import annotations

from pathlib import Path

import numpy as np

from text2motion.motion.contracts import TextAnnotation


def parse_text_file(path: Path) -> list[TextAnnotation]:
    annotations: list[TextAnnotation] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split("#")
        if len(parts) < 4:
            raise ValueError(f"malformed caption line in {path}: {raw_line!r}")
        caption, tokens, start, end = parts[0], parts[1], parts[2], parts[3]
        start_time = float(start)
        end_time = float(end)
        annotations.append(
            TextAnnotation(
                caption=caption,
                tokens=tokens.split(" ") if tokens else [],
                start_time=0.0 if np.isnan(start_time) else start_time,
                end_time=0.0 if np.isnan(end_time) else end_time,
            )
        )
    if not annotations:
        raise ValueError(f"no captions found in {path}")
    return annotations
