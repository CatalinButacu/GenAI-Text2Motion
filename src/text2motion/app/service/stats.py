from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def process_memory_gb() -> tuple[float, float]:
    import psutil

    process = psutil.Process()
    memory = process.memory_info()
    peak = getattr(memory, "peak_wset", memory.rss)
    return memory.rss / 1e9, peak / 1e9


def record_inference(row: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        print(json.dumps(row), file=handle)


def format_stats(stats: dict) -> str:
    return (
        f"{stats['frames']} frames / {stats['steps']} steps in {stats['total_ms'] / 1e3:.2f}s "
        f"({stats['realtime_factor']:.1f}x realtime) | "
        f"encode {stats['text_encode_ms']:.0f}ms "
        f"generate {stats['generate_ms']:.0f}ms ({stats['ms_per_step']:.1f} ms/step) "
        f"detokenize {stats['detokenize_ms']:.0f}ms "
        f"send+client-wait {stats['send_ms']:.0f}ms | "
        f"peak gpu {stats['peak_gpu_gb']:.2f} GB  host rss {stats['host_rss_gb']:.2f} GB "
        f"(peak {stats['host_peak_gb']:.2f} GB)"
    )
