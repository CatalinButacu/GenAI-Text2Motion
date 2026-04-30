from __future__ import annotations

import contextlib
import functools
import logging
import os
import time
import tracemalloc
from collections.abc import Generator

log = logging.getLogger(__name__)

TRUTHY = {"1", "true", "yes", "on"}  # env-var truthy set
PROFILE_MEM = os.environ.get("PROFILE_MEMORY", "0").strip().lower() in TRUTHY


@contextlib.contextmanager
def tracemallocSnapshot(label: str, topN: int = 10) -> Generator[None, None, None]:
    """CM logs elapsed time; also logs memory delta when PROFILE_MEMORY=1."""
    tStart = time.perf_counter()

    if not PROFILE_MEM:
        yield
        log.info("[perf] %s: %.3fs", label, time.perf_counter() - tStart)

        return

    already = tracemalloc.is_tracing()

    if not already:
        tracemalloc.start(25)

    snapBefore = tracemalloc.take_snapshot()
    memBefore = sum(s.size for s in snapBefore.statistics("lineno"))

    try:
        yield
    finally:
        elapsed = time.perf_counter() - tStart
        snapAfter = tracemalloc.take_snapshot()
        logDelta(label, snapBefore, snapAfter, memBefore, elapsed, topN)

        if not already:
            tracemalloc.stop()


def logDelta(label, snapBefore, snapAfter, memBefore, elapsed: float, topN: int) -> None:
    memAfter = sum(s.size for s in snapAfter.statistics("lineno"))
    diff = memAfter - memBefore
    sign = "+" if diff >= 0 else ""
    log.info(
        "[perf] %s: %.3fs  |  [mem] %s%d KB  (%.2f MB -> %.2f MB)",
        label,
        elapsed,
        sign,
        diff // 1024,
        memBefore / 1024 / 1024,
        memAfter / 1024 / 1024,
    )

    for rank, stat in enumerate(snapAfter.compare_to(snapBefore, "lineno")[:topN], 1):
        if stat.size_diff == 0:
            continue

        site = str(stat.traceback[0]) if stat.traceback else "<unknown>"
        log.debug(
            "[mem]  #%-2d  %+8.1f KB  count %+d  |  %s",
            rank,
            stat.size_diff / 1024,
            stat.count_diff,
            site,
        )


def profileMemory(fn):
    """Decorator: wraps a method with tracemalloc_snapshot when PROFILE_MEMORY=1."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with tracemallocSnapshot(fn.__qualname__):
            return fn(*args, **kwargs)

    return wrapper if PROFILE_MEM else fn
