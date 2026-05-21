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
def tracemalloc_snapshot(label: str, top_n: int = 10) -> Generator[None, None, None]:
    """CM logs elapsed time; also logs memory delta when PROFILE_MEMORY=1."""
    t_start = time.perf_counter()

    if not PROFILE_MEM:
        yield
        log.info("[perf] %s: %.3fs", label, time.perf_counter() - t_start)

        return

    already = tracemalloc.is_tracing()

    if not already:
        tracemalloc.start(25)

    snap_before = tracemalloc.take_snapshot()
    mem_before = sum(s.size for s in snap_before.statistics("lineno"))

    try:
        yield
    finally:
        elapsed = time.perf_counter() - t_start
        snap_after = tracemalloc.take_snapshot()
        log_delta(label, snap_before, snap_after, mem_before, elapsed, top_n)

        if not already:
            tracemalloc.stop()


def log_delta(label, snap_before, snap_after, mem_before, elapsed: float, top_n: int) -> None:
    mem_after = sum(s.size for s in snap_after.statistics("lineno"))
    diff = mem_after - mem_before
    sign = "+" if diff >= 0 else ""
    log.info(
        "[perf] %s: %.3fs  |  [mem] %s%d KB  (%.2f MB -> %.2f MB)",
        label,
        elapsed,
        sign,
        diff // 1024,
        mem_before / 1024 / 1024,
        mem_after / 1024 / 1024,
    )

    for rank, stat in enumerate(snap_after.compare_to(snap_before, "lineno")[:top_n], 1):
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


def profile_memory(fn):
    """Decorator: wraps a method with tracemalloc_snapshot when PROFILE_MEMORY=1."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with tracemalloc_snapshot(fn.__qualname__):
            return fn(*args, **kwargs)

    return wrapper if PROFILE_MEM else fn
