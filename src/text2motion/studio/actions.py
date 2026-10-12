from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_action(name: str, detail: str = "") -> None:
    logger.info(f"[UI] {name}{f' {detail}' if detail else ''}")


def log_toggle(name: str, value: bool) -> None:
    logger.info(f"[UI] {name} -> {'ON' if value else 'OFF'}")


def log_result(name: str, ok: bool, detail: str = "") -> None:
    logger.info(f"[UI] {name} -> {'OK' if ok else 'FAILED'}{f' ({detail})' if detail else ''}")
