from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


############################################################
#       LiveGenerationDebouncer
############################################################
class LiveGenerationDebouncer:
    def __init__(self) -> None:
        self.enabled = False
        self.prev_prompt = ""
        self.dirty_at = 0.0
        self.last_dispatched = ""
        self.regen_pending = False
        self.signature: tuple | None = None

    def mark_dirty(self, text: str, now: float) -> None:
        self.prev_prompt = text
        self.dirty_at = now

    def debounce_elapsed(self, now: float, debounce_s: float) -> bool:
        return self.prev_prompt != "" and (now - self.dirty_at) >= debounce_s
