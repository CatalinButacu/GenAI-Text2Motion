from __future__ import annotations

import threading
import traceback

from text2motion.studio.live.actions import log_result
from text2motion.studio.session.registry import StudioModelEntry


class ModelLoader:
    def __init__(self, state) -> None:
        self.state = state

    def start_model_load(self) -> None:
        entry = self.state.models[self.state.model_idx]
        self.state.loading_model = True
        self.state.load_status = f"loading {entry.label}..."
        threading.Thread(target=self.load_selected_model, args=(entry,), daemon=True).start()

    def load_selected_model(self, entry: StudioModelEntry) -> None:
        try:
            hello = self.state.client.load(
                entry.config, entry.ckpt, entry.tokenizer_ckpt, entry.backbone
            )
            self.state.backbone = entry.backbone
            self.state.apply_model_limits(hello)
            if self.state.max_steps:
                capped = min(self.state.steps or self.state.max_steps, self.state.max_steps)
                if capped != self.state.steps:
                    self.state.status = (
                        f"{entry.label} cannot run indefinitely -- clamped to {capped} steps"
                    )
                self.state.steps = capped
            self.state.load_status = f"active: {entry.label}"
            log_result("load model", True, entry.key)
        except Exception as exc:
            traceback.print_exc()
            log_result("load model", False, f"{type(exc).__name__}: {exc}")
            self.state.load_status = f"load failed: {type(exc).__name__}: {exc}"
        finally:
            self.state.loading_model = False
