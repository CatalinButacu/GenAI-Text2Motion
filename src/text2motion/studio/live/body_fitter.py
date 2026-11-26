from __future__ import annotations

import queue
import threading
import traceback

from text2motion.studio.live.actions import discard_pending_items, log_result


class BodyFitter:
    def __init__(self, state, avatar) -> None:
        self.state = state
        self.avatar = avatar

    def start_fit_thread(
        self, fit_body: bool
    ) -> tuple[queue.Queue | None, threading.Thread | None]:
        if not fit_body:
            return None, None
        fit_queue = queue.Queue(maxsize=self.state.fit_lookahead_chunks)
        take = self.state.stream_buffers.take
        fit_thread = threading.Thread(
            target=self.fit_motion_chunks, args=(fit_queue, take), daemon=True
        )
        fit_thread.start()
        return fit_queue, fit_thread

    def fit_motion_chunks(self, fit_queue: queue.Queue, take: int) -> None:
        buffers = self.state.stream_buffers
        buffers.fit_take = take
        buffers.fitting = True
        n_frames = 0
        try:
            while True:
                chunk = fit_queue.get()
                if chunk is None:
                    break
                if buffers.fit_is_stale():
                    continue
                n_frames += len(chunk.joints) * self.state.upsample
                self.avatar.fit_single_chunk(chunk.joints, n_frames, self.state.upsample)
        except Exception as exc:
            traceback.print_exc()
            log_result("body fit", False, f"{type(exc).__name__}: {exc}")
            self.state.status = f"body fit failed: {type(exc).__name__}: {exc}"
        else:
            log_result("body fit", True, f"{n_frames} frames")
            if buffers.fit_take == take and not self.state.generating and not buffers.cancel:
                self.state.status = f"done -- body fitted on {n_frames} frames"
        finally:
            if buffers.fit_take == take:
                buffers.fitting = False
                buffers.fit_progress = None

    def feed_fitter(self, fit_queue: queue.Queue, fit_thread: threading.Thread, chunk) -> None:
        while not self.state.stream_buffers.cancel and fit_thread.is_alive():
            try:
                fit_queue.put(chunk, timeout=0.2)
                return
            except queue.Full:
                continue

    def close_fitter(self, fit_queue: queue.Queue, fit_thread: threading.Thread) -> None:
        while fit_thread.is_alive():
            try:
                fit_queue.put(None, timeout=0.2)
                return
            except queue.Full:
                continue

    def stop_fitter(
        self,
        fit_queue: queue.Queue | None,
        fit_thread: threading.Thread | None,
        discard: bool,
    ) -> None:
        if fit_queue is None or fit_thread is None:
            return
        if discard:
            discard_pending_items(fit_queue)
        self.close_fitter(fit_queue, fit_thread)
