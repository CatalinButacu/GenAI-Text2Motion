from __future__ import annotations

import time

from text2motion.studio.contracts import ViewerHost


class PlaybackPacer:
    def __init__(self, host: ViewerHost, state) -> None:
        self.host = host
        self.state = state

    def frames_ahead_of_playhead(self, accepted_frames: int) -> int:
        buffers = self.state.stream_buffers
        played = buffers.trimmed_frames + self.host.scene.current_frame_id
        return accepted_frames - played

    def wait_for_playback(self, accepted_frames: int) -> None:
        while (
            not self.state.stream_buffers.cancel
            and self.frames_ahead_of_playhead(accepted_frames)
            > self.state.stream_lead_frames * self.state.upsample
        ):
            time.sleep(0.01)
