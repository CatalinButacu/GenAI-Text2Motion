from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from src.shared.config import MotionConfig
from src.shared.constants import CONSTS
from src.shared.vocab import ACTIONS

from .clip_ops import (
    build_action_query,
    compute_action_frames,
    generate_action_clips,
    sequence_clips,
    slerp_blend_frames,
)
from .models import MotionClip
from .augment import MotionReranker, MotionRetriever
from .ssm_model import SSMMotionModel


class MotionGenerator:
    def __init__(self, config: MotionConfig | None = None) -> None:
        self.cfg = config or MotionConfig()
        self.backend = SSMMotionModel(
            checkpoint_path=self.cfg.checkpoint_path,
            rvq_checkpoint_path=self.cfg.rvq_checkpoint_path,
        )
        self.reranker = MotionReranker() if self.cfg.rerank else None
        self.retriever = (
            MotionRetriever(
                index_path=self.cfg.retrieval_index_path,
                top_k=self.cfg.retrieval_top_k,
            )
            if self.cfg.use_retrieval
            else None
        )

    def sample_one(self, text: str, num_frames: int) -> MotionClip:
        return self.backend.generate_from_text_tokens(
            text,
            num_frames,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            cfg_scale=self.cfg.cfg_scale,
        )

    def generate(
        self,
        text: str,
        num_frames: int,
        init_pose: np.ndarray | None = None,
    ) -> MotionClip:
        query_text = self.retriever.augment_prompt(text) if self.retriever is not None else text

        if self.cfg.rerank and self.reranker is not None:
            candidates = [
                self.sample_one(query_text, num_frames)
                for _ in range(max(1, self.cfg.num_candidates))
            ]
            clip = self.reranker.rank(text, candidates)
        else:
            clip = self.sample_one(query_text, num_frames)

        return blend_init_pose(clip, init_pose, self.cfg.init_pose_blend_frames)

    def generate_for_scene(self, planned) -> dict[str, MotionClip]:
        duration = getattr(planned, "duration", CONSTS.runtime.default_clip_duration_s)
        total_frames = int(duration * CONSTS.runtime.motion_fps)
        action_clips = generate_action_clips(planned, total_frames, self)

        if not action_clips:
            raise RuntimeError("no motion clips generated -- prompt has no recognised actions")

        return sequence_clips(action_clips, self.cfg.blend_frames)

    def stream(self, text: str, num_frames: int) -> Iterator[np.ndarray]:
        return self.backend.stream_actions(
            [(text, num_frames)],
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )

    def stream_for_scene(self, planned) -> Iterator[np.ndarray]:
        actions = sorted(getattr(planned, "actions", []), key=lambda a: a.order)

        if not actions:
            raise RuntimeError("no actions to stream -- prompt has no recognised actions")

        total_duration = getattr(planned, "duration", CONSTS.runtime.default_clip_duration_s)
        total_frames = int(total_duration * CONSTS.runtime.motion_fps)
        n_actions = len(actions)

        queries: list[tuple[str, int]] = []

        for action in actions:
            act_def = ACTIONS.get(action.action_type)
            query = build_action_query(action, act_def)
            n_frames = compute_action_frames(
                action, total_frames, n_actions, self.cfg.min_action_frames
            )
            queries.append((query, n_frames))

        yield from self.backend.stream_actions(
            queries,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )


def blend_init_pose(
    clip: MotionClip, init_pose: np.ndarray | None, blend_frames: int
) -> MotionClip:
    if init_pose is None or clip.smplx_params is None:
        return clip

    n = min(blend_frames, len(clip.smplx_params))

    if n <= 0:
        return clip

    alpha = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a = np.broadcast_to(init_pose[None], (n, init_pose.shape[0])).copy()
    clip.smplx_params[:n] = slerp_blend_frames(a, clip.smplx_params[:n], alpha)

    return clip
