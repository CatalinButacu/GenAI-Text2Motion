"""Closed-loop coordinator for the streaming avatar demo.

Ties together M5's planner (action decomposition), M3's MotionSSM
(streaming latent generation), the frozen RVQ tokenizer (latent -> raw
frames), the world-state tracker, and the runtime condition predicates.

The runner is a generator: it yields raw 168-d SMPL-X frames as soon as
they're available, one per call to the inner loop. Callers route those
frames to whatever sink they want (renderer, JSON log, MP4 writer, ...).

The runner relies on the causal RVQ decoder (``causal_decoder=True`` at
tokenizer build time) so that frames emitted at latent step t do not
shift when later latents arrive. Without it, incremental-decode would
silently rewrite past frames.

Pipeline per action (one iteration of the closed loop):

  1. Initialise (or carry over) the streaming state from the previous
     action -- carry_over preserves the SSM hidden state so transitions
     are continuous (the Mamba-specific thesis claim).
  2. Reset per-action world bookkeeping (yaw_at_action_start, frames_in_action).
  3. Loop:
       a. stream_step the SSM to emit one new latent token.
       b. Decode latents-so-far into raw frames (cheap with the causal
          decoder; only new frames are emitted to the caller).
       c. For each NEW raw frame: update world state, yield it, then
          check the termination predicate.
       d. If the predicate fires, break inner loop -> advance action queue.
       e. If the SSM hits max_steps without termination -> log + advance.

The frames yielded are CPU numpy arrays, ready for the renderer.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import torch

from .planner import PlannedAction
from .world_state import WorldState

log = logging.getLogger(__name__)

# Structural types: the runner only needs callables, not specific classes.
# Keeps the test surface mockable and avoids beartype rejecting MagicMock.
PlannerFn = Callable[[str], list[PlannedAction]]


class StreamingRunner:
    """Stateful avatar runner. Build once per demo session; call :meth:`run`
    per user instruction. World state persists across calls so multi-turn
    instructions land the avatar in a coherent position.
    """

    def __init__(
        self,
        planner: PlannerFn,
        motion_model: Any,    # TextToMotionSSM (kept structural for test mocks)
        tokenizer: Any,       # MotionRVQTokenizer (frozen, causal_decoder=True)
        world: WorldState,
        max_action_latents: int | None = None,
    ) -> None:
        self.planner = planner
        self.motion_model = motion_model
        self.tokenizer = tokenizer
        self.world = world
        # Optional cap on latents per action -- defense against the LM
        # emitting an action whose termination condition never fires
        self.max_action_latents = max_action_latents

        if not getattr(tokenizer, "causal_decoder", False):
            log.warning(
                "[runner] tokenizer.causal_decoder is False; incremental "
                "decode will silently rewrite past frames as later latents "
                "arrive. Use causal_decoder=True for streaming-correct "
                "playback."
            )

    def run(self, instruction: str) -> Iterator[np.ndarray]:
        """Plan an instruction and yield raw frames as they're generated.

        Calls the planner once up-front to get the action list; if the
        planner raises (unparseable LM output), the exception propagates.
        Once the plan is in hand, runs each action in sequence, yielding
        every emitted frame in order.
        """
        plan = self.planner(instruction)
        log.info("[runner] instruction=%r decomposed into %d actions", instruction, len(plan))
        state = None

        for i, action in enumerate(plan):
            log.info("[runner] action %d/%d: %r (until %s)",
                     i + 1, len(plan), action.action_text, action.until.source)

            if i == 0:
                state = self.motion_model.stream_begin([action.action_text])
            else:
                # Transition: keep SSM hidden state, swap text condition.
                # carry_over also resets the per-action latent_step counter.
                assert state is not None
                new_cond = self.motion_model.condition_proj(
                    self.motion_model.text_encoder([action.action_text])
                )
                state.carry_over(new_cond)
            assert state is not None
            self.world.reset_for_new_action()
            yield from self.run_one_action(action, state)

    def run_one_action(
        self, action: PlannedAction, state
    ) -> Iterator[np.ndarray]:
        """Yield frames for one action until its termination predicate fires
        or the SSM exhausts its positional capacity.
        """
        tokens_buf: list[torch.Tensor] = []
        last_emitted_idx = 0
        down_t = self.tokenizer.down_t
        cap = self.max_action_latents or state.max_steps

        while state.latent_step < min(state.max_steps, cap):
            with torch.no_grad():
                logits, _, state = self.motion_model.stream_step(state)
            # Argmax sampling is the default; CFG / temperature would slot
            # in here. Logits shape: (B=1, 1, K, V) -> tokens (B=1, 1, K).
            tokens = logits.argmax(dim=-1)
            tokens_buf.append(tokens)

            # Incremental decode. The CAUSAL decoder makes past frames
            # invariant to future latents, so the only new frames are
            # the last `down_t` slots. We slice them out and emit.
            all_tokens = torch.cat(tokens_buf, dim=1)

            with torch.no_grad():
                motion = self.tokenizer.decode(all_tokens)[0]  # (T_raw, D)
            new_frames = motion[last_emitted_idx:].detach().cpu().numpy()
            last_emitted_idx = motion.shape[0]

            for j in range(new_frames.shape[0]):
                frame = new_frames[j]
                prev_pos = self.world.position.copy()
                self.world.update_from_frame(frame, prev_position=prev_pos)
                yield frame

                if action.until(self.world):
                    log.info(
                        "[runner] termination fired at latent_step=%d "
                        "frames_in_action=%d",
                        state.latent_step, self.world.frames_in_action,
                    )

                    return
        log.warning(
            "[runner] action %r reached latent cap (%d) without termination; "
            "advancing to next action",
            action.action_text, min(state.max_steps, cap),
        )
