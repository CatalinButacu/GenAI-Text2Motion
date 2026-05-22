"""Smoke test for the closed-loop StreamingRunner.

Mocks the motion model and the planner so we exercise the runner's
control flow (action queue, world updates, termination check) without
training a real model or downloading SBERT. The RVQ tokenizer is a real
freshly-initialised MotionRVQTokenizer with the causal decoder enabled.

Tested invariants:

  1. Runner yields frames at the right shape and rate.
  2. Termination predicates fire and stop the inner loop.
  3. World state updates per yielded frame.
  4. Multi-action plans transition via state.carry_over.

This is NOT a quality test -- with random weights the motion is noise.
The point is end-to-end wiring of planner -> motion -> tokenizer -> world
-> condition.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from src.modules.agent.conditions import parse_condition
from src.modules.agent.planner import PlannedAction
from src.modules.agent.runner import StreamingRunner
from src.modules.agent.world_state import WorldState
from src.modules.motion.rvq_tokenizer import MotionRVQTokenizer
from src.modules.motion.streaming import StreamingState

D_MODEL = 64
N_LAYERS = 2
D_INNER = 128  # = expand * d_model with expand=2
D_STATE = 16
RVQ_LATENT_DIM = 16
N_CODEBOOKS = 2
CODEBOOK_SIZE = 8
DOWN_T = 4
MAX_LATENT_STEPS = 8


def _make_streaming_state(batch: int = 1) -> StreamingState:
    return StreamingState(
        cond=torch.zeros(batch, D_MODEL),
        layer_h=[torch.zeros(batch, D_INNER, D_STATE) for _ in range(N_LAYERS)],
        layer_conv=[None] * N_LAYERS,
        latent_step=0,
        max_steps=MAX_LATENT_STEPS,
    )


def _fake_stream_step(state: StreamingState):
    """Generates one (B=1, T=1, K, V) logits tensor and advances the state."""
    state.latent_step += 1
    logits = torch.randn(1, 1, N_CODEBOOKS, CODEBOOK_SIZE)
    length_pred = torch.tensor([20.0])
    return logits, length_pred, state


def _make_runner(action_until: str = "duration(8)",
                 second_action: str | None = None) -> StreamingRunner:
    tokenizer = MotionRVQTokenizer(
        motion_dim=168,
        latent_dim=RVQ_LATENT_DIM,
        n_codebooks=N_CODEBOOKS,
        codebook_size=CODEBOOK_SIZE,
        down_t=DOWN_T,
        causal_decoder=True,
    ).eval()
    motion_model = MagicMock()
    state = _make_streaming_state()
    motion_model.stream_begin.return_value = state
    motion_model.stream_step.side_effect = _fake_stream_step
    # text_encoder + condition_proj only used on transitions (i >= 1)
    motion_model.text_encoder.return_value = torch.zeros(1, D_MODEL)
    motion_model.condition_proj.return_value = torch.zeros(1, D_MODEL)
    planner = MagicMock()
    actions = [
        PlannedAction(action_text="walk forward", until=parse_condition(action_until)),
    ]

    if second_action is not None:
        actions.append(
            PlannedAction(action_text="sit down", until=parse_condition(second_action)),
        )
    planner.return_value = actions
    world = WorldState()

    return StreamingRunner(planner, motion_model, tokenizer, world)


class TestStreamingRunner(unittest.TestCase):

    def test_yields_frames_with_right_shape(self):
        runner = _make_runner(action_until="duration(8)")
        frames = list(runner.run("walk forward"))
        self.assertGreater(len(frames), 0)

        for frame in frames:
            self.assertEqual(frame.shape, (168,))
            self.assertTrue(np.isfinite(frame).all())

    def test_termination_predicate_stops_action(self):
        """duration(4) must fire by frame 4 plus at most one decode chunk."""
        runner = _make_runner(action_until="duration(4)")
        frames = list(runner.run("test"))
        self.assertGreaterEqual(len(frames), 4)
        self.assertLessEqual(len(frames), 4 + DOWN_T)

    def test_world_state_increments_per_frame(self):
        runner = _make_runner(action_until="duration(6)")

        for i, frame in enumerate(runner.run("test"), start=1):
            self.assertEqual(runner.world.frames_in_action, i)

    def test_two_action_plan_transitions_via_carry_over(self):
        """A two-action plan must trigger state.carry_over on the second
        action and yield frames for both phases."""
        runner = _make_runner(action_until="duration(4)", second_action="duration(4)")
        frames = list(runner.run("walk then sit"))
        # Each action emits up to 4 + DOWN_T frames; total bounded above
        self.assertGreater(len(frames), 4)
        self.assertLessEqual(len(frames), (4 + DOWN_T) * 2)
        # The planner must have been called exactly once
        runner.planner.assert_called_once_with("walk then sit")
        # condition_proj is called once per transition (second action onward)
        self.assertEqual(runner.motion_model.condition_proj.call_count, 1)


if __name__ == "__main__":
    unittest.main()
