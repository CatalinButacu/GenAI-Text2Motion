from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from text2motion.generation.model.decode_step import CapturedDecodeStep, EagerDecodeStep
from text2motion.generation.model.mamba import MambaBackbone

if TYPE_CHECKING:
    from text2motion.generation.model.generator import MotionTokenGenerator


class CudaGraphDecode:
    def __init__(self, model: MotionTokenGenerator) -> None:
        self.model = model
        self.captured_steps: dict[int, CapturedDecodeStep] | None = None

    def enable(self) -> None:
        if not isinstance(self.model.backbone, MambaBackbone):
            raise ValueError(
                f"cuda-graph decode needs the fixed-size recurrent state of the mamba backbone, "
                f"got {type(self.model.backbone).__name__}; set service.cuda_graph_decode: false"
            )
        device = next(self.model.parameters()).device
        if device.type != "cuda":
            raise ValueError(
                f"cuda-graph decode needs a cuda device, the generator is on {device}; "
                "set service.cuda_graph_decode: false"
            )
        self.captured_steps = {}

    def decode_step_for(self, state: list, batch: int, text_vector: torch.Tensor):
        if self.captured_steps is None:
            return EagerDecodeStep(self.model, state, text_vector)
        if batch not in self.captured_steps:
            self.captured_steps[batch] = CapturedDecodeStep(self.model, batch, text_vector)
        captured = self.captured_steps[batch]
        captured.load_state(state)
        captured.load_text_vector(text_vector)
        return captured
