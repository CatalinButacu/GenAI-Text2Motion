from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from text2motion.generation.model.generator import MotionTokenGenerator


############################################################
#       EagerDecodeStep
############################################################
class EagerDecodeStep:
    def __init__(self, model: MotionTokenGenerator, state: list, text_vector: torch.Tensor) -> None:
        self.model = model
        self.state = state
        self.text_vector = text_vector

    def __call__(self, feed: torch.Tensor) -> torch.Tensor:
        embedded = self.model.add_per_step_text(
            self.model.embed_motion_tokens(feed), self.text_vector
        )
        h, self.state = self.model.backbone.step(embedded, self.state)
        return self.model.predict_token_logits(h)


############################################################
#       CapturedDecodeStep
############################################################
class CapturedDecodeStep:
    warmup_steps = 3

    def __init__(self, model: MotionTokenGenerator, batch: int, text_vector: torch.Tensor) -> None:
        device = next(model.parameters()).device
        codebooks = model.cfg.num_codebooks
        self.tokens = torch.zeros(batch, codebooks, dtype=torch.long, device=device)
        self.text_vector = text_vector.detach().clone()
        self.state = model.backbone.init_state(batch, device)
        side_stream = torch.cuda.Stream(device)
        side_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side_stream):
            for _ in range(self.warmup_steps):
                EagerDecodeStep(model, list(self.state), self.text_vector)(self.tokens)
        torch.cuda.current_stream(device).wait_stream(side_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            embedded = model.add_per_step_text(
                model.embed_motion_tokens(self.tokens), self.text_vector
            )
            h, next_state = model.backbone.step(embedded, list(self.state))
            for (ssm, conv), (next_ssm, next_conv) in zip(self.state, next_state, strict=True):
                ssm.copy_(next_ssm)
                conv.copy_(next_conv)
            self.logits = model.predict_token_logits(h)

    def load_state(self, state: list) -> None:
        for (ssm, conv), (source_ssm, source_conv) in zip(self.state, state, strict=True):
            ssm.copy_(source_ssm)
            conv.copy_(source_conv)

    def load_text_vector(self, text_vector: torch.Tensor) -> None:
        if text_vector.shape != self.text_vector.shape:
            raise ValueError(
                f"the captured decode step holds a text vector of shape "
                f"{tuple(self.text_vector.shape)}, got {tuple(text_vector.shape)}"
            )
        self.text_vector.copy_(text_vector)

    def __call__(self, feed: torch.Tensor) -> torch.Tensor:
        self.tokens.copy_(feed)
        self.graph.replay()
        return self.logits
