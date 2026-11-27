from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from text2motion.generation.decoder import StreamingMotionDecoder
from text2motion.generation.pipeline import TextToMotionGenerator


############################################################
#       LoadedInferencePipeline
############################################################
@dataclass(frozen=True)
class LoadedInferencePipeline:
    generator: TextToMotionGenerator
    decoder: StreamingMotionDecoder
    backbone: str
    checkpoint: str
    downsample: int
    trained_steps: int
    max_steps: int
    device: str
    identity: dict[str, str]
    anchor_every: int = 0
    anchor_context: int = 16

    def hello(self) -> dict:
        return {
            "type": "hello",
            "identity": self.identity,
            "ckpt": self.checkpoint,
            "backbone": self.backbone,
            "downsample": self.downsample,
            "trained_steps": self.trained_steps,
            "max_steps": self.max_steps,
            "device": self.device,
        }


type InferencePipelineLoader = Callable[[dict], LoadedInferencePipeline]
