from __future__ import annotations

from pathlib import Path

from pydantic import Field

from text2motion.generation.contracts import Backbone, StudioSamplingConfig
from text2motion.motion.contracts import DefaultConfigRequest
from text2motion.motion.representation import FPS

DEFAULT_SERVICE_CONFIG = Path("configs/service.yml")


############################################################
#       ServiceAddressRequest
############################################################
class ServiceAddressRequest(DefaultConfigRequest):
    host: str | None = Field(default=None, description="service address; default from the config")
    port: int | None = Field(default=None, description="service TCP port; default from the config")
    service_config: Path = Field(
        default=DEFAULT_SERVICE_CONFIG,
        description="YAML file with the address and runtime settings of the service",
    )


############################################################
#       HealthRequest
############################################################
class HealthRequest(ServiceAddressRequest):
    pass


############################################################
#       ModelServiceRequest
############################################################
class ModelServiceRequest(ServiceAddressRequest):
    ckpt: Path | None = Field(
        default=None, description="generator checkpoint; default from the config"
    )
    tokenizer_ckpt: Path | None = Field(
        default=None,
        description="frozen tokenizer checkpoint; it must be the one the generator was trained on",
    )
    backbone: Backbone = Field(
        default=Backbone.TRANSFORMER,
        description="sequence mixer: mamba has a fixed-size state, transformer has a cache that grows",
    )


############################################################
#       ServeRequest
############################################################
class ServeRequest(ModelServiceRequest):
    idle_seconds: int | None = Field(
        default=None,
        description="the service stops after this many seconds without a request",
    )


############################################################
#       StudioRequest
############################################################
class StudioRequest(ModelServiceRequest):
    model_dir: Path | None = Field(
        default=None,
        description="SMPL-X model folder; without it the studio shows the skeleton only",
    )
    studio_config: Path = Field(
        default=Path("configs/studio.yml"),
        description="YAML file with display, fit and panel settings",
    )
    model: str | None = Field(
        default=None,
        description="key of a model in the model registry; it sets config, checkpoint and backbone",
    )
    steps: int = Field(
        default=0,
        description="token steps for each generation; one step is 4 frames; 0 uses the model maximum",
    )
    seconds: float = Field(
        default=0.0,
        description="motion length in seconds for each generation; above 0 it replaces --steps",
    )
    cfg_scale: float = Field(
        default=StudioSamplingConfig().cfg_scale,
        description="guidance scale; higher follows the text more closely and varies less",
    )
    temperature: float = Field(
        default=StudioSamplingConfig().temperature,
        description="sampling temperature; higher gives more varied motion",
    )
    top_p: float = Field(
        default=StudioSamplingConfig().top_p,
        description="nucleus threshold; lower removes unlikely tokens",
    )
    fps: int = Field(
        default=FPS, description="frame rate of the motion data, before display upsampling"
    )
