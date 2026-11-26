from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

DEFAULT_SERVICE_CONFIG = Path("configs/service.yml")


DEFAULT_CONFIG = Path("configs/default.yaml")


CONFIG_HELP = "YAML file with model, data and path settings; it selects the experiment"


############################################################
#       CommandRequest
############################################################
class CommandRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    config: Path = Field(description=CONFIG_HELP)

    @field_validator("*", mode="before")
    @classmethod
    def member_from_name(cls, value: Any, info: ValidationInfo) -> Any:
        annotation = cls.model_fields[info.field_name or ""].annotation
        if not (isinstance(annotation, type) and issubclass(annotation, Enum)):
            return value
        if isinstance(value, str) and value in annotation.__members__:
            return annotation[value]
        return value


############################################################
#       DefaultConfigRequest
############################################################
class DefaultConfigRequest(CommandRequest):
    config: Path = Field(default=DEFAULT_CONFIG, description=CONFIG_HELP)


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
