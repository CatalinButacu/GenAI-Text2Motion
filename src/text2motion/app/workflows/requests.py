from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field

from text2motion.app.cli_requests import DefaultConfigRequest
from text2motion.generation.contracts import GeneratorTrainingRequest
from text2motion.tokenization.contracts import TokenizerKind


############################################################
#       Workflow
############################################################
class Workflow(Enum):
    DATA = "data"
    TOKENIZER = "tokenizer"
    GENERATOR = "generator"
    EVALUATE = "evaluate"


############################################################
#       RunRequest
############################################################
class RunRequest(DefaultConfigRequest):
    workflow: Workflow = Field(
        description="chain of steps to run: data, tokenizer, generator or evaluate; "
        "a step with the same recipe hash is skipped"
    )
    root: Path = Field(
        description="folder for the step markers and manifest.json of this workflow run"
    )
    tokenizer: TokenizerKind | None = Field(
        default=None, description="quantizer family; required by the tokenizer workflow"
    )
    exploratory: bool = Field(
        default=False,
        description="record a non-protocol tokenizer run that is not citable",
    )
    epochs: int = Field(default=50, gt=0, description="tokenizer passes over the training set")
    max_steps: int | None = Field(
        default=None, gt=0, description="stop the tokenizer after this many optimizer steps"
    )
    tokenizer_ckpt: Path | None = Field(
        default=None,
        description="frozen tokenizer checkpoint for the generator steps; default from the config",
    )
    overfit_steps: int = Field(
        default=1000, gt=0, description="maximum optimizer steps of each overfit gate"
    )
    generator_epochs: int = Field(
        default=GeneratorTrainingRequest().epochs,
        gt=0,
        description="target epoch of each generator; the same for both backbones",
    )
