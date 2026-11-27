from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field

from text2motion.app.requests.base import DefaultConfigRequest
from text2motion.generation.requests import GeneratorTrainingRequest
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
    run_name: str = Field(
        default="twins",
        pattern=r"^[A-Za-z0-9_]+$",
        description="twin run name; checkpoints are generator_<backbone>_<run_name>.pt",
    )
    pretrain_epochs: int | None = Field(
        default=None,
        gt=0,
        description="unconditional pretraining epochs before each overfit gate; none skips it",
    )
    overfit_steps: int = Field(
        default=1000, gt=0, description="maximum optimizer steps of each overfit gate"
    )
    batch_size: int = Field(
        default=8, gt=0, description="generator batch size; the same for both backbones"
    )
    eval_every: int = Field(
        default=5, gt=0, description="epochs between validation runs; the protocol uses 5"
    )
    val_ce_patience: int = Field(
        default=10,
        gt=0,
        description="stop after this many evaluations without a lower validation CE; protocol 10",
    )
    cfg_scale: float = Field(
        default=6.0, gt=0, description="classifier-free guidance scale for validation sampling"
    )
    generator_epochs: int = Field(
        default=GeneratorTrainingRequest().epochs,
        gt=0,
        description="target epoch of each generator; the same for both backbones",
    )
