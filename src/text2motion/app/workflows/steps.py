from __future__ import annotations

from pathlib import Path
from typing import Any

from text2motion.app.cli import COMMANDS
from text2motion.app.cli_requests import (
    EvaluateRequest,
    PrepareRequest,
    SanityOverfitRequest,
    TokenizeRequest,
    TrainGeneratorRequest,
    TrainTokenizerRequest,
)
from text2motion.app.workflows.requests import RunRequest, Workflow
from text2motion.common.runner import Step
from text2motion.generation.contracts import Backbone, BackboneChoice

DATA_STEP = "data"
TOKENIZER_STEP = "tokenizer"
TOKENIZE_STEP = "tokenize"
EVALUATE_STEP = "evaluate"
BACKBONES = (Backbone.TRANSFORMER, Backbone.MAMBA)


def data_step(app: Any, request: RunRequest) -> Step:
    def run(inputs: dict[str, Path], output: Path) -> None:
        COMMANDS["prepare"].handler(app, PrepareRequest(config=request.config))
        output.write_text(str(app.config.paths.hml3d_out_dir), encoding="utf-8")

    return Step(DATA_STEP, (), DATA_STEP, run)


def tokenizer_step(app: Any, request: RunRequest) -> Step:
    if request.tokenizer is None:
        raise ValueError("the tokenizer workflow needs --tokenizer")
    kind = request.tokenizer
    arguments = TrainTokenizerRequest(
        config=request.config,
        tokenizer=kind,
        exploratory=request.exploratory,
        epochs=request.epochs,
        max_steps=request.max_steps,
    )

    def run(inputs: dict[str, Path], output: Path) -> None:
        COMMANDS["train-tokenizer"].handler(app, arguments)
        output.write_text(kind.value, encoding="utf-8")

    return Step(TOKENIZER_STEP, (DATA_STEP,), TOKENIZER_STEP, run)


def tokenizer_checkpoint(app: Any, request: RunRequest) -> Path:
    return request.tokenizer_ckpt or app.config.paths.tokenizer_checkpoint


def tokenize_step(app: Any, request: RunRequest) -> Step:
    arguments = TokenizeRequest(
        config=request.config, tokenizer_ckpt=tokenizer_checkpoint(app, request)
    )

    def run(inputs: dict[str, Path], output: Path) -> None:
        COMMANDS["tokenize"].handler(app, arguments)
        output.write_text(str(arguments.token_pack), encoding="utf-8")

    return Step(TOKENIZE_STEP, (TOKENIZER_STEP,), TOKENIZE_STEP, run)


def sanity_step(app: Any, request: RunRequest, backbone: Backbone) -> Step:
    name = f"sanity-overfit-{backbone.value}"

    def run(inputs: dict[str, Path], output: Path) -> None:
        arguments = SanityOverfitRequest(
            config=request.config,
            exploratory=request.exploratory,
            backbone=backbone,
            tokenizer_ckpt=tokenizer_checkpoint(app, request),
            overfit_gate=output,
            steps=request.overfit_steps,
        )
        COMMANDS["sanity-overfit"].handler(app, arguments)

    return Step(name, (TOKENIZE_STEP,), f"{name}.json", run)


def generator_step(app: Any, request: RunRequest, backbone: Backbone) -> Step:
    gate = f"sanity-overfit-{backbone.value}"
    name = f"train-generator-{backbone.value}"

    def run(inputs: dict[str, Path], output: Path) -> None:
        arguments = TrainGeneratorRequest(
            config=request.config,
            exploratory=request.exploratory,
            backbone=backbone,
            tokenizer_ckpt=tokenizer_checkpoint(app, request),
            overfit_gate=inputs[gate],
            epochs=request.generator_epochs,
        )
        COMMANDS["train-generator"].handler(app, arguments)
        output.write_text(backbone.value, encoding="utf-8")

    return Step(name, (gate,), name, run)


def generator_steps(app: Any, request: RunRequest) -> list[Step]:
    steps = [data_step(app, request), tokenizer_step(app, request), tokenize_step(app, request)]
    for backbone in BACKBONES:
        steps.append(sanity_step(app, request, backbone))
        steps.append(generator_step(app, request, backbone))
    return steps


def evaluate_step(app: Any, request: RunRequest) -> Step:
    arguments = EvaluateRequest(
        config=request.config,
        backbone=BackboneChoice.BOTH,
        tokenizer_ckpt=tokenizer_checkpoint(app, request),
    )

    def run(inputs: dict[str, Path], output: Path) -> None:
        COMMANDS["evaluate"].handler(app, arguments)
        output.write_text(arguments.split.value, encoding="utf-8")

    needs = tuple(f"train-generator-{backbone.value}" for backbone in BACKBONES)
    return Step(EVALUATE_STEP, needs, EVALUATE_STEP, run)


def workflow_steps(app: Any, request: RunRequest) -> list[Step]:
    if request.workflow is Workflow.DATA:
        return [data_step(app, request)]
    if request.workflow is Workflow.TOKENIZER:
        return [data_step(app, request), tokenizer_step(app, request)]
    if request.workflow is Workflow.GENERATOR:
        return generator_steps(app, request)
    return [*generator_steps(app, request), evaluate_step(app, request)]
