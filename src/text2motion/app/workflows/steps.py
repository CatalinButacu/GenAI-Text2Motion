from __future__ import annotations

from pathlib import Path
from typing import Any

from text2motion.app.cli import COMMANDS
from text2motion.app.config.schema import twin_checkpoint_name
from text2motion.app.requests.data import PrepareRequest, TokenizeRequest, TrainTokenizerRequest
from text2motion.app.requests.evaluate import EvaluateRequest
from text2motion.app.requests.generator import (
    PretrainRequest,
    SanityOverfitRequest,
    TrainGeneratorRequest,
)
from text2motion.app.workflows.requests import RunRequest, Workflow
from text2motion.common.runner import Step
from text2motion.generation.contracts import Backbone, BackboneChoice
from text2motion.motion.contracts import Split

DATA_STEP = "data"
TOKENIZER_STEP = "tokenizer"
TOKENIZE_STEP = "tokenize"
EVALUATE_STEP = "evaluate"
BACKBONES = (Backbone.TRANSFORMER, Backbone.MAMBA)
PRETRAINED_SUFFIX = "_pretrained"


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


def tokenize_step(app: Any, request: RunRequest, needs: str) -> Step:
    arguments = TokenizeRequest(
        config=request.config, tokenizer_ckpt=tokenizer_checkpoint(app, request)
    )

    def run(inputs: dict[str, Path], output: Path) -> None:
        COMMANDS["tokenize"].handler(app, arguments)
        output.write_text(str(arguments.token_pack), encoding="utf-8")

    return Step(TOKENIZE_STEP, (needs,), TOKENIZE_STEP, run)


def tokenizer_chain(app: Any, request: RunRequest) -> list[Step]:
    if request.tokenizer_ckpt is not None:
        return [tokenize_step(app, request, DATA_STEP)]
    return [tokenizer_step(app, request), tokenize_step(app, request, TOKENIZER_STEP)]


def twin_checkpoint(app: Any, request: RunRequest, backbone: Backbone, suffix: str = "") -> Path:
    name = twin_checkpoint_name(backbone, request.run_name + suffix)
    return Path(app.config.paths.checkpoints_dir) / "generator" / name


def resume_exists(checkpoint: Path) -> bool:
    return checkpoint.with_name(f"{checkpoint.stem}_last.pt").exists()


def pretrained_weights(app: Any, request: RunRequest, backbone: Backbone) -> Path | None:
    if request.pretrain_epochs is None:
        return None
    return twin_checkpoint(app, request, backbone, PRETRAINED_SUFFIX)


def pretrain_step(app: Any, request: RunRequest, backbone: Backbone) -> Step:
    if request.pretrain_epochs is None:
        raise ValueError("the pretrain step needs --pretrain_epochs")
    epochs = request.pretrain_epochs
    name = f"pretrain-{backbone.value}"
    checkpoint = twin_checkpoint(app, request, backbone, PRETRAINED_SUFFIX)

    def run(inputs: dict[str, Path], output: Path) -> None:
        arguments = PretrainRequest(
            config=request.config,
            exploratory=request.exploratory,
            backbone=backbone,
            token_pack=Path(inputs[TOKENIZE_STEP].read_text(encoding="utf-8")),
            epochs=epochs,
            ckpt=checkpoint,
            resume=resume_exists(checkpoint),
        )
        COMMANDS["pretrain"].handler(app, arguments)
        output.write_text(str(checkpoint), encoding="utf-8")

    return Step(name, (TOKENIZE_STEP,), name, run)


def gate_needs(request: RunRequest, backbone: Backbone) -> tuple[str, ...]:
    if request.pretrain_epochs is None:
        return (TOKENIZE_STEP,)
    return (f"pretrain-{backbone.value}",)


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
            init_ckpt=pretrained_weights(app, request, backbone),
        )
        COMMANDS["sanity-overfit"].handler(app, arguments)

    return Step(name, gate_needs(request, backbone), f"{name}.json", run)


def generator_step(app: Any, request: RunRequest, backbone: Backbone) -> Step:
    gate = f"sanity-overfit-{backbone.value}"
    name = f"train-generator-{backbone.value}"
    checkpoint = twin_checkpoint(app, request, backbone)

    def run(inputs: dict[str, Path], output: Path) -> None:
        arguments = TrainGeneratorRequest(
            config=request.config,
            exploratory=request.exploratory,
            backbone=backbone,
            tokenizer_ckpt=tokenizer_checkpoint(app, request),
            overfit_gate=inputs[gate],
            epochs=request.generator_epochs,
            ckpt_name=checkpoint.name,
            batch_size=request.batch_size,
            eval_every=request.eval_every,
            val_ce_patience=request.val_ce_patience,
            cfg_scale=request.cfg_scale,
            resume=resume_exists(checkpoint),
            init_ckpt=pretrained_weights(app, request, backbone),
        )
        COMMANDS["train-generator"].handler(app, arguments)
        output.write_text(backbone.value, encoding="utf-8")

    return Step(name, (gate,), name, run)


def generator_steps(app: Any, request: RunRequest) -> list[Step]:
    steps = [data_step(app, request), *tokenizer_chain(app, request)]
    for backbone in BACKBONES:
        if request.pretrain_epochs is not None:
            steps.append(pretrain_step(app, request, backbone))
        steps.append(sanity_step(app, request, backbone))
        steps.append(generator_step(app, request, backbone))
    return steps


def evaluate_step(app: Any, request: RunRequest) -> Step:
    arguments = EvaluateRequest(
        config=request.config,
        backbone=BackboneChoice.BOTH,
        split=Split.VALIDATION,
        tokenizer_ckpt=tokenizer_checkpoint(app, request),
        run_name=request.run_name,
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
