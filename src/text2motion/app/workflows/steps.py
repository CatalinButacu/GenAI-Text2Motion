from __future__ import annotations

from pathlib import Path
from typing import Any

from text2motion.app.cli import COMMANDS
from text2motion.app.cli_requests import PrepareRequest, TrainTokenizerRequest
from text2motion.app.workflows.requests import RunRequest, Workflow
from text2motion.common.runner import Step

DATA_STEP = "data"
TOKENIZER_STEP = "tokenizer"


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


def workflow_steps(app: Any, request: RunRequest) -> list[Step]:
    if request.workflow is Workflow.DATA:
        return [data_step(app, request)]
    if request.workflow is Workflow.TOKENIZER:
        return [data_step(app, request), tokenizer_step(app, request)]
    raise ValueError(f"workflow {request.workflow.value} is not built yet (A28-T6)")
