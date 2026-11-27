from __future__ import annotations

import argparse
import functools
import importlib
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from jsonargparse import ArgumentParser as RequestParser
from jsonargparse.typing import register_type
from pydantic import BaseModel

from text2motion.app.runtime.run_log import LOG_LEVELS, configure_logging

logger = logging.getLogger(__name__)


############################################################
#       Command
############################################################
def imported(path: str) -> Any:
    module_name, attribute = path.split(":")
    return getattr(importlib.import_module(module_name), attribute)


@dataclass(frozen=True)
class Command:
    request: str
    run: str
    summary: str

    @property
    def request_model(self) -> Any:
        return imported(self.request)

    @property
    def handler(self) -> Any:
        return imported(self.run)


COMMANDS = MappingProxyType(
    {
        "prepare": Command(
            "text2motion.app.requests.data:PrepareRequest",
            "text2motion.app.stages.data:run_prepare",
            "AMASS -> HumanML3D-263 data preparation",
        ),
        "tokenize": Command(
            "text2motion.app.requests.data:TokenizeRequest",
            "text2motion.app.stages.data:run_tokenize",
            "encode a feature corpus into tokens",
        ),
        "train-tokenizer": Command(
            "text2motion.app.requests.data:TrainTokenizerRequest",
            "text2motion.app.stages.train_tokenizer:run_train_tokenizer",
            "train the motion tokenizer",
        ),
        "sanity-overfit": Command(
            "text2motion.app.requests.generator:SanityOverfitRequest",
            "text2motion.app.stages.overfit:run_sanity_overfit",
            "mandatory real-batch overfit gate",
        ),
        "train-generator": Command(
            "text2motion.app.requests.generator:TrainGeneratorRequest",
            "text2motion.app.stages.train_generator:run_train_generator",
            "train the text-to-motion generator",
        ),
        "pretrain": Command(
            "text2motion.app.requests.generator:PretrainRequest",
            "text2motion.app.stages.pretrain:run_pretrain",
            "unconditional token-corpus pretraining",
        ),
        "evaluate": Command(
            "text2motion.app.requests.evaluate:EvaluateRequest",
            "text2motion.app.stages.evaluate:run_evaluate",
            "citable generation eval (20-rep protocol)",
        ),
        "benchmark": Command(
            "text2motion.app.requests.evaluate:BenchmarkRequest",
            "text2motion.app.stages.evaluate:run_bench",
            "streaming latency/state-size benchmark",
        ),
        "serve": Command(
            "text2motion.app.requests.serve:ServeRequest",
            "text2motion.app.stages.serve:run_serve",
            "local streaming generation service",
        ),
        "studio": Command(
            "text2motion.app.requests.serve:StudioRequest",
            "text2motion.app.stages.serve:run_studio",
            "interactive aitviewer studio",
        ),
        "run": Command(
            "text2motion.app.workflows.requests:RunRequest",
            "text2motion.app.workflows.run:run_workflow",
            "run a workflow: data, tokenizer, generator or evaluate",
        ),
        "health": Command(
            "text2motion.app.requests.base:HealthRequest",
            "text2motion.app.stages.serve:run_health",
            "check the streaming service readiness",
        ),
    }
)


def register_value_enum(enum_class: type[Enum]) -> None:
    def member(text: str) -> Enum:
        return enum_class[text] if text in enum_class.__members__ else enum_class(text)

    register_type(enum_class, serializer=lambda item: item.value, deserializer=member)


@functools.cache
def register_command_enums() -> None:
    from text2motion.app.workflows.requests import Workflow
    from text2motion.evaluation.contracts import GenerationLengthPolicy
    from text2motion.generation.contracts import Backbone, BackboneChoice
    from text2motion.motion.contracts import PreparationStage, Split
    from text2motion.tokenization.contracts import TokenizerKind

    for command_enum in (
        Backbone,
        BackboneChoice,
        GenerationLengthPolicy,
        PreparationStage,
        Split,
        TokenizerKind,
        Workflow,
    ):
        register_value_enum(command_enum)


def with_flag_values(request_model: type[BaseModel], arguments: list[str]) -> list[str]:
    flags = {
        f"--{name}"
        for name, field in request_model.model_fields.items()
        if field.annotation is bool
    }
    completed: list[str] = []
    for index, argument in enumerate(arguments):
        completed.append(argument)
        following = arguments[index + 1] if index + 1 < len(arguments) else "--"
        if argument in flags and following.startswith("--"):
            completed.append("true")
    return completed


def parse_request(command: str, arguments: list[str]) -> Any:
    register_command_enums()
    selected = COMMANDS[command]
    request_model = selected.request_model
    parser = RequestParser(prog=f"text2motion {command}", description=selected.summary)
    parser.add_class_arguments(request_model, nested_key=None)
    parsed = parser.parse_args(with_flag_values(request_model, arguments))
    return request_model.model_validate(parsed.as_dict())


def build_parser() -> argparse.ArgumentParser:
    summaries = "\n".join(f"  {name:16s} {command.summary}" for name, command in COMMANDS.items())
    parser = argparse.ArgumentParser(
        prog="text2motion",
        description="Streaming text-to-motion: one entry point per stage.",
        epilog=f"commands:\n{summaries}\n\ntext2motion <command> --help shows the flags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=LOG_LEVELS,
        help="DEBUG adds detail lines; WARNING shows problems only",
    )
    parser.add_argument("command", choices=list(COMMANDS), metavar="command")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else list(argv)
    selected = build_parser().parse_args(arguments)
    configure_logging(selected.log_level)
    logger.info(f"command: text2motion {' '.join(arguments)}")
    request = parse_request(selected.command, selected.arguments)
    from text2motion.app.runtime.context import ApplicationBootstrap

    app = ApplicationBootstrap.from_config(
        request.config,
        device=getattr(request, "device", None),
        seed=getattr(request, "seed", None),
    )
    try:
        COMMANDS[selected.command].handler(app, request)
    except Exception:
        if app.config.failure_logs:
            logger.exception(f"command {selected.command} failed")
        raise


if __name__ == "__main__":
    main()
