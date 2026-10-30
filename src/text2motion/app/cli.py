from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from jsonargparse import ArgumentParser as RequestParser
from jsonargparse.typing import register_type
from pydantic import BaseModel

from text2motion.app.bootstrap import ApplicationBootstrap, ApplicationContext
from text2motion.app.commands.data import run_prepare, run_tokenize
from text2motion.app.commands.evaluation import run_bench, run_evaluate
from text2motion.app.commands.generator import run_pretrain, run_sanity_overfit, run_train_generator
from text2motion.app.commands.serving import run_health, run_serve, run_studio
from text2motion.app.commands.tokenizer import run_train_tokenizer
from text2motion.app.run_log import LOG_LEVELS, configure_logging
from text2motion.evaluation.contracts import (
    BenchmarkRequest,
    EvaluateRequest,
    GenerationLengthPolicy,
)
from text2motion.generation.contracts import (
    Backbone,
    BackboneChoice,
    PretrainRequest,
    SanityOverfitRequest,
    TrainGeneratorRequest,
)
from text2motion.motion.contracts import (
    CommandRequest,
    PreparationStage,
    PrepareRequest,
    Split,
)
from text2motion.streaming.pydant_requests import HealthRequest, ServeRequest, StudioRequest
from text2motion.tokenization.contracts import (
    TokenizeRequest,
    TokenizerKind,
    TrainTokenizerRequest,
)

logger = logging.getLogger(__name__)
COMMAND_ENUMS = (
    Backbone,
    BackboneChoice,
    GenerationLengthPolicy,
    PreparationStage,
    Split,
    TokenizerKind,
)


############################################################
#       Command
############################################################
@dataclass(frozen=True)
class Command:
    request_model: type[CommandRequest]
    handler: Callable[[ApplicationContext, Any], None]
    summary: str


COMMANDS = MappingProxyType(
    {
        "prepare": Command(PrepareRequest, run_prepare, "AMASS -> HumanML3D-263 data preparation"),
        "tokenize": Command(TokenizeRequest, run_tokenize, "encode a feature corpus into tokens"),
        "train-tokenizer": Command(
            TrainTokenizerRequest, run_train_tokenizer, "train the motion tokenizer"
        ),
        "sanity-overfit": Command(
            SanityOverfitRequest, run_sanity_overfit, "mandatory real-batch overfit gate"
        ),
        "train-generator": Command(
            TrainGeneratorRequest, run_train_generator, "train the text-to-motion generator"
        ),
        "pretrain": Command(
            PretrainRequest, run_pretrain, "unconditional token-corpus pretraining"
        ),
        "evaluate": Command(
            EvaluateRequest, run_evaluate, "citable generation eval (20-rep protocol)"
        ),
        "benchmark": Command(BenchmarkRequest, run_bench, "streaming latency/state-size benchmark"),
        "serve": Command(ServeRequest, run_serve, "local streaming generation service"),
        "studio": Command(StudioRequest, run_studio, "interactive aitviewer studio"),
        "health": Command(HealthRequest, run_health, "check the streaming service readiness"),
    }
)


def register_value_enum(enum_class: type[Enum]) -> None:
    def member(text: str) -> Enum:
        return enum_class[text] if text in enum_class.__members__ else enum_class(text)

    register_type(enum_class, serializer=lambda item: item.value, deserializer=member)


for command_enum in COMMAND_ENUMS:
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


def parse_request(command: str, arguments: list[str]) -> CommandRequest:
    selected = COMMANDS[command]
    parser = RequestParser(prog=f"text2motion {command}", description=selected.summary)
    parser.add_class_arguments(selected.request_model, nested_key=None)
    parsed = parser.parse_args(with_flag_values(selected.request_model, arguments))
    return selected.request_model.model_validate(parsed.as_dict())


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
