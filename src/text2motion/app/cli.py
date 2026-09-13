from __future__ import annotations

import argparse
from pathlib import Path

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.app.commands.data import run_prepare, run_tokenize
from text2motion.app.commands.evaluation import run_bench, run_evaluate
from text2motion.app.commands.generator import run_pretrain, run_sanity_overfit, run_train_generator
from text2motion.app.commands.serving import run_health, run_serve, run_studio
from text2motion.app.commands.tokenizer import run_train_tokenizer
from text2motion.evaluation.contracts import (
    GenerationEvaluationProtocol,
    GenerationLengthPolicy,
    StreamingBenchmarkProtocol,
)
from text2motion.generation.contracts import (
    Backbone,
    BackboneChoice,
    GeneratorPretrainingRequest,
    GeneratorTrainingRequest,
    SamplingConfig,
)
from text2motion.motion.contracts import MotionDataConfig, PreparationStage, Split
from text2motion.motion.representation import FPS
from text2motion.tokenization.contracts import (
    TokenizerKind,
    TokenizerTrainingRequest,
    TokenPackRequest,
)

_DEFAULT_CONFIG = "configs/default.yaml"


def add_config(parser: argparse.ArgumentParser, required: bool = True) -> None:
    parser.add_argument(
        "--config",
        required=required,
        default=None if required else _DEFAULT_CONFIG,
        help="path to a YAML config",
    )


def _add_data_commands(commands) -> None:
    data_defaults = MotionDataConfig()
    corpus_defaults = TokenPackRequest(features_dir=Path(), out_path=Path())
    prepare = commands.add_parser("prepare", help="AMASS -> HumanML3D-263 data preparation")
    add_config(prepare, required=False)
    prepare.add_argument("--stage", default=PreparationStage.ALL, type=PreparationStage, choices=list(PreparationStage))
    prepare.add_argument("--corpus", action="store_true")
    prepare.add_argument("--regen_dir", default="data/HumanML3D_263")
    prepare.add_argument("--out_dir", default="data/AMASS_263")
    prepare.add_argument("--min_frames", type=int, default=data_defaults.min_motion_len)
    prepare.add_argument("--limit", type=int, default=0)
    prepare.add_argument("--source_release", default=None)
    prepare.add_argument("--out", default="data/HumanML3D_official")
    prepare.set_defaults(func=run_prepare)

    tokenize = commands.add_parser("tokenize", help="encode a feature corpus into tokens")
    add_config(tokenize)
    tokenize.add_argument("--features_dir", default="data/AMASS_263/new_joint_vecs")
    tokenize.add_argument("--tokenizer_ckpt", default=None)
    tokenize.add_argument("--out", default="data/amass_tokens.npz")
    tokenize.add_argument("--segment_frames", type=int, default=corpus_defaults.segment_frames)
    tokenize.add_argument("--stride", type=int, default=corpus_defaults.stride)
    tokenize.set_defaults(func=run_tokenize)


def _add_tokenizer_command(commands) -> None:
    defaults = TokenizerTrainingRequest()
    train = commands.add_parser("train-tokenizer", help="train the motion tokenizer")
    add_config(train)
    train.add_argument(
        "--tokenizer", required=True, type=TokenizerKind, choices=list(TokenizerKind)
    )
    train.add_argument("--epochs", type=int, default=defaults.epochs)
    train.add_argument("--window", type=int, default=defaults.window)
    train.add_argument("--batch_size", type=int, default=defaults.batch_size)
    train.add_argument("--ema_decay", type=float, default=defaults.ema_decay)
    train.add_argument("--eval_every", type=int, default=defaults.eval_every)
    train.add_argument("--max_eval_clips", type=int, default=defaults.max_eval_clips)
    train.add_argument("--num_workers", type=int, default=defaults.num_workers)
    train.add_argument("--ckpt_name", default=None)
    train.add_argument("--resume", action="store_true")
    train.set_defaults(func=run_train_tokenizer)


def _add_generation_commands(commands) -> None:
    train_defaults = GeneratorTrainingRequest()
    pretrain_defaults = GeneratorPretrainingRequest(token_pack=Path())
    sanity = commands.add_parser("sanity-overfit", help="mandatory real-batch overfit gate")
    add_config(sanity)
    sanity.add_argument("--backbone", required=True, type=Backbone, choices=list(Backbone))
    sanity.add_argument("--tokenizer_ckpt", required=True)
    sanity.add_argument("--out", required=True)
    sanity.add_argument("--batch_size", type=int, default=4)
    sanity.add_argument("--steps", type=int, default=1000)
    sanity.add_argument("--lr", type=float, default=3e-3)
    sanity.set_defaults(func=run_sanity_overfit)

    train_gen = commands.add_parser("train-generator", help="train the text-to-motion generator")
    add_config(train_gen)
    train_gen.add_argument("--backbone", required=True, type=Backbone, choices=list(Backbone))
    train_gen.add_argument("--tokenizer_ckpt", default=None)
    train_gen.add_argument("--overfit_gate", required=True)
    train_gen.add_argument("--init_ckpt", default=None)
    train_gen.add_argument("--epochs", type=int, default=train_defaults.epochs)
    train_gen.add_argument("--batch_size", type=int, default=train_defaults.batch_size)
    train_gen.add_argument("--grad_accum", type=int, default=train_defaults.grad_accum)
    train_gen.add_argument("--cfg_scale", type=float, default=train_defaults.cfg_scale)
    train_gen.add_argument(
        "--eval_split",
        default=train_defaults.eval_split,
        type=Split,
        choices=[Split.VALIDATION, Split.TEST],
    )
    train_gen.add_argument("--eval_every", type=int, default=train_defaults.eval_every)
    train_gen.add_argument("--max_eval_clips", type=int, default=train_defaults.max_eval_clips)
    train_gen.add_argument("--eval_batch", type=int, default=train_defaults.eval_batch_size)
    train_gen.add_argument("--temperature", type=float, default=train_defaults.temperature)
    train_gen.add_argument("--our_vab_dir", default=None)
    train_gen.add_argument("--ckpt_name", default=None)
    train_gen.add_argument("--resume", action="store_true")
    train_gen.add_argument("--patience", type=int, default=train_defaults.patience)
    train_gen.set_defaults(func=run_train_generator)

    pretrain = commands.add_parser("pretrain", help="unconditional token-corpus pretraining")
    add_config(pretrain)
    pretrain.add_argument("--backbone", required=True, type=Backbone, choices=list(Backbone))
    pretrain.add_argument("--token_pack", default="data/amass_tokens_fsq8x1024.npz")
    pretrain.add_argument("--epochs", type=int, default=pretrain_defaults.epochs)
    pretrain.add_argument("--batch_size", type=int, default=pretrain_defaults.batch_size)
    pretrain.add_argument("--grad_accum", type=int, default=pretrain_defaults.grad_accum)
    pretrain.add_argument("--num_workers", type=int, default=pretrain_defaults.num_workers)
    pretrain.add_argument("--val_fraction", type=float, default=pretrain_defaults.val_fraction)
    pretrain.add_argument("--out", default=None)
    pretrain.add_argument("--resume", action="store_true")
    pretrain.set_defaults(func=run_pretrain)


def _add_evaluation_commands(commands) -> None:
    defaults = GenerationEvaluationProtocol()
    sampling = SamplingConfig()
    benchmark = StreamingBenchmarkProtocol()
    evaluate = commands.add_parser("evaluate", help="citable generation eval (20-rep protocol)")
    add_config(evaluate, required=False)
    evaluate.add_argument(
        "--backbone",
        default=BackboneChoice.BOTH,
        type=BackboneChoice,
        choices=list(BackboneChoice),
    )
    evaluate.add_argument(
        "--split", default=defaults.split, type=Split, choices=[Split.VALIDATION, Split.TEST]
    )
    evaluate.add_argument("--max_clips", type=int, default=defaults.max_clips)
    evaluate.add_argument("--temperature", type=float, default=sampling.temperature)
    evaluate.add_argument("--top_p", type=float, default=sampling.top_p)
    evaluate.add_argument("--cfg_scale", type=float, default=sampling.cfg_scale)
    evaluate.add_argument(
        "--length_mode",
        default=GenerationLengthPolicy.FIXED,
        type=GenerationLengthPolicy,
        choices=list(GenerationLengthPolicy),
    )
    evaluate.add_argument("--mm_clips", type=int, default=defaults.mm_clips)
    evaluate.add_argument("--mm_repeats", type=int, default=defaults.mm_repeats)
    evaluate.add_argument("--bootstrap", type=int, default=defaults.bootstrap)
    evaluate.add_argument("--eval_batch", type=int, default=defaults.batch_size)
    evaluate.add_argument("--ckpt", default=None)
    evaluate.add_argument("--tokenizer_ckpt", default=None)
    evaluate.add_argument("--text_encoder_ckpt", default=None)
    evaluate.add_argument("--device", default=None)
    evaluate.add_argument("--seed", type=int, default=None)
    evaluate.set_defaults(func=run_evaluate)

    bench = commands.add_parser("benchmark", help="streaming latency/state-size benchmark")
    add_config(bench, required=False)
    bench.add_argument("--horizons", type=int, nargs="+", default=list(benchmark.horizons))
    bench.add_argument("--device", default=None)
    bench.add_argument("--out", default=None)
    bench.add_argument("--warmup", type=int, default=benchmark.warmup)
    bench.add_argument("--streams", type=int, nargs="*", default=list(benchmark.streams))
    bench.add_argument("--streams_horizon", type=int, default=benchmark.streams_horizon)
    bench.set_defaults(func=run_bench)


def _add_runtime_commands(commands) -> None:
    sampling = SamplingConfig()
    serve = commands.add_parser("serve", help="local streaming generation service")
    add_config(serve, required=False)
    serve.add_argument("--ckpt", default=None)
    serve.add_argument("--tokenizer_ckpt", default=None)
    serve.add_argument(
        "--backbone", default=Backbone.TRANSFORMER, type=Backbone, choices=list(Backbone)
    )
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--idle_seconds", type=int, default=None)
    serve.set_defaults(func=run_serve)

    studio = commands.add_parser("studio", help="interactive aitviewer studio")
    add_config(studio, required=False)
    studio.add_argument("--ckpt", default=None)
    studio.add_argument("--tokenizer_ckpt", default=None)
    studio.add_argument(
        "--backbone", default=Backbone.TRANSFORMER, type=Backbone, choices=list(Backbone)
    )
    studio.add_argument("--model_dir", default=None)
    studio.add_argument("--studio_config", default="configs/studio.yml")
    studio.add_argument("--model", default=None)
    studio.add_argument("--steps", type=int, default=0)
    studio.add_argument("--seconds", type=float, default=0.0)
    studio.add_argument("--cfg_scale", type=float, default=6.0)
    studio.add_argument("--temperature", type=float, default=sampling.temperature)
    studio.add_argument("--top_p", type=float, default=sampling.top_p)
    studio.add_argument("--fps", type=int, default=FPS)
    studio.add_argument("--host", default=None)
    studio.add_argument("--port", type=int, default=None)
    studio.set_defaults(func=run_studio)

    health = commands.add_parser("health", help="check the streaming service readiness")
    add_config(health, required=False)
    health.add_argument("--host", default=None)
    health.add_argument("--port", type=int, default=None)
    health.set_defaults(func=run_health)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text2motion", description="Streaming text-to-motion: one entry point per stage."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    _add_data_commands(commands)
    _add_tokenizer_command(commands)
    _add_generation_commands(commands)
    _add_evaluation_commands(commands)
    _add_runtime_commands(commands)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    app = ApplicationBootstrap.from_config(
        args.config,
        device=getattr(args, "device", None),
        seed=getattr(args, "seed", None),
    )
    args.func(app, args)


if __name__ == "__main__":
    main()
