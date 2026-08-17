from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from text2motion.app.checkpoint import GeneratorBundle, OverfitGate, sha256_file
from text2motion.app.container import DEFAULT_TOKENIZER_CKPT, Application, ApplicationFactory
from text2motion.app.run_log import log_metrics, start_run
from text2motion.app.schema import load_dataclass
from text2motion.evaluation.benchmark import BenchmarkRequest, run_benchmark
from text2motion.evaluation.evaluator import EvaluationRequest
from text2motion.evaluation.matcher import LengthMode
from text2motion.generation.model import Backbone, BackboneChoice
from text2motion.generation.pipeline import MotionGenerator, SamplingConfig
from text2motion.generation.pretrain import PretrainingRequest, pretrain_generator
from text2motion.generation.trainer import (
    Amp,
    GeneratorTrainer,
    GeneratorTrainingRequest,
    overfit_single_batch,
    train_generator,
)
from text2motion.motion.dataset import Split
from text2motion.motion.preparation import (
    Stage,
    build_corpus,
    convert_official_release,
    regenerate,
)
from text2motion.streaming.decoder import StreamingMotionDecoder
from text2motion.streaming.protocol import Wire
from text2motion.streaming.service import ServiceModel, StreamingService
from text2motion.tokenization.corpus import CorpusRequest, tokenize_corpus
from text2motion.tokenization.evaluation import evaluate_reconstruction
from text2motion.tokenization.model import TokenizerKind
from text2motion.tokenization.trainer import (
    TokenizerTrainer,
    TokenizerTrainingRequest,
    train_tokenizer,
)

DEFAULT_CONFIG = "configs/default.yaml"


def enum_values(kind) -> list[str]:
    return [member.value for member in kind]


def add_config(parser: argparse.ArgumentParser, required: bool = True) -> None:
    if required:
        parser.add_argument("--config", required=True, help="path to a YAML config")
    else:
        parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to a YAML config")


def run_prepare(args: argparse.Namespace) -> None:
    if args.source_release:
        convert_official_release(Path(args.source_release), Path(args.out))
        return

    app = ApplicationFactory.from_config(args.config)
    request = app.preparation_request()

    if args.corpus:
        run_dir = start_run("pretrain_corpus", app.config, app.paths.outputs_dir, vars(args))
        summary = build_corpus(
            request, Path(args.regen_dir), Path(args.out_dir), args.min_frames, args.limit
        )
        log_metrics(run_dir, summary)
        print(summary)
        return

    regenerate(request, args.stage)


def run_train_tokenizer(args: argparse.Namespace) -> None:
    app = ApplicationFactory.from_config(args.config)
    config = replace(app.config, tokenizer=replace(app.config.tokenizer, kind=args.tokenizer))
    app.config = config

    module = app.tokenizer_module().to(app.device)
    commit_beta = (
        config.rvq_baseline.commitment_beta if args.tokenizer is TokenizerKind.RVQ else 0.0
    )
    trainer = TokenizerTrainer(
        module,
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        ema_decay=args.ema_decay,
        commit_beta=commit_beta,
    )

    repository = app.motions()
    loader = repository.window_loader(
        Split.TRAIN, args.window, args.batch_size, num_workers=args.num_workers
    )
    evaluator = app.evaluator()
    scaler = repository.scaler()

    def evaluate(split: Split) -> dict[str, float]:
        return evaluate_reconstruction(
            module,
            repository.root,
            scaler,
            downsample=config.tokenizer.downsample,
            device=app.device,
            max_clips=args.max_eval_clips,
            split=split,
            recon_fid=evaluator.recon_fid,
        )

    name = args.ckpt_name or f"tokenizer_{args.tokenizer.value}.pt"
    checkpoint_dir = app.checkpoints_dir()
    run_dir = start_run(f"tokenizer_{Path(name).stem}", config, app.paths.outputs_dir, vars(args))
    print(
        f"train clips: {len(loader.dataset)}  device: {app.device}  "
        f"codebook: {module.codebook_size}"
    )

    train_tokenizer(
        trainer,
        loader,
        TokenizerTrainingRequest(
            epochs=args.epochs,
            window=args.window,
            batch_size=args.batch_size,
            ema_decay=args.ema_decay,
            eval_every=args.eval_every,
            max_eval_clips=args.max_eval_clips,
            num_workers=args.num_workers,
            checkpoint_name=name,
            resume=args.resume,
        ),
        device=app.device,
        evaluate=evaluate,
        checkpoint_path=checkpoint_dir / name,
        resume_path=checkpoint_dir / f"{Path(name).stem}_last.pt",
        on_metrics=lambda record: log_metrics(run_dir, record),
    )

    best_path = checkpoint_dir / name
    if best_path.is_file():
        module.load_state_dict(torch.load(best_path, map_location=app.device))
        module.eval()
        test_metrics = evaluate(Split.TEST)
        log_metrics(
            run_dir,
            {"epoch": args.epochs, "split": Split.TEST.value, "final": True, **test_metrics},
        )
        print(
            f"  [TEST once | val-selected best] clips {test_metrics['clips']}  "
            f"recon-FID {test_metrics['recon_fid']:.4f}  MPJPE {test_metrics['mpjpe_mm']:.1f}mm"
        )


def run_tokenize(args: argparse.Namespace) -> None:
    app = ApplicationFactory.from_config(args.config)
    tokenizer = app.tokenizer(args.tokenizer_ckpt)
    run_dir = start_run("tokenize_corpus", app.config, app.paths.outputs_dir, vars(args))
    summary = tokenize_corpus(
        tokenizer,
        app.motions().scaler(),
        CorpusRequest(
            features_dir=Path(args.features_dir),
            out_path=Path(args.out),
            segment_frames=args.segment_frames,
            stride=args.stride,
        ),
    )
    log_metrics(run_dir, summary)
    print(summary)


def build_generator_trainer(
    app: Application, args: argparse.Namespace, dropout: float | None = None
) -> tuple[GeneratorTrainer, object]:
    tokenizer = app.tokenizer(args.tokenizer_ckpt)
    overrides = {} if dropout is None else {"dropout": dropout}
    architecture = app.architecture(args.backbone, tokenizer.codebook_size, **overrides)
    return tokenizer, architecture


def run_sanity_overfit(args: argparse.Namespace) -> None:
    app = ApplicationFactory.from_config(args.config)
    repository = app.motions()
    loader = repository.loader(Split.TRAIN, args.batch_size)
    batch = next(iter(loader))
    motion = batch.features.to(app.device)
    lengths = batch.lengths.to(app.device)

    tokenizer, architecture = build_generator_trainer(app, args, dropout=0.0)
    tokenizer.module.requires_grad_(False)
    scaler = repository.scaler()
    train_config = replace(
        app.config.train,
        lr=args.lr,
        cfg_dropout=0.0,
        pkeep=1.0,
        amp=Amp.OFF,
        grad_accum=1,
        ema_decay=0.0,
    )
    trainer = GeneratorTrainer(
        architecture.build(app.device),
        tokenizer.module,
        train_config,
        downsample=app.config.tokenizer.downsample,
        text_encoder=app.text_encoder(),
        mean=scaler.mean,
        std=scaler.std,
    )

    metrics = overfit_single_batch(trainer, motion, lengths, list(batch.captions), args.steps)
    gate = OverfitGate(
        config_sha256=sha256_file(args.config),
        tokenizer_sha256=sha256_file(args.tokenizer_ckpt),
        backbone=args.backbone,
        resolved_generator_config=architecture.as_dict(),
        batch_size=args.batch_size,
        **metrics,
    )
    gate.validate_metrics()
    gate.save(args.out)
    print(f"PASS: wrote real-batch overfit gate -> {args.out}")


def run_train_generator(args: argparse.Namespace) -> None:
    app = ApplicationFactory.from_config(args.config)
    tokenizer = app.tokenizer(args.tokenizer_ckpt)
    architecture = app.architecture(args.backbone, tokenizer.codebook_size)

    OverfitGate.load(args.overfit_gate).verify_setup(
        config_path=args.config,
        tokenizer_ckpt=args.tokenizer_ckpt,
        backbone=args.backbone,
        resolved_config=architecture.as_dict(),
    )
    print(f"verified mandatory real-batch overfit gate: {args.overfit_gate}")

    module = architecture.build(app.device)
    if args.init_ckpt:
        module.load_state_dict(torch.load(args.init_ckpt, map_location=app.device))
        print(f"initialised generator from pretrained {args.init_ckpt}")

    text_encoder = app.text_encoder()
    repository = app.motions()
    loader = repository.loader(Split.TRAIN, args.batch_size)
    scaler = repository.scaler()

    trainer = GeneratorTrainer(
        module,
        tokenizer.module,
        replace(app.config.train, grad_accum=args.grad_accum),
        downsample=app.config.tokenizer.downsample,
        text_encoder=text_encoder,
        mean=scaler.mean,
        std=scaler.std,
    )
    trainer.build_scheduler(args.epochs * max(1, len(loader) // args.grad_accum))
    print(
        f"generator params: {sum(p.numel() for p in module.parameters()):,} "
        f"({architecture.parameter_label})"
    )

    evaluator = app.evaluator(args.our_vab_dir)
    pipeline = MotionGenerator(module, text_encoder, tokenizer)
    sampling = SamplingConfig(temperature=args.temperature, cfg_scale=args.cfg_scale)

    name = args.ckpt_name or f"generator_{args.backbone.value}.pt"
    checkpoint_dir = app.checkpoints_dir()
    checkpoint_path = checkpoint_dir / name
    run_dir = start_run(
        f"generator_{args.backbone.value}", app.config, app.paths.outputs_dir, extra=vars(args)
    )
    log_metrics(run_dir, {"train_clips": len(loader.dataset), "device": app.device})
    print(
        f"train clips: {len(loader.dataset)}  device: {app.device}  backbone: {args.backbone.value}"
    )

    def evaluate(split: Split) -> dict[str, float]:
        return evaluator.evaluate_generation(
            pipeline,
            split=split,
            max_clips=args.max_eval_clips,
            sampling=sampling,
            batch_size=args.eval_batch,
        )

    def save_best(epoch: int, validation_fid: float) -> None:
        GeneratorBundle.capture(
            module,
            text_encoder,
            backbone=args.backbone,
            tokenizer_ckpt=args.tokenizer_ckpt,
            resolved_generator_config=architecture.as_dict(),
            epoch=epoch,
            validation_fid=validation_fid,
        ).save(checkpoint_path)
        print(f"  saved atomic best bundle -> {checkpoint_path} (FID {validation_fid:.4f})")

    train_generator(
        trainer,
        loader,
        GeneratorTrainingRequest(
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            eval_every=args.eval_every,
            eval_split=args.eval_split,
            max_eval_clips=args.max_eval_clips,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            checkpoint_name=name,
            resume=args.resume,
        ),
        device=app.device,
        evaluate=evaluate,
        save_best=save_best,
        resume_path=checkpoint_dir / f"{Path(name).stem}_last.pt",
        run_dir=run_dir,
        on_metrics=lambda record: log_metrics(run_dir, record),
    )


def run_pretrain(args: argparse.Namespace) -> None:
    app = ApplicationFactory.from_config(args.config)
    vocab = 1
    for level in app.config.tokenizer.fsq_levels:
        vocab *= level

    architecture = app.architecture(args.backbone, vocab)
    module = architecture.build(app.device)
    print(
        f"pretrain {args.backbone.value}: {sum(p.numel() for p in module.parameters()):,} params, "
        f"codebooks {architecture.config.num_codebooks} x vocab {vocab}"
    )

    out_path = Path(args.out or f"checkpoints/generator_{args.backbone.value}_pretrained.pt")
    run_dir = start_run(
        f"pretrain_{args.backbone.value}", app.config, app.paths.outputs_dir, vars(args)
    )
    pretrain_generator(
        architecture,
        module,
        PretrainingRequest(
            token_pack=Path(args.token_pack),
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            num_workers=args.num_workers,
            val_fraction=args.val_fraction,
            out_path=out_path,
            resume=args.resume,
        ),
        train=app.config.train,
        device=app.device,
        out_path=out_path,
        on_metrics=lambda record: log_metrics(run_dir, record),
        seed=app.config.seed,
    )


def run_evaluate(args: argparse.Namespace) -> None:
    app = ApplicationFactory.from_config(args.config, device=args.device, seed=args.seed)
    tokenizer = app.tokenizer(args.tokenizer_ckpt)
    evaluator = app.evaluator()

    backbones = (
        [Backbone.TRANSFORMER, Backbone.MAMBA]
        if args.backbone is BackboneChoice.BOTH
        else [Backbone(args.backbone)]
    )
    request = EvaluationRequest(
        split=args.split,
        max_clips=args.max_clips,
        sampling=SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            cfg_scale=args.cfg_scale,
            stop_at_end=args.length_mode is LengthMode.END_TOKEN,
        ),
        mm_clips=args.mm_clips,
        mm_repeats=args.mm_repeats,
        bootstrap=args.bootstrap,
        batch_size=args.eval_batch,
    )

    run_dir = start_run(
        f"evaluate_{args.split.value}", app.config, app.paths.outputs_dir, vars(args)
    )
    print(
        f"device {app.device}  split {args.split.value}  cfg_scale {args.cfg_scale}  "
        f"temp {args.temperature}  top_p {args.top_p}  length {args.length_mode.value}  "
        f"(20-rep, our_vab)"
    )

    for backbone in backbones:
        checkpoint = args.ckpt if len(backbones) == 1 else None
        checkpoint = checkpoint or f"checkpoints/generator_{backbone.value}.pt"
        if not Path(checkpoint).is_file():
            print(f"{backbone.value:12s} SKIP (no checkpoint yet)")
            continue

        generator = app.generator(
            backbone,
            checkpoint,
            tokenizer,
            tokenizer_checkpoint=args.tokenizer_ckpt,
            text_encoder_checkpoint=args.text_encoder_ckpt,
        )
        report = evaluator.evaluate(generator, request)
        log_metrics(
            run_dir, {"backbone": backbone.value, "ckpt": str(checkpoint), **report.as_dict()}
        )
        top1, top2, top3 = report.r_precision
        print(
            f"{backbone.value:12s} clips {report.clips:4d}  FID {report.fid:6.3f}  "
            f"R@1 {top1:.3f}+/-{report.r_top1_std:.3f}  R@2 {top2:.3f}  R@3 {top3:.3f}  "
            f"MM {report.matching_distance:.3f}  Div {report.diversity:.3f}  "
            f"MModality {report.multimodality:.3f}"
        )
        del generator
        if app.device == "cuda":
            torch.cuda.empty_cache()


def run_bench(args: argparse.Namespace) -> None:
    app = ApplicationFactory.from_config(args.config, device=args.device)
    run_benchmark(
        BenchmarkRequest(
            horizons=tuple(args.horizons),
            warmup=args.warmup,
            streams=tuple(args.streams),
            streams_horizon=args.streams_horizon,
            out_path=Path(args.out),
            label=str(args.config),
        ),
        generator=app.config.generator,
        codebook_size=app.config.generator.codebook_size,
        num_codebooks=app.config.tokenizer.num_quantizers,
        seed=app.config.seed,
        device=app.device,
    )


def build_service_model(args: argparse.Namespace) -> ServiceModel:
    app = ApplicationFactory.from_config(args.config)
    tokenizer = app.tokenizer(args.tokenizer_ckpt)
    generator = app.generator(
        args.backbone,
        args.ckpt,
        tokenizer,
        tokenizer_checkpoint=args.tokenizer_ckpt,
        use_kernel=False,
    )
    scaler = app.motions().scaler()
    decoder = StreamingMotionDecoder(
        tokenizer.module,
        downsample=app.config.tokenizer.downsample,
        mean=torch.from_numpy(scaler.mean).to(app.device),
        std=torch.from_numpy(scaler.std).to(app.device),
    )
    return ServiceModel(
        generator=generator,
        decoder=decoder,
        backbone=str(args.backbone),
        checkpoint=str(args.ckpt),
        downsample=app.config.tokenizer.downsample,
        max_steps=app.config.generator.max_seq_len + 1 - app.config.generator.text_prefix_len,
        device=app.device,
    )


def run_serve(args: argparse.Namespace) -> None:
    def loader(request: dict) -> ServiceModel:
        return build_service_model(
            argparse.Namespace(
                config=request["config"],
                ckpt=request["ckpt"],
                tokenizer_ckpt=request["tokenizer_ckpt"],
                backbone=Backbone(request["backbone"]),
            )
        )

    StreamingService(build_service_model(args), loader=loader).serve(
        args.host, args.port, args.idle_minutes * 60
    )


def run_studio(args: argparse.Namespace) -> None:
    from text2motion.streaming.protocol import connect_or_spawn
    from text2motion.studio.config import StudioConfig
    from text2motion.studio.viewer import StreamingStudioViewer

    studio_config = (
        load_dataclass(StudioConfig, args.studio_config)
        if Path(args.studio_config).is_file()
        else StudioConfig()
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] connecting to the motion service (backbone={args.backbone})...")
    client = connect_or_spawn(
        args.config,
        args.ckpt,
        args.tokenizer_ckpt,
        str(args.backbone),
        args.host,
        args.port,
    )
    model_dir = args.model_dir or str(studio_config.fit.model_dir)
    model_dir = model_dir if model_dir and Path(model_dir).exists() else ""
    if not model_dir:
        print("[load] SMPL-X model dir not found -> skeleton-only studio")

    StreamingStudioViewer(
        client=client,
        model_dir=model_dir,
        device=device,
        args=args,
        config=studio_config,
    ).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text2motion", description="Streaming text-to-motion: one entry point per stage."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="AMASS -> HumanML3D-263 data preparation")
    add_config(prepare, required=False)
    prepare.add_argument(
        "--stage", default=Stage.ALL, type=Stage, choices=list(Stage), help="pipeline stage(s)"
    )
    prepare.add_argument("--corpus", action="store_true", help="build the all-AMASS 263 corpus")
    prepare.add_argument("--regen_dir", default="data/HumanML3D_263")
    prepare.add_argument("--out_dir", default="data/AMASS_263")
    prepare.add_argument("--min_frames", type=int, default=40)
    prepare.add_argument("--limit", type=int, default=0)
    prepare.add_argument("--source_release", default=None, help="parquet release to convert")
    prepare.add_argument("--out", default="data/HumanML3D_official")
    prepare.set_defaults(func=run_prepare)

    train_tok = subparsers.add_parser("train-tokenizer", help="train the motion tokenizer")
    add_config(train_tok)
    train_tok.add_argument(
        "--tokenizer", required=True, type=TokenizerKind, choices=list(TokenizerKind)
    )
    train_tok.add_argument("--epochs", type=int, default=50)
    train_tok.add_argument("--window", type=int, default=64)
    train_tok.add_argument("--batch_size", type=int, default=128)
    train_tok.add_argument("--ema_decay", type=float, default=0.99)
    train_tok.add_argument("--eval_every", type=int, default=5)
    train_tok.add_argument("--max_eval_clips", type=int, default=None)
    train_tok.add_argument("--num_workers", type=int, default=0)
    train_tok.add_argument("--ckpt_name", default=None)
    train_tok.add_argument("--resume", action="store_true")
    train_tok.set_defaults(func=run_train_tokenizer)

    tokenize = subparsers.add_parser("tokenize", help="encode a feature corpus into tokens")
    add_config(tokenize)
    tokenize.add_argument("--features_dir", default="data/AMASS_263/new_joint_vecs")
    tokenize.add_argument("--tokenizer_ckpt", default=DEFAULT_TOKENIZER_CKPT)
    tokenize.add_argument("--out", default="data/amass_tokens.npz")
    tokenize.add_argument("--segment_frames", type=int, default=196)
    tokenize.add_argument("--stride", type=int, default=196)
    tokenize.set_defaults(func=run_tokenize)

    sanity = subparsers.add_parser("sanity-overfit", help="mandatory real-batch overfit gate")
    add_config(sanity)
    sanity.add_argument("--backbone", required=True, type=Backbone, choices=list(Backbone))
    sanity.add_argument("--tokenizer_ckpt", required=True)
    sanity.add_argument("--out", required=True)
    sanity.add_argument("--batch_size", type=int, default=4)
    sanity.add_argument("--steps", type=int, default=1000)
    sanity.add_argument("--lr", type=float, default=3e-3)
    sanity.set_defaults(func=run_sanity_overfit)

    train_gen = subparsers.add_parser("train-generator", help="train the text-to-motion generator")
    add_config(train_gen)
    train_gen.add_argument("--backbone", required=True, type=Backbone, choices=list(Backbone))
    train_gen.add_argument("--tokenizer_ckpt", default=DEFAULT_TOKENIZER_CKPT)
    train_gen.add_argument("--overfit_gate", required=True)
    train_gen.add_argument("--init_ckpt", default=None)
    train_gen.add_argument("--epochs", type=int, default=60)
    train_gen.add_argument("--batch_size", type=int, default=64)
    train_gen.add_argument("--grad_accum", type=int, default=1)
    train_gen.add_argument("--cfg_scale", type=float, default=1.0)
    train_gen.add_argument(
        "--eval_split", default=Split.VALIDATION, type=Split, choices=[Split.VALIDATION, Split.TEST]
    )
    train_gen.add_argument("--eval_every", type=int, default=5)
    train_gen.add_argument("--max_eval_clips", type=int, default=600)
    train_gen.add_argument("--eval_batch", type=int, default=32)
    train_gen.add_argument("--temperature", type=float, default=1.0)
    train_gen.add_argument("--our_vab_dir", default="data/t2m_glove/glove")
    train_gen.add_argument("--ckpt_name", default=None)
    train_gen.add_argument("--resume", action="store_true")
    train_gen.set_defaults(func=run_train_generator)

    pretrain = subparsers.add_parser("pretrain", help="unconditional token-corpus pretraining")
    add_config(pretrain)
    pretrain.add_argument("--backbone", required=True, type=Backbone, choices=list(Backbone))
    pretrain.add_argument("--token_pack", default="data/amass_tokens_fsq8x1024.npz")
    pretrain.add_argument("--epochs", type=int, default=30)
    pretrain.add_argument("--batch_size", type=int, default=64)
    pretrain.add_argument("--grad_accum", type=int, default=1)
    pretrain.add_argument("--num_workers", type=int, default=0)
    pretrain.add_argument("--val_fraction", type=float, default=0.05)
    pretrain.add_argument("--out", default=None)
    pretrain.add_argument("--resume", action="store_true")
    pretrain.set_defaults(func=run_pretrain)

    evaluate = subparsers.add_parser("evaluate", help="citable generation eval (20-rep protocol)")
    add_config(evaluate, required=False)
    evaluate.add_argument(
        "--backbone", default=BackboneChoice.BOTH, type=BackboneChoice, choices=list(BackboneChoice)
    )
    evaluate.add_argument(
        "--split", default=Split.TEST, type=Split, choices=[Split.VALIDATION, Split.TEST]
    )
    evaluate.add_argument("--max_clips", type=int, default=100000)
    evaluate.add_argument("--temperature", type=float, default=1.0)
    evaluate.add_argument("--top_p", type=float, default=0.9)
    evaluate.add_argument("--cfg_scale", type=float, default=1.0)
    evaluate.add_argument(
        "--length_mode", default=LengthMode.FIXED, type=LengthMode, choices=list(LengthMode)
    )
    evaluate.add_argument("--mm_clips", type=int, default=100)
    evaluate.add_argument("--mm_repeats", type=int, default=30)
    evaluate.add_argument("--bootstrap", type=int, default=200)
    evaluate.add_argument("--eval_batch", type=int, default=32)
    evaluate.add_argument("--ckpt", default=None)
    evaluate.add_argument("--tokenizer_ckpt", default=DEFAULT_TOKENIZER_CKPT)
    evaluate.add_argument("--text_encoder_ckpt", default=None)
    evaluate.add_argument("--device", default=None)
    evaluate.add_argument("--seed", type=int, default=None)
    evaluate.set_defaults(func=run_evaluate)

    bench = subparsers.add_parser("benchmark", help="streaming latency/state-size benchmark")
    add_config(bench, required=False)
    bench.add_argument("--horizons", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    bench.add_argument("--device", default=None)
    bench.add_argument("--out", default="outputs/streaming_bench.json")
    bench.add_argument("--warmup", type=int, default=32)
    bench.add_argument("--streams", type=int, nargs="*", default=[])
    bench.add_argument("--streams_horizon", type=int, default=256)
    bench.set_defaults(func=run_bench)

    serve = subparsers.add_parser("serve", help="local streaming generation service")
    add_config(serve, required=False)
    serve.add_argument("--ckpt", default="checkpoints/generator/generator_transformer_100m.pt")
    serve.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/fsq_g8_v1024.pt")
    serve.add_argument(
        "--backbone", default=Backbone.TRANSFORMER, type=Backbone, choices=list(Backbone)
    )
    serve.add_argument("--host", default=Wire.HOST)
    serve.add_argument("--port", type=int, default=Wire.PORT)
    serve.add_argument("--idle_minutes", type=int, default=10)
    serve.set_defaults(func=run_serve)

    studio = subparsers.add_parser("studio", help="interactive aitviewer studio")
    add_config(studio, required=False)
    studio.add_argument("--ckpt", default="checkpoints/generator/generator_transformer_100m.pt")
    studio.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/fsq_g8_v1024.pt")
    studio.add_argument(
        "--backbone", default=Backbone.TRANSFORMER, type=Backbone, choices=list(Backbone)
    )
    studio.add_argument("--model_dir", default=None)
    studio.add_argument("--studio_config", default="configs/studio.yml")
    studio.add_argument("--steps", type=int, default=49)
    studio.add_argument("--cfg_scale", type=float, default=6.0)
    studio.add_argument("--temperature", type=float, default=1.0)
    studio.add_argument("--top_p", type=float, default=0.9)
    studio.add_argument("--fps", type=int, default=20)
    studio.add_argument("--host", default=Wire.HOST)
    studio.add_argument("--port", type=int, default=Wire.PORT)
    studio.set_defaults(func=run_studio)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
