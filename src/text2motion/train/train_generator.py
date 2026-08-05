from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from text2motion.data.hml3d.dataset import build_dataloader
from text2motion.eval.context import EvalContext, GenerationPipeline, SamplingCfg
from text2motion.eval.generation_eval import evaluate_generation
from text2motion.model.generator import MotionGenerator
from text2motion.model.text_encoder import CLIPTextEncoder
from text2motion.model.tokenizer import ResidualFsqTokenizer
from text2motion.shared.config import load_config
from text2motion.shared.run_log import log_metrics, start_run
from text2motion.shared.seed import seed_everything
from text2motion.train.trainer import GeneratorTrainer


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    seed_everything(cfg.seed, cfg.deterministic)  # twin fairness (ADR 0001) + bit-exact determinism

    loader = build_dataloader(cfg.paths, cfg.hml3d, cfg.data, "train", args.batch_size)

    tokenizer = ResidualFsqTokenizer(cfg.tokenizer)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location="cpu"))
    tokenizer.to(device).eval()

    n_layers = cfg.generator.mamba_n_layers if args.backbone == "mamba" else cfg.generator.n_layers
    gen_cfg = replace(
        cfg.generator,
        backbone=args.backbone,
        n_layers=n_layers,
        num_codebooks=cfg.tokenizer.num_quantizers,
        codebook_size=tokenizer.codebook_size,
    )
    generator = MotionGenerator(gen_cfg).to(device)
    if args.init_ckpt:  # initialise from the AMASS-pretrained motion prior (fresh optimizer/epoch)
        generator.load_state_dict(torch.load(args.init_ckpt, map_location=device))
        print(f"initialised generator from pretrained {args.init_ckpt}")
    text_encoder = CLIPTextEncoder(cfg.text_encoder).to(device)

    out_dir = Path(cfg.paths.hml3d_out_dir)
    our_mean = np.load(out_dir / "Mean.npy").astype(np.float32)
    our_std = np.load(out_dir / "Std.npy").astype(np.float32)

    train_cfg = replace(cfg.train, grad_accum=args.grad_accum)
    trainer = GeneratorTrainer(generator, tokenizer, train_cfg, text_encoder, our_mean, our_std)
    trainer.build_scheduler(args.epochs * max(1, len(loader) // args.grad_accum))
    print(
        f"generator params: {sum(p.numel() for p in generator.parameters()):,} "
        f"(backbone {args.backbone}, {n_layers} layers)"
    )

    eval_ctx = EvalContext.from_config(cfg, device=device, our_vab_dir=args.our_vab_dir)
    eval_pipeline = GenerationPipeline(generator, text_encoder, tokenizer)
    eval_sampling = SamplingCfg(temperature=args.temperature, cfg_scale=args.cfg_scale)

    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = (
        args.ckpt_name or f"generator_{args.backbone}.pt"
    )  # ablations must not clobber winners
    ckpt_path = ckpt_dir / ckpt_name
    resume_path = ckpt_dir / f"{Path(ckpt_name).stem}_last.pt"
    run_dir = start_run(f"generator_{args.backbone}", cfg, cfg.paths.outputs_dir, extra=vars(args))
    log_metrics(run_dir, {"train_clips": len(loader.dataset), "device": device})
    print(f"train clips: {len(loader.dataset)}  device: {device}  backbone: {args.backbone}")

    best_fid = float("inf")
    start_epoch = 0
    if args.resume and resume_path.is_file():
        state = torch.load(resume_path, map_location="cpu")
        generator.load_state_dict(state["generator"])
        text_encoder.load_state_dict(state["text_encoder"])
        trainer.opt.load_state_dict(state["optimizer"])  # Adam moments cast to param device
        trainer.ema.shadow = {k: v.to(device) for k, v in state["ema"].items()}
        if trainer.scheduler is not None and state.get("scheduler") is not None:
            trainer.scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_fid = state["best_fid"]
        del state
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"resumed from {resume_path} at epoch {start_epoch} (best FID {best_fid:.4f})")

    for epoch in range(start_epoch, args.epochs):
        generator.train()
        text_encoder.train()
        totals: dict[str, float] = {}
        steps = 0
        for motion, lengths, captions in loader:
            text_emb = trainer.encode(list(captions))
            parts = trainer.train_step(motion.to(device), text_emb, lengths.to(device))
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            steps += 1
            if steps % 100 == 0:  # intra-epoch heartbeat: a slow 3h epoch must not read as a hang
                (Path(run_dir) / "heartbeat").write_text(str(steps), encoding="utf-8")
        means = {k: v / steps for k, v in totals.items()}
        log_metrics(run_dir, {"epoch": epoch + 1, **means})
        print(
            f"epoch {epoch + 1:3d}  ce {means['ce']:.4f}  ric {means['ric']:.4f}  "
            f"rot6d {means['rot6d']:.4f}  foot {means['foot']:.4f}  total {means['total']:.4f}"
        )

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            trainer.ema.copy_to(generator)
            metrics = evaluate_generation(
                eval_pipeline,
                eval_ctx,
                sampling=eval_sampling,
                max_clips=args.max_eval_clips,
                split=args.eval_split,
            )
            trainer.ema.restore(generator)
            log_metrics(run_dir, {"epoch": epoch + 1, "split": args.eval_split, **metrics})
            print(
                f"  [gen-eval:{args.eval_split}] clips {metrics['clips']}  FID {metrics['fid']:.4f}  "
                f"R@1 {metrics['r_top1']:.3f}  R@3 {metrics['r_top3']:.3f}  "
                f"MM {metrics['mm_dist']:.3f}  Div {metrics['diversity']:.3f}"
            )
            if metrics["fid"] < best_fid:
                best_fid = metrics["fid"]
                trainer.ema.copy_to(generator)
                torch.save(generator.state_dict(), ckpt_path)
                trainer.ema.restore(generator)
                encoder_path = Path(ckpt_path).with_name(Path(ckpt_path).stem + "_text_encoder.pt")
                torch.save({"text_encoder": text_encoder.state_dict()}, encoder_path)
                print(f"  saved best -> {ckpt_path} (FID {best_fid:.4f}) + {encoder_path.name}")

        torch.save(
            {
                "generator": generator.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "optimizer": trainer.opt.state_dict(),
                "ema": trainer.ema.shadow,
                "scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
                "epoch": epoch,
                "best_fid": best_fid,
            },
            resume_path,
        )

        if device == "cuda":
            torch.cuda.empty_cache()  # defrag at epoch boundary (4GB-card fragmentation OOM lesson)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the text-to-motion generator (Contribution B)."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", required=True, choices=["mamba", "transformer"])
    parser.add_argument(
        "--tokenizer_ckpt",
        default="checkpoints/tokenizer/tokenizer_fsq.pt",
        help="frozen tokenizer state_dict to load; must match cfg.tokenizer architecture",
    )
    parser.add_argument(
        "--init_ckpt",
        default=None,
        help="pretrained generator weights to initialise from (AMASS pretrain); fresh optimizer",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="micro-batches per optimizer step; grad_accum x batch_size = effective batch",
    )
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG at in-train eval")
    parser.add_argument("--eval_split", default="val", choices=["val", "test"])
    parser.add_argument(
        "--eval_every", type=int, default=5
    )  # the peak sits early; sample it densely
    parser.add_argument("--max_eval_clips", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--our_vab_dir", default="data/t2m_glove/glove")
    parser.add_argument(
        "--ckpt_name", default=None, help="checkpoint filename (ablations must not clobber winners)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume from <ckpt_name stem>_last.pt"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
