import argparse
from dataclasses import replace

import torch

from text2motion.app.config import load_config
from text2motion.generation.contracts import Backbone
from text2motion.generation.model import GeneratorModelSpec
from text2motion.generation.trainer import GeneratorTrainer
from text2motion.motion.representation import DIM
from text2motion.tokenization.model import build_tokenizer_network


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = build_tokenizer_network(config.tokenizer, config.rvq_baseline)
    tokenizer.load_state_dict(torch.load(args.tokenizer_ckpt, map_location="cpu"))
    tokenizer.to(device).eval()

    for backbone in Backbone:
        architecture = GeneratorModelSpec.resolve(
            config.generator,
            backbone,
            tokenizer.codebook_size,
            config.tokenizer.num_quantizers,
            use_kernel=False,
        )
        torch.manual_seed(config.seed)
        generator = architecture.create_model(device)
        trainer = GeneratorTrainer(
            generator,
            tokenizer,
            replace(config.train, lr=3e-4, cfg_dropout=0.0, pkeep=1.0),
            downsample=config.tokenizer.downsample,
        )
        parameter_count = sum(parameter.numel() for parameter in generator.parameters())

        motion = torch.randn(args.batch_size, args.window, DIM, device=device)
        text = torch.randn(
            args.batch_size,
            config.generator.text_prefix_len,
            config.generator.d_text,
            device=device,
        )
        lengths = torch.linspace(
            args.window,
            max(config.tokenizer.downsample, args.window // 2),
            args.batch_size,
            device=device,
        ).long()

        first = trainer.train_step(motion, text, lengths)
        for step in range(args.steps):
            last = trainer.train_step(motion, text, lengths)
            if step % args.report_every == 0:
                print(
                    f"{backbone.value} step {step:3d}  ce {last['ce']:.4f}  "
                    f"total {last['total']:.4f}"
                )
        verdict = "PASS" if last["ce"] < 0.25 * first["ce"] else "FAIL"
        print(
            f"{backbone.value}: {parameter_count:,} params  ce {first['ce']:.3f} -> "
            f"{last['ce']:.3f}  [{verdict}]"
        )
        del generator, trainer
        if device == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Count and sanity-check the 100M twins.")
    parser.add_argument("--config", default="configs/generator/final100m_fsq8x1024.yaml")
    parser.add_argument("--tokenizer_ckpt", default="checkpoints/tokenizer/fsq_g8_v1024.pt")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--report_every", type=int, default=30)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
