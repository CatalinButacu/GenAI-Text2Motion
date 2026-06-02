from __future__ import annotations

import argparse
import logging
import random

import numpy as np
import torch

from src.modules.render.config import RenderConfig
from src.modules.runtime import StreamCapabilityError
from src.pipeline import Pipeline
from src.shared.config import PipelineConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text-to-Motion (file / viewer / chat)")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    p.add_argument("prompt", nargs="?", default="a person walks forward")
    p.add_argument("--mode", choices=["file", "viewer", "chat", "stream"], default="file")
    p.add_argument("--name", dest="output_name", default="output")
    p.add_argument("--output-dir", dest="output_dir", default="outputs")
    p.add_argument("--fps", type=int, default=PipelineConfig.fps)
    p.add_argument(
        "--max-duration", dest="max_duration", type=float, default=PipelineConfig.duration
    )
    p.add_argument("--device", default=default_device, choices=["cuda", "cpu"])

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", dest="top_p", type=float, default=1.0)
    p.add_argument("--cfg-scale", dest="cfg_scale", type=float, default=1.0)

    p.add_argument(
        "--rerank",
        action="store_true",
        help="Generate --num-candidates motion candidates and select the best via SBERT reranking",
    )
    p.add_argument(
        "--num-candidates",
        dest="num_candidates",
        type=int,
        default=4,
        help="Number of candidates to generate when --rerank is set (default: 4)",
    )
    p.add_argument(
        "--use-retrieval",
        dest="use_retrieval",
        action="store_true",
        help="Augment prompts with top-K retrieved training examples before generation",
    )
    p.add_argument(
        "--retrieval-index",
        dest="retrieval_index",
        default="data/retrieval_index.npz",
        help="Path to retrieval index .npz (build with scripts/data/build_retrieval_index.py)",
    )

    p.add_argument("--ssm-checkpoint", dest="ssm_checkpoint", default=None)
    p.add_argument("--rvq-checkpoint", dest="rvq_checkpoint", default=None)

    p.add_argument("--gender", choices=["neutral", "male", "female"], default=RenderConfig.gender)

    p.add_argument("--seed", type=int, default=None, help="Reproducibility")
    p.add_argument(
        "--max-stream-chunks",
        dest="max_stream_chunks",
        type=int,
        default=None,
        help="Optional cap for streaming validation chunks",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Parse + plan only; skip motion + render",
    )

    verbosity = p.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true")
    verbosity.add_argument("-q", "--quiet", action="store_true")

    return p.parse_args()


def setup_logging(verbose: bool, quiet: bool) -> None:
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_summary(result: dict | None) -> None:
    if not result:
        return

    parsed = result.get("parsed_scene")

    if parsed:
        print(f"entities: {[e.name for e in parsed.entities]}")
        print(f"actions : {[a.action_type for a in parsed.actions]}")


def build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig(
        output_dir=args.output_dir,
        duration=args.max_duration,
        fps=args.fps,
        device=args.device,
    )
    cfg.motion.temperature = args.temperature
    cfg.motion.top_p = args.top_p
    cfg.motion.cfg_scale = args.cfg_scale
    cfg.motion.rerank = args.rerank
    cfg.motion.num_candidates = args.num_candidates
    cfg.motion.use_retrieval = args.use_retrieval
    cfg.motion.retrieval_index_path = args.retrieval_index

    if args.ssm_checkpoint:
        cfg.motion.checkpoint_path = args.ssm_checkpoint

    if args.rvq_checkpoint:
        cfg.motion.rvq_checkpoint_path = args.rvq_checkpoint

    cfg.render.gender = args.gender

    if args.seed is not None:
        cfg.planner.random_seed = args.seed

    return cfg


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose, args.quiet)

    if args.seed is not None:
        seed_everything(args.seed)

    pipe = Pipeline(build_config(args))

    if args.dry_run:
        result = pipe.parse_and_plan(args.prompt)
        print_summary(result)
        return

    if args.mode == "chat":
        pipe.run_chat()
        return

    if args.mode == "viewer":
        pipe.run_viewer(args.prompt)
        return

    if args.mode == "stream":
        try:
            summary = pipe.validate_streaming(
                args.prompt,
                max_stream_chunks=args.max_stream_chunks,
            )
        except StreamCapabilityError as e:
            print(f"error: {e}")
            return

        print("\nstream summary")

        for key, value in summary.items():
            print(f"  {key}: {value}")

        return

    result = pipe.render_to_file(args.prompt, output_name=args.output_name)

    if result is None:
        return

    print(f"\nvideo -> {result.get('video_path', '')}")
    print_summary(result)


if __name__ == "__main__":
    main()
