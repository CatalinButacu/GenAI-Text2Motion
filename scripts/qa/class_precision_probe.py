# python scripts/qa/class_precision_probe.py --config configs/generator/a7/width_320_step50.yaml --backbone transformer --ckpt runs/checkpoints/generator/generator_transformer_w320_a7_step50.pt --out docs/paper/pics/class_precision_g4_transformer.csv
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.evaluation.contracts import EvaluationSample, GenerationEvaluationSet
from text2motion.evaluation.metrics import own_caption_ranks
from text2motion.evaluation.samples import chosen_caption_index
from text2motion.generation.contracts import Backbone, SamplingConfig
from text2motion.motion.contracts import Split

SEED = 2026
TOKENIZER = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
CLASS_MAP = Path("docs/paper/pics/verb_cluster_map.csv")
CLASS_COLUMN = "class_10"
SAMPLING = SamplingConfig(temperature=1.0, cfg_scale=6.0)
BATCH = 32
TOP_K = 3


def verb_classes(path: Path, column: str) -> dict[str, int]:
    with path.open(encoding="utf-8") as handle:
        return {row["verb"]: int(row[column]) for row in csv.DictReader(handle)}


def sample_classes(caption_tokens: list[list[str]], classes: dict[str, int]) -> frozenset[int]:
    found = set()
    for tokens in caption_tokens:
        for token in tokens:
            word, _, tag = token.rpartition("/")
            if tag == "VERB" and word.lower() in classes:
                found.add(classes[word.lower()])
    return frozenset(found)


def reference_set(samples: list[EvaluationSample], run: int) -> GenerationEvaluationSet:
    chosen = [chosen_caption_index(sample, run) for sample in samples]
    return GenerationEvaluationSet(
        sample_ids=[sample.sample_id for sample in samples],
        reference=[sample.reference for sample in samples],
        generated=[sample.reference for sample in samples],
        token_lists=[
            [sample.caption_tokens[index]] for sample, index in zip(samples, chosen, strict=True)
        ],
        captions=[sample.captions[index] for sample, index in zip(samples, chosen, strict=True)],
        requested=len(samples),
        dropped={},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--backbone", type=Backbone)
    parser.add_argument("--ckpt")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    if not arguments.reference and (arguments.backbone is None or arguments.ckpt is None):
        parser.error("a generator probe needs --backbone and --ckpt; real clips need --reference")
    app = ApplicationBootstrap.from_config(arguments.config, device=arguments.device, seed=SEED)
    evaluator = app.create_humanml3d_generation_evaluator()
    generator = None
    if not arguments.reference:
        generator = app.load_text_to_motion_generator(
            arguments.backbone,
            arguments.ckpt,
            app.load_tokenizer(TOKENIZER),
            tokenizer_checkpoint=TOKENIZER,
        ).eval()
    classes = verb_classes(CLASS_MAP, CLASS_COLUMN)
    samples, _ = evaluator.evaluation_samples(Split.VALIDATION, None)
    labels = {
        sample.sample_id: sample_classes(sample.caption_tokens, classes) for sample in samples
    }

    rng = np.random.default_rng(SEED)
    hits: dict[str, np.ndarray] = {}
    scored: dict[str, int] = defaultdict(int)
    sample_ids: list[str] = []
    for run in range(arguments.reps):
        split = (
            reference_set(samples, run)
            if generator is None
            else evaluator.generate_evaluation_set(
                generator, Split.VALIDATION, None, SAMPLING, BATCH, run=run
            )
        )
        if run and split.sample_ids != sample_ids:
            raise RuntimeError(f"generation run {run} scored a different sample set than run 0")
        sample_ids = split.sample_ids
        motion = evaluator.compute_motion_embeddings(split.generated)
        text, _ = evaluator.caption_embeddings(split)
        ranks = own_caption_ranks(text, motion, rng.permutation(len(text)))
        for sample_id, rank in zip(sample_ids, ranks, strict=True):
            hits.setdefault(sample_id, np.zeros(TOP_K))
            if rank >= 0:
                hits[sample_id][rank:] += 1
                scored[sample_id] += 1
        print(
            f"run {run + 1}/{arguments.reps}  R@1 {(ranks[ranks >= 0] == 0).mean():.3f}", flush=True
        )
    del generator
    torch.cuda.empty_cache()

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "classes", "r_top1", "r_top3"])
        for sample_id in sample_ids:
            if not scored[sample_id]:
                raise RuntimeError(f"{sample_id} fell outside every pool of 32")
            rates = hits[sample_id] / scored[sample_id]
            members = " ".join(str(label) for label in sorted(labels[sample_id]))
            writer.writerow([sample_id, members, f"{rates[0]:.4f}", f"{rates[2]:.4f}"])
    per_class: dict[int, list[float]] = defaultdict(list)
    for sample_id in sample_ids:
        for label in labels[sample_id]:
            per_class[label].append(hits[sample_id][0] / scored[sample_id])
    for label in sorted(per_class):
        print(f"class {label}: clips {len(per_class[label])}  R@1 {np.mean(per_class[label]):.3f}")


if __name__ == "__main__":
    main()
