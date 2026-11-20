# python scripts/qa/verb_generation_probe.py --verbs 100 --samples 32 --out docs/paper/pics/verb_generation_validation_32.csv
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.evaluation.embeddings import compute_guo_text_embeddings
from text2motion.generation.contracts import Backbone, SamplingConfig

SEED = 2026
CONFIG = "configs/generator/size_sweep/width_320.yaml"
TOKENIZER = "runs/checkpoints/tokenizer/fsq_g8_v1024.pt"
GENERATORS = (
    (Backbone.TRANSFORMER, "runs/checkpoints/generator/generator_transformer_w320_p1.pt"),
    (Backbone.MAMBA, "runs/checkpoints/generator/generator_mamba_w320_p1.pt"),
)
INVENTORY = Path("docs/notes/verb_inventory.csv")
ROOT = Path("data/processed/humanml3d")
NOT_MOTION = frozenset(
    {
        "left",
        "side",
        "back",
        "clockwise",
        "head",
        "outstretche",
        "swinge",
        "take",
        "put",
        "make",
        "get",
        "go",
        "come",
        "set",
        "keep",
        "repeat",
        "perform",
        "act",
        "return",
        "proceed",
        "follow",
        "let",
        "give",
        "avoid",
        "pretend",
        "alternate",
        "check",
        "play",
        "end",
        "face",
        "warm",
    }
)
SAMPLES = 16
TOKEN_STEPS = 30
MAX_FRAMES = 196
MIN_FRAMES = 40
TOP_K = 5
SAMPLING = SamplingConfig(temperature=1.0, top_p=0.9, cfg_scale=6.0)


def third_person(verb: str) -> str:
    if verb.endswith(("s", "sh", "ch", "x", "z", "o")):
        return f"{verb}es"
    if verb.endswith("y") and verb[-2:-1] not in "aeiou":
        return f"{verb[:-1]}ies"
    return f"{verb}s"


def chosen_verbs(count: int) -> list[tuple[str, int]]:
    rows = list(csv.DictReader(INVENTORY.open(encoding="utf-8")))
    verbs = [
        (row["verb"], int(row["train_clips"])) for row in rows if row["verb"] not in NOT_MOTION
    ]
    if len(verbs) < count:
        raise ValueError(f"only {len(verbs)} motion verbs in {INVENTORY}")
    return verbs[:count]


def real_clips(verbs: list[str], downsample: int, samples: int) -> dict[str, list[np.ndarray]]:
    wanted = set(verbs)
    found: dict[str, list[str]] = {verb: [] for verb in verbs}
    clip_ids = [
        c for c in (ROOT / "train.txt").read_text(encoding="utf-8").split() if not c.startswith("M")
    ]
    random.Random(SEED).shuffle(clip_ids)
    for clip_id in clip_ids:
        path = ROOT / "texts" / f"{clip_id}.txt"
        if not path.exists():
            continue
        tags = {
            token.rpartition("/")[0]
            for line in path.read_text(encoding="utf-8").splitlines()
            if "#" in line
            for token in line.split("#")[1].split()
            if token.endswith("/VERB")
        }
        for verb in tags & wanted:
            if len(found[verb]) < samples:
                found[verb].append(clip_id)
    motions: dict[str, list[np.ndarray]] = {}
    for verb, ids in found.items():
        motions[verb] = []
        for clip_id in ids:
            raw = np.load(ROOT / "new_joint_vecs" / f"{clip_id}.npy").astype(np.float32)
            frames = (min(len(raw), MAX_FRAMES) // downsample) * downsample
            if frames >= MIN_FRAMES and np.isfinite(raw).all():
                motions[verb].append(raw[:frames])
    return motions


def retrieval(motion_emb: np.ndarray, text_emb: np.ndarray, own: int) -> tuple[float, float]:
    distances = np.linalg.norm(motion_emb[:, None, :] - text_emb[None, :, :], axis=-1)
    ranks = (distances < distances[:, own : own + 1]).sum(axis=1)
    return float((ranks == 0).mean()), float((ranks < TOP_K).mean())


def top5_interval(rate: float, count: int) -> tuple[float, float]:
    if count == 0:
        raise ValueError("a top-5 interval needs at least one motion")
    interval = stats.binomtest(round(rate * count), count).proportion_ci(method="wilson")
    return float(interval.low), float(interval.high)


def generated_motions(pipeline, resources, verb: str, samples: int) -> list[np.ndarray]:
    torch.manual_seed(SEED)
    with torch.no_grad():
        batch = pipeline.generate_batch(
            [f"a person {third_person(verb)}"] * samples, TOKEN_STEPS, SAMPLING
        )
    if batch is None:
        raise RuntimeError(f"the generator returned no clip for {verb!r}")
    return [resources.denormalize(item.motion.features.cpu().numpy()) for item in batch]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbs", type=int, default=100)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--transformer_ckpt", default=dict(GENERATORS)[Backbone.TRANSFORMER])
    parser.add_argument("--mamba_ckpt", default=dict(GENERATORS)[Backbone.MAMBA])
    parser.add_argument(
        "--out", type=Path, default=Path("docs/paper/pics/verb_generation_validation.csv")
    )
    arguments = parser.parse_args()

    app = ApplicationBootstrap.from_config(CONFIG, seed=SEED)
    tokenizer = app.load_tokenizer(TOKENIZER)
    evaluator = app.create_humanml3d_generation_evaluator()
    resources = evaluator.resources
    verbs = chosen_verbs(arguments.verbs)
    names = [verb for verb, _ in verbs]
    text_emb = compute_guo_text_embeddings(
        resources.text_embedder,
        [
            resources.build_guo_text_inputs(["a/DET", "person/NOUN", f"{verb}/VERB"])
            for verb in names
        ],
        resources.device,
    )
    rows = {verb: {"verb": verb, "train_clips": clips} for verb, clips in verbs}
    real = real_clips(names, app.config.tokenizer.downsample, arguments.samples)
    for own, verb in enumerate(names):
        rows[verb]["real_n"] = len(real[verb])
        top1, top5 = retrieval(evaluator.compute_motion_embeddings(real[verb]), text_emb, own)
        rows[verb]["real_top1"], rows[verb]["real_top5"] = top1, top5
        rows[verb]["real_top5_low"], rows[verb]["real_top5_high"] = top5_interval(
            top5, len(real[verb])
        )
    checkpoints = {
        Backbone.TRANSFORMER: arguments.transformer_ckpt,
        Backbone.MAMBA: arguments.mamba_ckpt,
    }
    for backbone, checkpoint in checkpoints.items():
        pipeline = app.load_text_to_motion_generator(
            backbone, checkpoint, tokenizer, tokenizer_checkpoint=TOKENIZER
        ).eval()
        for own, verb in enumerate(names):
            motions = generated_motions(pipeline, resources, verb, arguments.samples)
            embeddings = evaluator.compute_motion_embeddings(motions)
            top1, top5 = retrieval(embeddings, text_emb, own)
            rows[verb][f"{backbone.value}_top1"], rows[verb][f"{backbone.value}_top5"] = top1, top5
            low, high = top5_interval(top5, len(motions))
            rows[verb][f"{backbone.value}_top5_low"], rows[verb][f"{backbone.value}_top5_high"] = (
                low,
                high,
            )
            print(verb, backbone.value, f"top5 {top5:.3f}", flush=True)
        del pipeline
        torch.cuda.empty_cache()

    table = list(rows.values())
    clips = np.log([row["train_clips"] for row in table])
    for column in ("real_top5", "transformer_top5", "mamba_top5"):
        rho, p_value = stats.spearmanr(clips, [row[column] for row in table])
        print(f"spearman log(train clips) vs {column}: rho {rho:.3f} p {p_value:.4f}", flush=True)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
