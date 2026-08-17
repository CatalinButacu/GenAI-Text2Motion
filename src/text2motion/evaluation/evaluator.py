from __future__ import annotations

import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from text2motion.evaluation.matcher import EvaluationContext
from text2motion.evaluation.metrics import (
    bootstrap_fid,
    diversity,
    embed_motions,
    embed_texts,
    fid,
    mm_dist,
    r_precision,
)
from text2motion.generation.pipeline import MotionGenerator, SamplingConfig
from text2motion.motion.dataset import FEATURES_DIR, Split, parse_text_file

MAX_EVAL_FRAMES = 196
MIN_EVAL_TOKENS = 2
R_PRECISION_POOL = 32


def clip_seed(clip_id: str) -> int:
    return zlib.crc32(clip_id.encode("utf-8"))


@dataclass(frozen=True)
class EvaluationRequest:
    split: Split = Split.TEST
    max_clips: int = 100000
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    mm_clips: int = 100
    mm_repeats: int = 30
    bootstrap: int = 200
    reps: int = 20
    batch_size: int = 1


@dataclass(frozen=True)
class EvaluationReport:
    clips: int
    fid: float
    r_precision: tuple[float, float, float]
    diversity: float
    multimodality: float
    matching_distance: float
    r_top1_std: float = 0.0
    fid_ci_lo: float = float("nan")
    fid_ci_hi: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        record = asdict(self)
        top1, top2, top3 = self.r_precision
        record.pop("r_precision")
        record.update({"r_top1": top1, "r_top2": top2, "r_top3": top3})
        return record


@dataclass(frozen=True)
class GeneratedSplit:
    reference: list[np.ndarray]
    generated: list[np.ndarray]
    token_lists: list[list[list[str]]]
    captions: list[str]
    requested: int
    dropped: dict[str, int]


class MotionEvaluator:
    def __init__(self, context: EvaluationContext) -> None:
        self.context = context

    def embed(self, feats: list[np.ndarray]) -> np.ndarray:
        return embed_motions(
            self.context.motion_matcher,
            feats,
            self.context.eval_mean,
            self.context.eval_std,
            self.context.device,
        )

    def recon_fid(self, reference: list[np.ndarray], recon: list[np.ndarray]) -> float:
        return fid(self.embed(reference), self.embed(recon))

    def reference_clip(self, clip_id: str) -> tuple[np.ndarray, int] | None:
        vec_path = self.context.out_dir / FEATURES_DIR / f"{clip_id}.npy"
        if not vec_path.is_file():
            return None
        feat = np.load(vec_path).astype(np.float32)
        if np.isnan(feat).any():
            return None
        token_len = min(feat.shape[0], MAX_EVAL_FRAMES) // self.context.downsample
        if token_len < MIN_EVAL_TOKENS:
            return None
        return feat[: token_len * self.context.downsample], token_len

    @torch.no_grad()
    def generate_split(
        self,
        generator: MotionGenerator,
        split: Split | str,
        max_clips: int | None,
        sampling: SamplingConfig,
        batch_size: int = 1,
    ) -> GeneratedSplit:
        generator.eval()
        ids = self.context.clip_ids(split)[:max_clips]
        dropped = {"missing_files": 0, "too_short": 0, "no_caption": 0, "empty_generation": 0}

        entries: list[dict] = []
        for clip_id in ids:
            text_path = self.context.text_dir / f"{clip_id}.txt"
            if not text_path.is_file():
                dropped["missing_files"] += 1
                continue
            clip = self.reference_clip(clip_id)
            if clip is None:
                dropped["too_short"] += 1
                continue
            feat, token_len = clip

            annotations = parse_text_file(text_path)
            caption_tokens = [item.tokens for item in annotations if item.tokens]
            if not caption_tokens:
                dropped["no_caption"] += 1
                continue

            entries.append(
                {
                    "clip_id": clip_id,
                    "feat": feat,
                    "token_len": token_len,
                    "caption": annotations[0].caption,
                    "caption_tokens": caption_tokens,
                    "generated": None,
                }
            )

        buckets: dict[int, list[int]] = defaultdict(list)
        for position, entry in enumerate(entries):
            buckets[entry["token_len"]].append(position)

        for token_len, positions in buckets.items():
            for start in range(0, len(positions), max(1, batch_size)):
                group = positions[start : start + max(1, batch_size)]
                batch = generator.generate_batch(
                    [entries[i]["caption"] for i in group],
                    token_len,
                    sampling,
                    seeds=[clip_seed(entries[i]["clip_id"]) for i in group],
                )
                if batch is None:
                    continue
                for position, item in zip(group, batch, strict=True):
                    entries[position]["generated"] = self.context.denormalize(
                        item.motion.features.cpu().numpy()
                    )

        scorable = [entry for entry in entries if entry["generated"] is not None]
        dropped["empty_generation"] += len(entries) - len(scorable)
        if not scorable:
            raise RuntimeError(f"no scorable clips in split {split!r}; dropped {dropped}")

        return GeneratedSplit(
            reference=[entry["feat"] for entry in scorable],
            generated=[entry["generated"] for entry in scorable],
            token_lists=[entry["caption_tokens"] for entry in scorable],
            captions=[entry["caption"] for entry in scorable],
            requested=len(ids),
            dropped=dropped,
        )

    @torch.no_grad()
    def multimodality(
        self,
        generator: MotionGenerator,
        captions: list[str],
        token_lens: list[int],
        sampling: SamplingConfig,
        mm_clips: int,
        mm_repeats: int,
    ) -> float:
        rng = np.random.default_rng(0)
        per_caption = []
        for caption, token_len in zip(captions[:mm_clips], token_lens[:mm_clips], strict=False):
            batch = generator.generate_batch([caption] * mm_repeats, token_len, sampling)
            if batch is None or len(batch) < 2:
                continue
            decoded = [
                self.context.denormalize(item.motion.features.cpu().numpy()) for item in batch
            ]
            emb = self.embed(decoded)
            first = rng.integers(0, len(emb), 10)
            second = rng.integers(0, len(emb), 10)
            per_caption.append(float(np.linalg.norm(emb[first] - emb[second], axis=1).mean()))
        return float(np.mean(per_caption)) if per_caption else float("nan")

    def score(self, split: GeneratedSplit, reps: int, bootstrap: int) -> dict[str, float]:
        reference_emb = self.embed(split.reference)
        generated_emb = self.embed(split.generated)

        flat, owner = [], []
        for clip_index, token_lists in enumerate(split.token_lists):
            for tokens in token_lists:
                flat.append(self.context.build_text(tokens))
                owner.append(clip_index)
        text_emb = embed_texts(self.context.text_matcher, flat, self.context.device)
        owner = np.array(owner)
        per_clip = [np.where(owner == index)[0] for index in range(len(split.generated))]

        rng = np.random.default_rng(0)
        precisions, distances = [], []
        for _ in range(reps):
            chosen = np.array([rng.choice(indices) for indices in per_clip])
            selected = text_emb[chosen]
            perm = rng.permutation(len(generated_emb))
            precisions.append(
                r_precision(
                    selected[perm], generated_emb[perm], pool_size=R_PRECISION_POOL, top_k=3
                )
            )
            distances.append(mm_dist(selected[perm], generated_emb[perm]))
        precisions = np.stack(precisions)

        interval = (
            bootstrap_fid(reference_emb, generated_emb, resamples=bootstrap)
            if bootstrap > 0
            else {}
        )
        return {
            "clips": len(split.generated),
            "fid": fid(reference_emb, generated_emb),
            **interval,
            "r_top1": float(precisions[:, 0].mean()),
            "r_top1_std": float(precisions[:, 0].std()),
            "r_top2": float(precisions[:, 1].mean()),
            "r_top3": float(precisions[:, 2].mean()),
            "mm_dist": float(np.mean(distances)),
            "diversity": diversity(generated_emb, num_pairs=300),
        }

    def evaluate(
        self, generator: MotionGenerator, request: EvaluationRequest | None = None
    ) -> EvaluationReport:
        request = request or EvaluationRequest()
        split = self.generate_split(generator, request.split, request.max_clips, request.sampling)
        if sum(split.dropped.values()):
            print(
                f"WARNING: scored {len(split.generated)} of {split.requested} requested clips "
                f"(dropped {split.dropped}). FID is only comparable at equal clip counts."
            )

        metrics = self.score(split, reps=request.reps, bootstrap=request.bootstrap)

        multimodality = float("nan")
        if request.mm_clips > 0:
            token_lens = [
                min(feat.shape[0], MAX_EVAL_FRAMES) // self.context.downsample
                for feat in split.reference
            ]
            multimodality = self.multimodality(
                generator,
                split.captions,
                token_lens,
                request.sampling,
                request.mm_clips,
                request.mm_repeats,
            )

        return EvaluationReport(
            clips=metrics["clips"],
            fid=metrics["fid"],
            r_precision=(metrics["r_top1"], metrics["r_top2"], metrics["r_top3"]),
            diversity=metrics["diversity"],
            multimodality=multimodality,
            matching_distance=metrics["mm_dist"],
            r_top1_std=metrics["r_top1_std"],
            fid_ci_lo=metrics.get("fid_ci_lo", float("nan")),
            fid_ci_hi=metrics.get("fid_ci_hi", float("nan")),
        )

    @torch.no_grad()
    def evaluate_generation(
        self,
        generator: MotionGenerator,
        split: Split | str = Split.VALIDATION,
        max_clips: int | None = None,
        sampling: SamplingConfig | None = None,
        batch_size: int = 1,
    ) -> dict[str, float]:
        sampling = sampling or SamplingConfig()
        generated = self.generate_split(generator, split, max_clips, sampling, batch_size)
        if sum(generated.dropped.values()):
            print(
                f"  [gen-eval] scored {len(generated.generated)}/{generated.requested} clips, "
                f"dropped {generated.dropped}"
            )

        reference_emb = self.embed(generated.reference)
        generated_emb = self.embed(generated.generated)
        text_emb = embed_texts(
            self.context.text_matcher,
            [self.context.build_text(tokens[0]) for tokens in generated.token_lists],
            self.context.device,
        )

        rng = np.random.default_rng(0)
        perm = rng.permutation(len(generated_emb))
        precision = r_precision(
            text_emb[perm], generated_emb[perm], pool_size=R_PRECISION_POOL, top_k=3
        )
        return {
            "clips": len(generated.generated),
            "requested_clips": generated.requested,
            "dropped": sum(generated.dropped.values()),
            "fid": fid(reference_emb, generated_emb),
            "r_top1": float(precision[0]),
            "r_top3": float(precision[2]),
            "mm_dist": mm_dist(text_emb, generated_emb),
            "diversity": diversity(generated_emb),
        }
