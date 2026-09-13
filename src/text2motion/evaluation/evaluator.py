from __future__ import annotations

import zlib
from collections import defaultdict

import numpy as np
import torch

from text2motion.evaluation.contracts import (
    DIVERSITY_PAIRS,
    R_PRECISION_POOL,
    GenerationEvaluationProtocol,
    GenerationEvaluationReport,
    GenerationEvaluationSet,
)
from text2motion.evaluation.matcher import GuoEvaluationResources
from text2motion.evaluation.metrics import (
    bootstrap_fid,
    compute_guo_motion_embeddings,
    compute_guo_text_embeddings,
    diversity,
    fid,
    mm_dist,
    r_precision,
)
from text2motion.generation.contracts import SamplingConfig
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.motion.annotations import parse_text_file
from text2motion.motion.contracts import Split
from text2motion.motion.storage import FEATURES_DIR


def deterministic_generation_seed(clip_id: str) -> int:
    return zlib.crc32(clip_id.encode("utf-8"))


class HumanMl3dGenerationEvaluator:
    def __init__(self, resources: GuoEvaluationResources, max_frames: int, min_tokens: int = 2) -> None:
        self.resources = resources
        self.max_frames = max_frames
        self.min_tokens = min_tokens

    def compute_motion_embeddings(self, feats: list[np.ndarray]) -> np.ndarray:
        return compute_guo_motion_embeddings(
            self.resources.motion_embedder,
            feats,
            self.resources.guo_motion_mean,
            self.resources.guo_motion_std,
            self.resources.device,
        )

    def calculate_reconstruction_fid(self, reference: list[np.ndarray], recon: list[np.ndarray]) -> float:
        return fid(self.compute_motion_embeddings(reference), self.compute_motion_embeddings(recon))

    def load_scorable_reference_clip(self, clip_id: str) -> tuple[np.ndarray, int] | None:
        vec_path = self.resources.motion_dataset_dir / FEATURES_DIR / f"{clip_id}.npy"
        if not vec_path.is_file():
            return None
        feat = np.load(vec_path).astype(np.float32)
        if np.isnan(feat).any():
            return None
        token_len = min(feat.shape[0], self.max_frames) // self.resources.tokenizer_downsample_factor
        if token_len < self.min_tokens:
            return None
        return feat[: token_len * self.resources.tokenizer_downsample_factor], token_len

    @torch.no_grad()
    def generate_evaluation_set(
        self,
        generator: TextToMotionGenerator,
        split: Split | str,
        max_clips: int | None,
        sampling: SamplingConfig,
        batch_size: int = 1,
    ) -> GenerationEvaluationSet:
        generator.eval()
        ids = self.resources.clip_ids(split)[:max_clips]
        dropped = {"missing_files": 0, "too_short": 0, "no_caption": 0, "empty_generation": 0}

        entries: list[dict] = []
        for clip_id in ids:
            text_path = self.resources.annotation_dir / f"{clip_id}.txt"
            if not text_path.is_file():
                dropped["missing_files"] += 1
                continue
            clip = self.load_scorable_reference_clip(clip_id)
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
                    seeds=[deterministic_generation_seed(entries[i]["clip_id"]) for i in group],
                )
                if batch is None:
                    continue
                for position, item in zip(group, batch, strict=True):
                    entries[position]["generated"] = self.resources.denormalize(
                        item.motion.features.cpu().numpy()
                    )

        scorable = [entry for entry in entries if entry["generated"] is not None]
        dropped["empty_generation"] += len(entries) - len(scorable)
        if not scorable:
            raise RuntimeError(f"no scorable clips in split {split!r}; dropped {dropped}")

        return GenerationEvaluationSet(
            reference=[entry["feat"] for entry in scorable],
            generated=[entry["generated"] for entry in scorable],
            token_lists=[entry["caption_tokens"] for entry in scorable],
            captions=[entry["caption"] for entry in scorable],
            requested=len(ids),
            dropped=dropped,
        )

    @torch.no_grad()
    def calculate_multimodality(
        self,
        generator: TextToMotionGenerator,
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
                self.resources.denormalize(item.motion.features.cpu().numpy()) for item in batch
            ]
            emb = self.compute_motion_embeddings(decoded)
            first = rng.integers(0, len(emb), 10)
            second = rng.integers(0, len(emb), 10)
            per_caption.append(float(np.linalg.norm(emb[first] - emb[second], axis=1).mean()))
        return float(np.mean(per_caption)) if per_caption else float("nan")

    def calculate_generation_metrics(self, split: GenerationEvaluationSet, reps: int, bootstrap: int) -> dict[str, float]:
        reference_emb = self.compute_motion_embeddings(split.reference)
        generated_emb = self.compute_motion_embeddings(split.generated)

        flat, owner = [], []
        for clip_index, token_lists in enumerate(split.token_lists):
            for tokens in token_lists:
                flat.append(self.resources.build_guo_text_inputs(tokens))
                owner.append(clip_index)
        text_emb = compute_guo_text_embeddings(self.resources.text_embedder, flat, self.resources.device)
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
            "diversity": diversity(generated_emb, num_pairs=DIVERSITY_PAIRS),
        }

    def run_generation_evaluation_protocol(
        self, generator: TextToMotionGenerator, request: GenerationEvaluationProtocol | None = None
    ) -> GenerationEvaluationReport:
        request = request or GenerationEvaluationProtocol()
        split = self.generate_evaluation_set(
            generator,
            request.split,
            request.max_clips,
            request.sampling,
            request.batch_size,
        )
        if sum(split.dropped.values()):
            print(
                f"WARNING: scored {len(split.generated)} of {split.requested} requested clips "
                f"(dropped {split.dropped}). FID is only comparable at equal clip counts."
            )

        metrics = self.calculate_generation_metrics(split, reps=request.reps, bootstrap=request.bootstrap)

        multimodality = float("nan")
        if request.mm_clips > 0:
            token_lens = [
                min(feat.shape[0], self.max_frames) // self.resources.tokenizer_downsample_factor
                for feat in split.reference
            ]
            multimodality = self.calculate_multimodality(
                generator,
                split.captions,
                token_lens,
                request.sampling,
                request.mm_clips,
                request.mm_repeats,
            )

        return GenerationEvaluationReport(
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
    def evaluate_checkpoint_selection_metrics(
        self,
        generator: TextToMotionGenerator,
        split: Split | str = Split.VALIDATION,
        max_clips: int | None = None,
        sampling: SamplingConfig | None = None,
        batch_size: int = 1,
    ) -> dict[str, float]:
        sampling = sampling or SamplingConfig()
        generated = self.generate_evaluation_set(generator, split, max_clips, sampling, batch_size)
        if sum(generated.dropped.values()):
            print(
                f"  [gen-eval] scored {len(generated.generated)}/{generated.requested} clips, "
                f"dropped {generated.dropped}"
            )

        reference_emb = self.compute_motion_embeddings(generated.reference)
        generated_emb = self.compute_motion_embeddings(generated.generated)
        text_emb = compute_guo_text_embeddings(
            self.resources.text_embedder,
            [self.resources.build_guo_text_inputs(tokens[0]) for tokens in generated.token_lists],
            self.resources.device,
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
