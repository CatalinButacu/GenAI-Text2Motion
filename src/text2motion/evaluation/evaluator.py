from __future__ import annotations

import logging
import zlib
from collections import defaultdict

import numpy as np
import torch

from text2motion.evaluation.contracts import (
    DIVERSITY_PAIRS,
    GUO_MAX_FRAMES,
    GUO_MIN_FRAMES,
    R_PRECISION_POOL,
    EvaluationSample,
    GenerationEvaluationProtocol,
    GenerationEvaluationReport,
    GenerationEvaluationSet,
)
from text2motion.evaluation.metrics import (
    bootstrap_fid,
    compute_guo_motion_embeddings,
    compute_guo_text_embeddings,
    diversity,
    fid,
    mm_dist,
    r_precision,
    replication_summary,
)
from text2motion.evaluation.nn_matcher import GuoEvaluationResources
from text2motion.generation.contracts import SamplingConfig
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.motion.contracts import Split, TextAnnotation
from text2motion.motion.datasets import parse_text_file
from text2motion.motion.representation import FPS
from text2motion.motion.storage import FEATURES_DIR

logger = logging.getLogger(__name__)


def deterministic_generation_seed(clip_id: str, run: int = 0) -> int:
    return zlib.crc32(clip_id.encode("utf-8")) + run


def chosen_caption_index(sample: EvaluationSample, run: int) -> int:
    seed = deterministic_generation_seed(sample.sample_id, run)
    return int(np.random.default_rng(seed).integers(len(sample.captions)))


############################################################
#       HumanMl3dGenerationEvaluator
############################################################
class HumanMl3dGenerationEvaluator:
    def __init__(self, resources: GuoEvaluationResources) -> None:
        self.resources = resources

    def compute_motion_embeddings(self, feats: list[np.ndarray]) -> np.ndarray:
        return compute_guo_motion_embeddings(
            self.resources.motion_embedder,
            feats,
            self.resources.guo_motion_mean,
            self.resources.guo_motion_std,
            self.resources.device,
        )

    def calculate_reconstruction_fid(
        self, reference: list[np.ndarray], recon: list[np.ndarray]
    ) -> float:
        return fid(self.compute_motion_embeddings(reference), self.compute_motion_embeddings(recon))

    @staticmethod
    def __guo_pieces(
        clip_id: str, motion: np.ndarray, annotations: list[TextAnnotation]
    ) -> list[tuple[str, np.ndarray, list[TextAnnotation]]]:
        whole = [item for item in annotations if item.start_time == 0.0 and item.end_time == 0.0]
        tagged = [item for item in annotations if item not in whole]
        pieces = [(clip_id, motion, whole)] if whole else []
        for index, item in enumerate(tagged):
            segment = motion[int(item.start_time * FPS) : int(item.end_time * FPS)]
            pieces.append((f"{clip_id}#{index}", segment, [item]))
        return pieces

    def evaluation_samples(
        self, split: Split | str, max_clips: int | None
    ) -> tuple[list[EvaluationSample], dict[str, int]]:
        downsample = self.resources.tokenizer_downsample_factor
        dropped = {"missing_files": 0, "not_finite": 0, "out_of_length": 0, "no_caption": 0}
        samples: list[EvaluationSample] = []
        for clip_id in self.resources.clip_ids(split)[:max_clips]:
            text_path = self.resources.annotation_dir / f"{clip_id}.txt"
            vec_path = self.resources.motion_dataset_dir / FEATURES_DIR / f"{clip_id}.npy"
            if not text_path.is_file() or not vec_path.is_file():
                dropped["missing_files"] += 1
                continue
            motion = np.load(vec_path).astype(np.float32)
            if not np.isfinite(motion).all():
                dropped["not_finite"] += 1
                continue
            annotations = parse_text_file(text_path)
            for sample_id, piece, piece_annotations in self.__guo_pieces(
                clip_id, motion, annotations
            ):
                if not GUO_MIN_FRAMES <= piece.shape[0] < GUO_MAX_FRAMES:
                    dropped["out_of_length"] += 1
                    continue
                captioned = [item for item in piece_annotations if item.tokens]
                if not captioned:
                    dropped["no_caption"] += 1
                    continue
                token_len = piece.shape[0] // downsample
                samples.append(
                    EvaluationSample(
                        sample_id=sample_id,
                        reference=piece[: token_len * downsample],
                        token_len=token_len,
                        captions=[item.caption for item in captioned],
                        caption_tokens=[item.tokens for item in captioned],
                    )
                )
        return samples, dropped

    @torch.no_grad()
    def generate_evaluation_set(
        self,
        generator: TextToMotionGenerator,
        split: Split | str,
        max_clips: int | None,
        sampling: SamplingConfig,
        batch_size: int = 1,
        run: int = 0,
    ) -> GenerationEvaluationSet:
        generator.eval()
        samples, dropped = self.evaluation_samples(split, max_clips)
        chosen = [chosen_caption_index(sample, run) for sample in samples]
        generated: list[np.ndarray | None] = [None] * len(samples)

        buckets: dict[int, list[int]] = defaultdict(list)
        for position, sample in enumerate(samples):
            buckets[sample.token_len].append(position)

        for token_len, positions in buckets.items():
            for start in range(0, len(positions), max(1, batch_size)):
                group = positions[start : start + max(1, batch_size)]
                batch = generator.generate_batch(
                    [samples[i].captions[chosen[i]] for i in group],
                    token_len,
                    sampling,
                    seeds=[deterministic_generation_seed(samples[i].sample_id, run) for i in group],
                )
                if batch is None:
                    continue
                for position, item in zip(group, batch, strict=True):
                    generated[position] = np.asarray(
                        self.resources.denormalize(item.motion.features.cpu().numpy())
                    )

        scorable = [index for index, motion in enumerate(generated) if motion is not None]
        dropped["empty_generation"] = len(samples) - len(scorable)
        if not scorable:
            raise RuntimeError(f"no scorable samples in split {split!r}; dropped {dropped}")

        return GenerationEvaluationSet(
            sample_ids=[samples[i].sample_id for i in scorable],
            reference=[samples[i].reference for i in scorable],
            generated=[motion for motion in generated if motion is not None],
            token_lists=[[samples[i].caption_tokens[chosen[i]]] for i in scorable],
            captions=[samples[i].captions[chosen[i]] for i in scorable],
            requested=len(samples),
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
                np.asarray(self.resources.denormalize(item.motion.features.cpu().numpy()))
                for item in batch
            ]
            emb = self.compute_motion_embeddings(decoded)
            first = rng.integers(0, len(emb), 10)
            second = rng.integers(0, len(emb), 10)
            per_caption.append(float(np.linalg.norm(emb[first] - emb[second], axis=1).mean()))
        return float(np.mean(per_caption)) if per_caption else float("nan")

    def caption_embeddings(
        self, split: GenerationEvaluationSet
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        flat, owner = [], []
        for clip_index, token_lists in enumerate(split.token_lists):
            for tokens in token_lists:
                flat.append(self.resources.build_guo_text_inputs(tokens))
                owner.append(clip_index)
        text_emb = compute_guo_text_embeddings(
            self.resources.text_embedder, flat, self.resources.device
        )
        owner = np.array(owner)
        per_clip = [np.where(owner == index)[0] for index in range(len(split.generated))]
        return text_emb, per_clip

    def calculate_run_metrics(
        self,
        reference_emb: np.ndarray,
        generated_emb: np.ndarray,
        caption_emb: np.ndarray,
        per_clip: list[np.ndarray],
        rng: np.random.Generator,
    ) -> dict[str, float]:
        chosen = np.array([rng.choice(indices) for indices in per_clip])
        order = rng.permutation(len(generated_emb))
        selected = caption_emb[chosen][order]
        motions = generated_emb[order]
        top1, top2, top3 = r_precision(selected, motions, pool_size=R_PRECISION_POOL, top_k=3)
        return {
            "fid": fid(reference_emb, generated_emb),
            "r_top1": float(top1),
            "r_top2": float(top2),
            "r_top3": float(top3),
            "mm_dist": mm_dist(selected, motions),
            "diversity": diversity(generated_emb, num_pairs=DIVERSITY_PAIRS),
        }

    def run_generation_evaluation_protocol(
        self, generator: TextToMotionGenerator, request: GenerationEvaluationProtocol | None = None
    ) -> GenerationEvaluationReport:
        request = request or GenerationEvaluationProtocol()
        if request.reps < 1:
            raise ValueError(f"the protocol needs at least 1 generation run, got {request.reps}")
        split = self.generate_evaluation_set(
            generator,
            request.split,
            request.max_clips,
            request.sampling,
            request.batch_size,
        )
        if sum(split.dropped.values()):
            logger.warning(
                f"scored {len(split.generated)} of {split.requested} requested clips "
                f"(dropped {split.dropped}). FID is only comparable at equal clip counts."
            )

        reference_emb = self.compute_motion_embeddings(split.reference)
        first_emb = self.compute_motion_embeddings(split.generated)
        caption_emb, per_clip = self.caption_embeddings(split)
        rng = np.random.default_rng(0)
        runs = [self.calculate_run_metrics(reference_emb, first_emb, caption_emb, per_clip, rng)]
        for run in range(1, request.reps):
            repeated = self.generate_evaluation_set(
                generator,
                request.split,
                request.max_clips,
                request.sampling,
                request.batch_size,
                run=run,
            )
            if repeated.sample_ids != split.sample_ids:
                raise RuntimeError(f"generation run {run} scored a different sample set than run 0")
            repeated_emb = self.compute_motion_embeddings(repeated.generated)
            run_caption_emb, run_per_clip = self.caption_embeddings(repeated)
            runs.append(
                self.calculate_run_metrics(
                    reference_emb, repeated_emb, run_caption_emb, run_per_clip, rng
                )
            )
            logger.info(f"generation run {run + 1}/{request.reps}  FID {runs[-1]['fid']:.4f}")
        summary = {name: replication_summary([run[name] for run in runs]) for name in runs[0]}
        bootstrap = (
            bootstrap_fid(reference_emb, first_emb, resamples=request.bootstrap)
            if request.bootstrap > 0
            else {}
        )

        multimodality = float("nan")
        if request.mm_clips > 0:
            token_lens = [
                feat.shape[0] // self.resources.tokenizer_downsample_factor
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
            clips=len(split.generated),
            runs=request.reps,
            fid=summary["fid"][0],
            r_precision=(summary["r_top1"][0], summary["r_top2"][0], summary["r_top3"][0]),
            diversity=summary["diversity"][0],
            multimodality=multimodality,
            matching_distance=summary["mm_dist"][0],
            fid_ci95=summary["fid"][1],
            r_precision_ci95=(summary["r_top1"][1], summary["r_top2"][1], summary["r_top3"][1]),
            diversity_ci95=summary["diversity"][1],
            matching_distance_ci95=summary["mm_dist"][1],
            fid_bootstrap_lo=bootstrap.get("fid_ci_lo", float("nan")),
            fid_bootstrap_hi=bootstrap.get("fid_ci_hi", float("nan")),
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
            logger.info(
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
