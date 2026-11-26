from __future__ import annotations

import logging

import numpy as np
import torch

from text2motion.evaluation.contracts import (
    EvaluationSample,
    GenerationEvaluationProtocol,
    GenerationEvaluationReport,
    GenerationEvaluationSet,
)
from text2motion.evaluation.embeddings import (
    compute_guo_motion_embeddings,
    compute_guo_text_embeddings,
)
from text2motion.evaluation.generated_set import collect_scorable_set, generate_sample_motions
from text2motion.evaluation.metrics import bootstrap_fid, fid
from text2motion.evaluation.nn_matcher import GuoEvaluationResources
from text2motion.evaluation.reports import (
    build_protocol_report,
    calculate_run_metrics,
    checkpoint_selection_record,
)
from text2motion.evaluation.samples import chosen_caption_index, read_evaluation_samples
from text2motion.generation.contracts import SamplingConfig
from text2motion.generation.pipeline import TextToMotionGenerator
from text2motion.motion.contracts import Split

logger = logging.getLogger(__name__)


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

    def evaluation_samples(
        self, split: Split | str, max_clips: int | None, clip_list: tuple[str, ...] | None = None
    ) -> tuple[list[EvaluationSample], dict[str, int]]:
        return read_evaluation_samples(self.resources, split, max_clips, clip_list)

    @torch.no_grad()
    def generate_evaluation_set(
        self,
        generator: TextToMotionGenerator,
        split: Split | str,
        max_clips: int | None,
        sampling: SamplingConfig,
        batch_size: int = 1,
        run: int = 0,
        clip_list: tuple[str, ...] | None = None,
    ) -> GenerationEvaluationSet:
        generator.eval()
        samples, dropped = self.evaluation_samples(split, max_clips, clip_list)
        chosen = [chosen_caption_index(sample, run) for sample in samples]
        generated = generate_sample_motions(
            self.resources, generator, samples, chosen, sampling, batch_size, run
        )
        return collect_scorable_set(split, samples, chosen, generated, dropped)

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

    def calculate_protocol_runs(
        self,
        generator: TextToMotionGenerator,
        request: GenerationEvaluationProtocol,
        split: GenerationEvaluationSet,
        reference_emb: np.ndarray,
        first_emb: np.ndarray,
    ) -> list[dict[str, float]]:
        caption_emb, per_clip = self.caption_embeddings(split)
        rng = np.random.default_rng(0)
        runs = [calculate_run_metrics(reference_emb, first_emb, caption_emb, per_clip, rng)]
        for run in range(1, request.reps):
            repeated = self.generate_evaluation_set(
                generator,
                request.split,
                request.max_clips,
                request.sampling,
                request.batch_size,
                run=run,
                clip_list=request.clip_list,
            )
            if repeated.sample_ids != split.sample_ids:
                raise RuntimeError(f"generation run {run} scored a different sample set than run 0")
            repeated_emb = self.compute_motion_embeddings(repeated.generated)
            captions = self.caption_embeddings(repeated)
            runs.append(calculate_run_metrics(reference_emb, repeated_emb, *captions, rng))
            logger.info(f"generation run {run + 1}/{request.reps}  FID {runs[-1]['fid']:.4f}")
        return runs

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
            clip_list=request.clip_list,
        )
        warn_about_dropped_clips(split)

        reference_emb = self.compute_motion_embeddings(split.reference)
        first_emb = self.compute_motion_embeddings(split.generated)
        runs = self.calculate_protocol_runs(generator, request, split, reference_emb, first_emb)
        bootstrap = (
            bootstrap_fid(reference_emb, first_emb, resamples=request.bootstrap)
            if request.bootstrap > 0
            else {}
        )

        multimodality = protocol_multimodality(self, generator, request, split)
        clips = len(split.generated)
        return build_protocol_report(runs, clips, request.reps, bootstrap, multimodality)

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

        return checkpoint_selection_record(generated, reference_emb, generated_emb, text_emb)


def warn_about_dropped_clips(split: GenerationEvaluationSet) -> None:
    if sum(split.dropped.values()):
        logger.warning(
            f"scored {len(split.generated)} of {split.requested} requested clips "
            f"(dropped {split.dropped}). FID is only comparable at equal clip counts."
        )


def protocol_multimodality(
    evaluator: HumanMl3dGenerationEvaluator,
    generator: TextToMotionGenerator,
    request: GenerationEvaluationProtocol,
    split: GenerationEvaluationSet,
) -> float:
    if request.mm_clips <= 0:
        return float("nan")
    downsample = evaluator.resources.tokenizer_downsample_factor
    token_lens = [feat.shape[0] // downsample for feat in split.reference]
    return evaluator.calculate_multimodality(
        generator,
        split.captions,
        token_lens,
        request.sampling,
        request.mm_clips,
        request.mm_repeats,
    )
