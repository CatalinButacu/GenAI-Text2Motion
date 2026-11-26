from __future__ import annotations

from pathlib import Path

from pydantic import Field

from text2motion.app.requests.base import (
    CommandRequest,
)
from text2motion.generation.contracts import Backbone
from text2motion.generation.requests import GeneratorPretrainingRequest, GeneratorTrainingRequest
from text2motion.motion.contracts import CorpusLayout


############################################################
#       SanityOverfitRequest
############################################################
class SanityOverfitRequest(CommandRequest):
    exploratory: bool = Field(
        default=False,
        description="record a non-protocol run that is not citable; protocol runs use seed 2026",
    )
    backbone: Backbone = Field(
        description="sequence mixer: mamba has a fixed-size state, transformer has a cache that grows"
    )
    tokenizer_ckpt: Path = Field(
        description="frozen tokenizer checkpoint; it must be the one the generator was trained on"
    )
    overfit_gate: Path = Field(
        description="output gate file; train-generator refuses to start without it"
    )
    batch_size: int = Field(
        default=4,
        description="clips in the one batch that the model must memorise",
    )
    steps: int = Field(
        default=1000,
        description="maximum optimizer steps before the gate fails",
    )
    lr: float = Field(
        default=3e-3,
        description="learning rate for the gate only; high on purpose, to memorise fast",
    )
    init_ckpt: Path | None = Field(
        default=None,
        description="pretrained weights to start the gate from, as the run that uses the gate does",
    )


############################################################
#       TrainGeneratorRequest
############################################################
class TrainGeneratorRequest(CommandRequest):
    exploratory: bool = Field(
        default=False,
        description="record a non-protocol run that is not citable; protocol runs use seed 2026",
    )
    backbone: Backbone = Field(
        description="sequence mixer: mamba has a fixed-size state, transformer has a cache that grows"
    )
    tokenizer_ckpt: Path | None = Field(
        default=None,
        description="frozen tokenizer checkpoint; it must be the one the generator was trained on",
    )
    overfit_gate: Path = Field(
        description="passed gate file from sanity-overfit for this config, tokenizer and backbone"
    )
    init_ckpt: Path | None = Field(
        default=None,
        description="pretrained weights to start from; without it training starts from random weights",
    )
    text_cache: bool = Field(
        default=True,
        description="reuse the frozen CLIP layer states of the train captions from a disk cache; the results are the same, the step is faster",
    )
    compile: bool = Field(
        default=True,
        description="compile the generator with torch.compile; about 7% faster for mamba, the results are not bit-identical to an eager run; without CUDA or Triton the run stays eager",
    )
    epochs: int = Field(
        default=GeneratorTrainingRequest().epochs,
        description="target epoch; the cosine schedule spans exactly this many epochs",
    )
    batch_size: int = Field(
        default=GeneratorTrainingRequest().batch_size,
        description="clips per step; keep it equal for both backbones",
    )
    num_workers: int = Field(
        default=GeneratorTrainingRequest().num_workers,
        description="data loader processes; more is faster, and each one needs RAM; 0 needs the least",
    )
    grad_accum: int = Field(
        default=GeneratorTrainingRequest().grad_accum,
        description="batches per optimizer step; effective batch is batch_size times this",
    )
    cfg_scale: float = Field(
        default=GeneratorTrainingRequest().cfg_scale,
        description="guidance scale for the validation samples; higher follows the text more closely",
    )
    eval_every: int = Field(
        default=GeneratorTrainingRequest().eval_every,
        description="epochs between validation runs; each run can select a new best checkpoint",
    )
    max_eval_clips: int = Field(
        default=GeneratorTrainingRequest().max_eval_clips,
        description="validation clips; fewer is faster and gives a noisier FID",
    )
    eval_batch_size: int = Field(
        default=GeneratorTrainingRequest().eval_batch_size,
        description="clips generated together in validation; changes speed, not the result",
    )
    temperature: float = Field(
        default=GeneratorTrainingRequest().temperature,
        description="sampling temperature for validation; higher gives more varied motion",
    )
    vocab_dir: Path | None = Field(
        default=None,
        description="word-vector folder of the evaluator; default from the config",
    )
    ckpt_name: str | None = Field(
        default=None,
        description="file name of the best checkpoint inside the checkpoint folder",
    )
    resume: bool = Field(
        default=False, description="continue from the saved last state of ckpt_name"
    )
    fresh_selection: bool = Field(
        default=False,
        description="with --resume: reset earlier best FID and validation-CE patience for this run",
    )
    resume_run_dir: Path | None = Field(
        default=None,
        description="existing run folder to continue the log in; keeps one folder for one training",
    )
    patience: int = Field(
        default=GeneratorTrainingRequest().patience,
        description="stop after this many validations without a better FID; 0 never stops early",
    )
    clean_ce: bool = Field(
        default=GeneratorTrainingRequest().clean_ce,
        description="after each epoch, log the teacher-forced CE on a fixed train subset and on validation",
    )
    ce_batches: int = Field(
        default=GeneratorTrainingRequest().ce_batches,
        description="batches of --eval_batch_size clips for each clean CE; 0 uses the full split",
    )
    val_ce_patience: int = Field(
        default=GeneratorTrainingRequest().val_ce_patience,
        ge=0,
        description="stop on this consecutive epoch without a lower clean validation CE; default 6 allows five retries; 0 disables",
    )
    stop_epoch: int = Field(
        default=GeneratorTrainingRequest().stop_epoch,
        ge=0,
        description="stop after this epoch while --epochs still sets the cosine length; 0 runs to --epochs",
    )
    select_on_val_ce: bool = Field(
        default=GeneratorTrainingRequest().select_on_val_ce,
        description="save the best bundle at the lowest clean validation CE, not at the lowest val FID",
    )
    save_r1_floor: float = Field(
        default=GeneratorTrainingRequest().save_r1_floor,
        ge=0,
        le=1,
        description="save the best-FID bundle only at an evaluation whose val R@1 is at least this value; an evaluation below it counts as no improvement; default 0 keeps every run unchanged",
    )
    accept_legacy_resume: bool = Field(
        default=False,
        description="with --resume: load an old resume file; its RNG and schedule state start fresh",
    )
    extend_schedule: bool = Field(
        default=False,
        description="with --resume: accept a new --epochs; the learning rate follows the new cosine",
    )


############################################################
#       PretrainRequest
############################################################
class PretrainRequest(CommandRequest):
    exploratory: bool = Field(
        default=False,
        description="record a non-protocol run that is not citable; protocol runs use seed 2026",
    )
    backbone: Backbone = Field(
        description="sequence mixer: mamba has a fixed-size state, transformer has a cache that grows"
    )
    token_pack: Path = Field(
        default=CorpusLayout().pretraining_token_pack,
        description="token file from the tokenize command; the unlabeled motion to learn from",
    )
    epochs: int = Field(
        default=GeneratorPretrainingRequest(token_pack=Path()).epochs,
        description="passes over the token corpus; too many memorise it",
    )
    batch_size: int = Field(
        default=GeneratorPretrainingRequest(token_pack=Path()).batch_size,
        description="windows per step; keep it equal for both backbones",
    )
    grad_accum: int = Field(
        default=GeneratorPretrainingRequest(token_pack=Path()).grad_accum,
        description="batches per optimizer step; effective batch is batch_size times this",
    )
    num_workers: int = Field(
        default=GeneratorPretrainingRequest(token_pack=Path()).num_workers,
        description="data loader processes; 0 needs the least RAM",
    )
    val_fraction: float = Field(
        default=GeneratorPretrainingRequest(token_pack=Path()).val_fraction,
        description="share of clips held out, by clip; shows if the model memorises the corpus",
    )
    ckpt: Path | None = Field(
        default=None,
        description="output checkpoint path; default is generator_<backbone>_pretrained.pt",
    )
    resume: bool = Field(
        default=False, description="continue from the saved state of this checkpoint"
    )
    accept_legacy_resume: bool = Field(
        default=False,
        description="with --resume: load an old resume file; its RNG and schedule state start fresh",
    )
    extend_schedule: bool = Field(
        default=False,
        description="with --resume: accept a new --epochs; the learning rate follows the new cosine",
    )
