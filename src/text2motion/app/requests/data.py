from __future__ import annotations

from pathlib import Path

from pydantic import Field

from text2motion.app.requests.base import (
    CommandRequest,
    DefaultConfigRequest,
)
from text2motion.dataset.contracts import MotionDataConfig
from text2motion.motion.contracts import CorpusLayout, PreparationStage
from text2motion.tokenization.contracts import TokenizerKind


############################################################
#       PrepareRequest
############################################################
class PrepareRequest(DefaultConfigRequest):
    stage: PreparationStage = Field(
        default=PreparationStage.ALL,
        description="run one preparation stage only (clock, joints, features, stats); all runs them in sequence",
    )
    corpus: bool = Field(
        default=False,
        description="build the unlabeled AMASS pretraining corpus and not the captioned data set",
    )
    adopt_pose_folder: bool = Field(
        default=False,
        description="stage amass: accept an existing pose folder without a manifest after a new "
        "extraction of a fixed sample matches the files on disk",
    )
    regen_dir: Path = Field(
        default=CorpusLayout().regen_dir,
        description="folder with the regenerated poses and joints that the corpus is built from",
    )
    corpus_dir: Path = Field(
        default=CorpusLayout().amass_corpus_dir,
        description="output folder of the AMASS 263-feature corpus",
    )
    min_frames: int = Field(
        default=MotionDataConfig().min_motion_len,
        description="clips shorter than this are dropped; a higher value gives fewer, longer clips",
    )
    limit: int = Field(
        default=0,
        description="convert only the first N sequences, for a quick test; 0 converts all",
    )
    source_release: Path | None = Field(
        default=None,
        description="folder of an official HumanML3D release to convert; skips all other stages",
    )
    converted_dir: Path = Field(
        default=CorpusLayout().official_dir,
        description="output folder for the converted official release",
    )


############################################################
#       TokenizeRequest
############################################################
class TokenizeRequest(CommandRequest):
    features_dir: Path = Field(
        default=CorpusLayout().amass_features_dir,
        description="folder of 263-feature clips to encode",
    )
    tokenizer_ckpt: Path | None = Field(
        default=None,
        description="frozen tokenizer checkpoint; default from the config",
    )
    token_pack: Path = Field(
        default=CorpusLayout().amass_token_pack,
        description="output .npz file of tokens; pretraining reads this file",
    )
    segment_frames: int = Field(
        default=MotionDataConfig().max_motion_len,
        gt=0,
        description="window length in frames; clips shorter than one window are not encoded",
    )
    stride: int = Field(
        default=MotionDataConfig().max_motion_len,
        gt=0,
        description="step between windows; a smaller stride gives more, overlapping windows",
    )
    regen_dir: Path = Field(
        default=CorpusLayout().regen_dir,
        description="folder with pose_data; the heldout clips are found and excluded through it",
    )


############################################################
#       TrainTokenizerRequest
############################################################
class TrainTokenizerRequest(CommandRequest):
    exploratory: bool = Field(
        default=False,
        description="record a non-protocol run that is not citable; protocol runs use seed 2026",
    )
    tokenizer: TokenizerKind = Field(
        description="quantizer family: fsq has no learned codebook, rvq is the baseline"
    )
    epochs: int = Field(default=50, gt=0, description="passes over the training set")
    window: int = Field(
        default=64,
        gt=0,
        description="training crop in frames; a longer crop gives more context and uses more memory",
    )
    batch_size: int = Field(default=128, gt=0, description="clips per step; limited by GPU memory")
    ema_decay: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="decay of the averaged weights that are evaluated; higher is smoother and slower",
    )
    eval_every: int = Field(
        default=5, gt=0, description="epochs between reconstruction evaluations"
    )
    max_eval_clips: int | None = Field(
        default=None,
        gt=0,
        description="cap on evaluation clips; a low cap is faster and gives a noisier FID",
    )
    max_steps: int | None = Field(
        default=None,
        gt=0,
        description="stop after this many optimizer steps, for equal-step comparisons; not with --resume",
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        description="data loader processes; 0 loads in the training process and needs the least RAM",
    )
    amass_share: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="share of AMASS clips in each epoch; 0 trains on HumanML3D only",
    )
    amass_features: Path | None = Field(
        default=None,
        description="folder with the AMASS 263 features; required when amass_share is above 0",
    )
    ckpt_name: str | None = Field(
        default=None,
        description="checkpoint file name inside the checkpoint folder; also names the run",
    )
    resume: bool = Field(
        default=False, description="continue from the saved state of this checkpoint"
    )
    accept_legacy_resume: bool = Field(
        default=False,
        description="with --resume: load an old resume file; its RNG state starts fresh",
    )
