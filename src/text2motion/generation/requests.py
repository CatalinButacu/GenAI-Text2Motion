from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from text2motion.generation.contracts import SamplingConfig


############################################################
#       TextToMotionGenerationRequest
############################################################
@dataclass(frozen=True)
class TextToMotionGenerationRequest:
    prompt: str
    token_steps: int
    sampling: SamplingConfig = SamplingConfig()
    chunk_tokens: int = 1


############################################################
#       GeneratorTrainingRequest
############################################################
@dataclass(frozen=True)
class GeneratorTrainingRequest:
    epochs: int = 60
    batch_size: int = 64
    num_workers: int = 0
    grad_accum: int = 1
    eval_every: int = 5
    max_eval_clips: int = 600
    eval_batch_size: int = 32
    temperature: float = 1.0
    cfg_scale: float = 1.0
    checkpoint_name: str | None = None
    resume: bool = False
    fresh_selection: bool = False
    patience: int = 0
    accept_legacy_resume: bool = False
    extend_schedule: bool = False
    clean_ce: bool = True
    ce_batches: int = 50
    val_ce_patience: int = 6
    stop_epoch: int = 0
    select_on_val_ce: bool = False
    save_r1_floor: float = 0.0


############################################################
#       GeneratorPretrainingRequest
############################################################
@dataclass(frozen=True)
class GeneratorPretrainingRequest:
    token_pack: Path
    epochs: int = 30
    batch_size: int = 64
    grad_accum: int = 1
    num_workers: int = 0
    val_fraction: float = 0.05
    out_path: Path | None = None
    resume: bool = False
    accept_legacy_resume: bool = False
    extend_schedule: bool = False
