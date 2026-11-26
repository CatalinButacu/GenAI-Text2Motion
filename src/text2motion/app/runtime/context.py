from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from text2motion.app.config.loading import load_config
from text2motion.app.config.schema import (
    ApplicationConfig,
    MotionServiceConfig,
)
from text2motion.app.runtime.environment import apply_precision, resolve_device, seed_everything
from text2motion.app.runtime.model_loader import ModelLoader
from text2motion.app.runtime.run_log import ensure_logging_configured
from text2motion.dataset.training.module import MotionDataModule
from text2motion.evaluation.evaluator import HumanMl3dGenerationEvaluator
from text2motion.evaluation.nn_matcher import GuoEvaluationResources
from text2motion.motion.contracts import PreparationRequest
from text2motion.motion.storage import FAILURE_DUMP_FILE

logger = logging.getLogger(__name__)


############################################################
#       ApplicationContext
############################################################
@dataclass
class ApplicationContext:
    config: ApplicationConfig
    device: str
    config_path: Path
    seed: int

    @property
    def models(self) -> ModelLoader:
        return ModelLoader(self.config, self.device)

    def create_motion_repository(self) -> MotionDataModule:
        root = self.config.paths.hml3d_out_dir
        if root is None:
            raise ValueError("paths.hml3d_out_dir must be set (regenerated 263 output root)")
        return MotionDataModule(
            root=Path(root),
            data=self.config.data,
            texts_dir=self.config.paths.texts_dir,
            seed=self.config.seed,
        )

    def load_guo_evaluation_resources(
        self, vocab_dir: str | Path | None = None
    ) -> GuoEvaluationResources:
        path_config = self.config.paths
        if path_config.eval_matcher is None or path_config.eval_stats_dir is None:
            raise ValueError(
                "paths.eval_matcher and paths.eval_stats_dir must be set to score anything: FID "
                "and R-precision are defined by the frozen Guo et al. evaluator (ADR 0001)."
            )
        return GuoEvaluationResources.from_frozen_guo_assets(
            motion_dataset_dir=required_path(path_config.hml3d_out_dir, "hml3d_out_dir"),
            eval_stats_dir=path_config.eval_stats_dir,
            eval_matcher=path_config.eval_matcher,
            vocab_dir=Path(vocab_dir) if vocab_dir else path_config.our_vab_dir,
            tokenizer_downsample_factor=self.config.tokenizer.downsample,
            annotation_dir=path_config.texts_dir,
            device=self.device,
        )

    def create_humanml3d_generation_evaluator(
        self, vocab_dir: str | Path | None = None
    ) -> HumanMl3dGenerationEvaluator:
        return HumanMl3dGenerationEvaluator(self.load_guo_evaluation_resources(vocab_dir))

    def create_hml3d_preparation_request(self) -> PreparationRequest:
        path_config = self.config.paths
        return PreparationRequest(
            out_dir=required_path(path_config.hml3d_out_dir, "hml3d_out_dir"),
            amass_dir=path_config.amass_dir,
            smplx_models=path_config.smplx_models,
            index_csv=path_config.hml3d_index_csv,
            device=self.device,
        )

    def checkpoint_directory(self, stage: str | None = None) -> Path:
        directory = Path(self.config.paths.checkpoints_dir)
        if stage:
            directory = directory / stage
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def failure_dump_path(self, run_dir: Path) -> Path | None:
        if not self.config.failure_logs:
            return None
        return Path(run_dir) / FAILURE_DUMP_FILE

    def with_config(self, config_path: str | Path) -> ApplicationContext:
        return ApplicationBootstrap.from_config(config_path, device=self.device)

    def with_service(self, service: MotionServiceConfig) -> ApplicationContext:
        return ApplicationContext(
            config=replace(self.config, service=service),
            device=self.device,
            config_path=self.config_path,
            seed=self.seed,
        )

    def with_service_file(self, service_config: str | Path) -> ApplicationContext:
        from text2motion.app.config.loading import load_service_config

        path = Path(service_config)
        if not path.is_file():
            raise FileNotFoundError(f"service config {path} not found (pass --service_config)")
        return self.with_service(load_service_config(path))


def required_path(value: Path | None, key: str) -> Path:
    if value is None:
        raise ValueError(f"paths.{key} must be set in the config")
    return value


############################################################
#       ApplicationBootstrap
############################################################
class ApplicationBootstrap:
    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        device: str | None = None,
        seed: int | None = None,
    ) -> ApplicationContext:
        ensure_logging_configured()
        config = load_config(config_path)
        resolved_seed = config.seed if seed is None else seed
        seed_everything(resolved_seed, config.deterministic)
        apply_precision(config.precision)
        torch.autograd.set_detect_anomaly(config.detect_anomaly)
        resolved_device = resolve_device(config.device, device)
        logger.info(
            f"config {config_path}  device {resolved_device}  seed {resolved_seed}  "
            f"deterministic {config.deterministic}  precision {config.precision.value}  "
            f"failure_logs {config.failure_logs}  detect_anomaly {config.detect_anomaly}"
        )
        return ApplicationContext(
            config=config,
            device=resolved_device,
            config_path=Path(config_path),
            seed=resolved_seed,
        )
