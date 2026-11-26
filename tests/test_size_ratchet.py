import importlib
import sys
from collections import Counter
from pathlib import Path
from types import MappingProxyType

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO / "scripts" / "maintenance"))
AUDIT = importlib.import_module("size_audit")

LIMITS = AUDIT.Limits(250, 30, 10)
SOURCE_PREFIX = "src/text2motion/"
ALLOWED = MappingProxyType(
    {
        "file": (
            "app/service/handlers.py",
            "app/runtime/context.py",
            "dataset/sources/amass/corpus.py",
            "evaluation/metrics.py",
            "evaluation/nn_matcher.py",
            "generation/contracts.py",
            "generation/generator_trainer.py",
            "generation/nn_backbones.py",
            "generation/nn_generator.py",
            "motion/representation.py",
            "studio/avatar.py",
            "studio/fitting.py",
            "studio/generation.py",
            "studio/scene.py",
            "studio/session.py",
            "tokenization/nn_quantizers.py",
            "tokenization/nn_tokenizer.py",
            "tokenization/tokenizer_trainer.py",
        ),
        "function": (
            "app/runtime/context.py::ApplicationContext.load_text_to_motion_generator",
            "app/runtime/run_log.py::start_run",
            "app/service/client.py::MotionInferenceClient.generate",
            "app/service/handlers.py::MotionInferenceServer.handle_generate",
            "app/service/handlers.py::MotionInferenceServer.serve_client",
            "app/service/client.py::motion_service_client",
            "app/service/client.py::start_motion_server",
            "app/stages/serve.py::load_inference_pipeline",
            "app/stages/evaluate.py::run_evaluate",
            "app/stages/serve.py::run_studio",
            "app/stages/pretrain.py::run_pretrain",
            "app/stages/overfit.py::run_sanity_overfit",
            "app/stages/generator.py::run_train_generator",
            "app/stages/tokenizer.py::run_train_tokenizer",
            "common/engine.py::restore_clean_ce",
            "dataset/sources/amass/corpus.py::heldout_sources",
            "dataset/sources/amass/extractor.py::AmassPoseExtractor.__init__",
            "dataset/sources/amass/poses.py::extract_poses",
            "dataset/sources/humanml3d/build.py::build_joint_clips",
            "dataset/sources/humanml3d/official.py::write_motions",
            "dataset/training/windows.py::MotionWindowDataset.__init__",
            "dataset/training/windows.py::MotionWindowDataset.load_windowable_clips",
            "evaluation/metrics.py::transition_statistics",
            "evaluation/nn_matcher.py::GuoEvaluationResources.from_frozen_guo_assets",
            "generation/decoder.py::measure_decoder_context",
            "generation/decoder.py::steer_stream",
            "generation/generator_trainer.py::GeneratorTrainer.__init__",
            "generation/generator_trainer.py::GeneratorTrainer.batch_loss_parts",
            "generation/generator_trainer.py::finish_epoch",
            "generation/generator_trainer.py::overfit_single_batch",
            "generation/generator_trainer.py::train_generator",
            "generation/losses.py::generator_loss",
            "generation/nn_backbones.py::MambaMixer.forward",
            "generation/nn_generator.py::MotionTokenGenerator.check_stream_budget",
            "generation/nn_generator.py::MotionTokenGenerator.stream_token_indices",
            "generation/pretrain.py::pretrain_generator",
            "motion/kinematics.py::Skeleton.inverse_kinematics_np",
            "motion/normalization.py::fit_train_stats",
            "motion/normalization.py::normalization_stats",
            "motion/representation.py::encode_joints",
            "studio/fitting.py::fit_smplx_to_joints",
            "studio/fitting.py::solve_smplx_pose_from_joints",
            "studio/generation.py::StudioGenerationController.accept_chunk",
            "studio/generation.py::StudioGenerationController.start_generation",
            "studio/generation.py::StudioGenerationController.stream_generation",
            "studio/scene.py::StreamSceneUpdater.append_chunks",
            "studio/session.py::StudioSession.__init__",
            "tokenization/corpus.py::build_token_pack",
            "tokenization/nn_quantizers.py::FiniteScalarQuantizer.__init__",
            "tokenization/nn_tokenizer.py::FiniteScalarQuantizationTokenizer.__init__",
            "tokenization/reconstruction_metrics.py::reconstruction_metrics",
            "tokenization/tokenizer_trainer.py::TokenizerTrainer.train_step",
            "tokenization/tokenizer_trainer.py::train_tokenizer",
            "tokenization/tokenizer_trainer.py::train_tokenizer.after_epoch",
        ),
        "class": (
            "app/runtime/context.py::ApplicationContext",
            "generation/generator_trainer.py::GeneratorTrainer",
            "generation/nn_generator.py::MotionTokenGenerator",
            "motion/kinematics.py::Skeleton",
            "studio/avatar.py::SmplxAvatarController",
            "studio/generation.py::StudioGenerationController",
            "studio/scene.py::StreamSceneUpdater",
            "studio/scene.py::StudioSceneNodes",
            "studio/session.py::StudioStreamBuffers",
        ),
    }
)


def short_name(measure) -> str:
    path = measure.path.removeprefix(SOURCE_PREFIX)

    if measure.kind == "file":
        return path

    return f"{path}::{measure.name}"


def ratchet_problems(measures, allowed) -> list[str]:
    problems = []

    for kind in AUDIT.KINDS:
        limit = LIMITS.of_kind(kind)
        chosen = [item for item in measures if item.kind == kind]
        existing = {short_name(item) for item in chosen}
        over = Counter(short_name(item) for item in chosen if item.size > limit)
        listed = Counter(allowed[kind])

        for name in sorted(over - listed):
            problems.append(f"new offender: {kind} {name} is over the limit {limit}")

        for name in sorted(listed - over):
            reason = "is no longer over the limit" if name in existing else "no longer exists"
            problems.append(f"remove from ALLOWED: {kind} {name} {reason}")

    return problems


def function_measure(name: str, size: int):
    path = f"{SOURCE_PREFIX}app/sample.py"

    return AUDIT.Measure("function", "app", "no", path, name, 1, size)


def allowed_functions(*names: str) -> dict[str, tuple[str, ...]]:
    return {"file": (), "function": names, "class": ()}


def test_source_sizes_match_the_allow_list(monkeypatch):
    monkeypatch.chdir(REPO)
    problems = ratchet_problems(AUDIT.source_measures(), ALLOWED)
    assert not problems, "\n".join(problems)


def test_a_listed_offender_and_a_name_at_the_limit_pass():
    measures = [function_measure("old", 31), function_measure("small", 30)]
    assert ratchet_problems(measures, allowed_functions("app/sample.py::old")) == []


def test_a_new_offender_fails():
    measures = [function_measure("grown", 31)]
    assert ratchet_problems(measures, allowed_functions()) == [
        "new offender: function app/sample.py::grown is over the limit 30"
    ]


def test_a_listed_name_that_is_small_or_gone_fails():
    measures = [function_measure("fixed", 30)]
    allowed = allowed_functions("app/sample.py::fixed", "app/sample.py::gone")
    assert ratchet_problems(measures, allowed) == [
        "remove from ALLOWED: function app/sample.py::fixed is no longer over the limit",
        "remove from ALLOWED: function app/sample.py::gone no longer exists",
    ]
