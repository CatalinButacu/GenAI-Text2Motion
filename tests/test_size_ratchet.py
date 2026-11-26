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
            "generation/training/trainer.py",
            "motion/representation.py",
            "studio/avatar.py",
            "studio/fitting.py",
            "studio/live/controller.py",
            "studio/scene.py",
            "tokenization/nn_quantizers.py",
            "tokenization/nn_tokenizer.py",
            "tokenization/tokenizer_trainer.py",
        ),
        "function": (
            "generation/training/trainer.py::GeneratorTrainer.__init__",
            "generation/training/trainer.py::GeneratorTrainer.batch_loss_parts",
            "motion/kinematics.py::Skeleton.inverse_kinematics_np",
            "motion/normalization.py::fit_train_stats",
            "motion/normalization.py::normalization_stats",
            "motion/representation.py::encode_joints",
            "studio/fitting.py::fit_smplx_to_joints",
            "studio/fitting.py::solve_smplx_pose_from_joints",
            "studio/live/controller.py::StudioGenerationController.accept_chunk",
            "studio/live/controller.py::StudioGenerationController.start_generation",
            "studio/live/controller.py::StudioGenerationController.stream_generation",
            "studio/scene.py::StreamSceneUpdater.append_chunks",
            "studio/session/state.py::StudioSession.__init__",
            "tokenization/corpus.py::build_token_pack",
            "tokenization/nn_quantizers.py::FiniteScalarQuantizer.__init__",
            "tokenization/nn_tokenizer.py::FiniteScalarQuantizationTokenizer.__init__",
            "tokenization/reconstruction_metrics.py::reconstruction_metrics",
            "tokenization/tokenizer_trainer.py::TokenizerTrainer.train_step",
            "tokenization/tokenizer_trainer.py::train_tokenizer",
            "tokenization/tokenizer_trainer.py::train_tokenizer.after_epoch",
        ),
        "class": (
            "generation/training/trainer.py::GeneratorTrainer",
            "motion/kinematics.py::Skeleton",
            "studio/avatar.py::SmplxAvatarController",
            "studio/live/controller.py::StudioGenerationController",
            "studio/scene.py::StreamSceneUpdater",
            "studio/scene.py::StudioSceneNodes",
            "studio/session/buffers.py::StudioStreamBuffers",
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
