# .\.venv\Scripts\python.exe scripts/qa/golden_outputs.py save-tokenizer|save-generator
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from text2motion.app.bootstrap import ApplicationBootstrap, apply_precision
from text2motion.app.config import Float32Precision, load_config
from text2motion.generation.contracts import Backbone, SamplingConfig
from text2motion.generation.losses import soft_decode
from text2motion.motion.contracts import Split
from text2motion.motion.storage import sha256_file
from text2motion.tokenization.contracts import TokenizerKind
from text2motion.tokenization.nn_model import TokenizerNetwork, build_tokenizer_network

TOKENIZER_GOLDEN_PATH = Path("tests/golden/tokenizer_golden.json")
FSQ_CONFIG = Path("configs/tokenizer/fsq_g8_v1024.yaml")
FSQ_CHECKPOINT = Path("artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt")
RVQ_CONFIG = Path("configs/tokenizer/rvq_l8_1024.yaml")
RVQ_CHECKPOINT = Path("artifacts/checkpoints/tokenizer/rvq_l8_1024.pt")
TOKENIZER_INPUTS = (FSQ_CONFIG, FSQ_CHECKPOINT, RVQ_CONFIG, RVQ_CHECKPOINT)
GENERATOR_GOLDEN_PATH = Path("tests/golden/generator_golden.json")
GENERATOR_CONFIG = Path("configs/generator/gen_pilot_fsq8x1024.yaml")
GENERATOR_CASES = (
    (
        Backbone.TRANSFORMER,
        Path("artifacts/checkpoints/generator/generator_transformer_bs8_e60.pt"),
    ),
    (Backbone.MAMBA, Path("artifacts/checkpoints/generator/generator_mamba_bs8_e60.pt")),
)
GOLDEN_CLIPS = 32
GENERATOR_INPUTS = (
    GENERATOR_CONFIG,
    FSQ_CHECKPOINT,
    *(checkpoint for _, checkpoint in GENERATOR_CASES),
    Path("data/official_evaluator/extracted/text_mot_match/model/finest.tar"),
    Path("data/eval_stats/mean.npy"),
    Path("data/eval_stats/std.npy"),
    Path("data/HumanML3D_official/Mean.npy"),
    Path("data/HumanML3D_official/Std.npy"),
    Path("data/HumanML3D_official/val.txt"),
)


def tensor_fingerprint(tensor: torch.Tensor) -> dict[str, object]:
    contiguous = tensor.detach().cpu().contiguous()
    raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
    }


def input_pins(paths: tuple[Path, ...]) -> dict[str, str]:
    return {path.as_posix(): sha256_file(path) for path in paths}


def __tokenizer_network(
    config_path: Path,
    kind: TokenizerKind,
    checkpoint: Path | None = None,
    quantizer: str | None = None,
) -> TokenizerNetwork:
    config = load_config(config_path)
    tokenizer_config = replace(config.tokenizer, kind=kind)
    if quantizer is not None:
        tokenizer_config = replace(tokenizer_config, quantizer=quantizer, quant_dropout=0.5)
    torch.manual_seed(7)
    network = build_tokenizer_network(tokenizer_config, config.rvq_baseline)
    if checkpoint is None:
        return network
    network.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return network


def __case_outputs(name: str, network: TokenizerNetwork) -> dict[str, torch.Tensor]:
    results: dict[str, torch.Tensor] = {}
    torch.manual_seed(0)
    motion = torch.randn(2, 64, 263)
    network.eval()
    with torch.no_grad():
        indices = network.encode(motion)
        results[f"{name}/indices"] = indices
        results[f"{name}/decode"] = network.decode(indices)
        reconstruction = network(motion)
        results[f"{name}/recon"] = reconstruction[0]
        results[f"{name}/commit"] = reconstruction[2]
        torch.manual_seed(1)
        logits = torch.randn(2, 16, indices.shape[-1], 1024)
        results[f"{name}/soft_decode"] = soft_decode(logits, network)
    if name == "rvq_l8":
        return results
    network.train()
    torch.manual_seed(3)
    training = network(motion)
    results[f"{name}/train_recon"] = training[0].detach()
    results[f"{name}/train_indices"] = training[1]
    return results


def tokenizer_outputs() -> dict[str, torch.Tensor]:
    cases = {
        "fsq_g8": __tokenizer_network(FSQ_CONFIG, TokenizerKind.FINITE_SCALAR, FSQ_CHECKPOINT),
        "rvq_l8": __tokenizer_network(RVQ_CONFIG, TokenizerKind.RESIDUAL_VECTOR, RVQ_CHECKPOINT),
        "fsq_residual": __tokenizer_network(
            FSQ_CONFIG, TokenizerKind.FINITE_SCALAR, quantizer="residual"
        ),
    }
    results: dict[str, torch.Tensor] = {}
    for name, network in cases.items():
        results |= __case_outputs(name, network)
    return results


def __save_tokenizer_golden() -> None:
    outputs = {name: tensor_fingerprint(value) for name, value in tokenizer_outputs().items()}
    record = {
        "torch": torch.__version__,
        "inputs": input_pins(TOKENIZER_INPUTS),
        "outputs": outputs,
    }
    TOKENIZER_GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKENIZER_GOLDEN_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(outputs)} fingerprints -> {TOKENIZER_GOLDEN_PATH}")


def generator_metrics() -> dict[str, dict[str, float]]:
    app = ApplicationBootstrap.from_config(GENERATOR_CONFIG, device="cuda")
    apply_precision(Float32Precision.HIGHEST)
    tokenizer = app.load_tokenizer(FSQ_CHECKPOINT)
    evaluator = app.create_humanml3d_generation_evaluator()
    metrics: dict[str, dict[str, float]] = {}
    for backbone, checkpoint in GENERATOR_CASES:
        generator = app.load_text_to_motion_generator(
            backbone, checkpoint, tokenizer, tokenizer_checkpoint=FSQ_CHECKPOINT
        )
        metrics[backbone.value] = evaluator.evaluate_checkpoint_selection_metrics(
            generator,
            Split.VALIDATION,
            max_clips=GOLDEN_CLIPS,
            sampling=SamplingConfig(),
            batch_size=8,
        )
    return metrics


def __save_generator_golden() -> None:
    record = {
        "torch": torch.__version__,
        "inputs": input_pins(GENERATOR_INPUTS),
        "outputs": generator_metrics(),
    }
    GENERATOR_GOLDEN_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"saved metrics of {len(GENERATOR_CASES)} generators -> {GENERATOR_GOLDEN_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="write the golden output fingerprints")
    parser.add_argument("command", choices=["save-tokenizer", "save-generator"])
    arguments = parser.parse_args()
    if arguments.command == "save-generator":
        __save_generator_golden()
        return
    __save_tokenizer_golden()


if __name__ == "__main__":
    main()
