# python scripts/evaluation/tokenizer_pair_bootstrap.py --split val --resamples 1000 --out artifacts/outputs/qa/tokenizer_pair_bootstrap.json
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.evaluation.metrics import paired_fid_delta
from text2motion.motion.contracts import Split
from text2motion.tokenization.metrics import evaluate_tokenizer_reconstruction

SEED = 2026
CONFIGS = MappingProxyType(
    {
        "fsq_g4_v512": "configs/tokenizer/fsq_g4_v512.yaml",
        "rvq_l4_512": "configs/tokenizer/rvq_l4_512.yaml",
        "fsq_g6_v512": "configs/tokenizer/tokenizer_isovocab.yaml",
        "rvq_l6_512": "configs/tokenizer/rvq_l6_512.yaml",
        "fsq_g8_v512": "configs/tokenizer/fsq_g8_v512.yaml",
        "rvq_l8_512": "configs/tokenizer/rvq_l8_512.yaml",
        "fsq_g4_v1024": "configs/tokenizer/fsq_g4_v1024.yaml",
        "rvq_l4_1024": "configs/tokenizer/rvq_l4_1024.yaml",
        "fsq_g6_v1024": "configs/tokenizer/fsq_g6_v1024.yaml",
        "rvq_l6_1024": "configs/tokenizer/rvq_l6_1024.yaml",
        "fsq_g8_v1024": "configs/tokenizer/fsq_g8_v1024.yaml",
        "rvq_l8_1024": "configs/tokenizer/rvq_l8_1024.yaml",
    }
)
PAIRS = (
    ("fsq_g4_v512", "rvq_l4_512"),
    ("fsq_g6_v512", "rvq_l6_512"),
    ("fsq_g8_v512", "rvq_l8_512"),
    ("fsq_g4_v1024", "rvq_l4_1024"),
    ("fsq_g6_v1024", "rvq_l6_1024"),
    ("fsq_g8_v1024", "rvq_l8_1024"),
)


class CapturingScorer:
    def __init__(self, evaluator) -> None:
        self.evaluator = evaluator
        self.reference: list[np.ndarray] = []
        self.reconstruction: list[np.ndarray] = []

    def __call__(self, reference: list[np.ndarray], reconstruction: list[np.ndarray]) -> float:
        self.reference = reference
        self.reconstruction = reconstruction
        return self.evaluator.calculate_reconstruction_fid(reference, reconstruction)


def reconstruction_embeddings(
    stem: str, split: Split, evaluator
) -> tuple[np.ndarray, np.ndarray, float]:
    app = ApplicationBootstrap.from_config(CONFIGS[stem], seed=SEED)
    module = app.create_tokenizer_network()
    checkpoint = Path(app.config.paths.checkpoints_dir) / "tokenizer" / f"{stem}.pt"
    module.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    module.to(app.device).eval()
    repository = app.create_motion_repository()
    scorer = CapturingScorer(evaluator)
    metrics = evaluate_tokenizer_reconstruction(
        module,
        repository.root,
        repository.load_scaler(),
        max_frames=app.config.data.max_motion_len,
        device=app.device,
        split=split,
        shuffle_seed=SEED,
        reconstruction_fid_scorer=scorer,
    )
    reference = evaluator.compute_motion_embeddings(scorer.reference)
    reconstruction = evaluator.compute_motion_embeddings(scorer.reconstruction)
    return reference, reconstruction, float(metrics["recon_fid"])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Split, default=Split.VALIDATION)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/outputs/qa/tokenizer_pair_bootstrap.json")
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    base = ApplicationBootstrap.from_config(CONFIGS["fsq_g8_v1024"], seed=SEED)
    evaluator = base.create_humanml3d_generation_evaluator()
    rows = []
    for fsq, rvq in PAIRS:
        real, fsq_embeddings, fsq_fid = reconstruction_embeddings(fsq, arguments.split, evaluator)
        real_again, rvq_embeddings, rvq_fid = reconstruction_embeddings(
            rvq, arguments.split, evaluator
        )
        if not np.array_equal(real, real_again):
            raise ValueError(f"{fsq} and {rvq} were scored on different clips")
        delta = paired_fid_delta(real, fsq_embeddings, rvq_embeddings, arguments.resamples, SEED)
        row = {
            "fsq": fsq,
            "rvq": rvq,
            "fsq_fid": fsq_fid,
            "rvq_fid": rvq_fid,
            "clips": len(real),
            **delta,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(
        json.dumps({"split": arguments.split.value, "pairs": rows}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
