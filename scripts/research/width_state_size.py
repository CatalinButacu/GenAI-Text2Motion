# python scripts/research/width_state_size.py --out artifacts/outputs/qa/width_state_bytes.csv
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from text2motion.app.config import load_config
from text2motion.evaluation.benchmark import benchmark_streaming_horizons
from text2motion.generation.contracts import Backbone
from text2motion.generation.nn_model import GeneratorModelSpec

WIDTHS = (256, 320, 384, 448, 512)
HORIZONS = (1, 16, 48, 95)
CODEBOOK_SIZE = 1024


def state_rows(width: int, backbone: Backbone) -> list[dict[str, int | str | float]]:
    config = load_config(f"configs/generator/size_sweep/width_{width}.yaml")
    spec = GeneratorModelSpec.resolve(
        config.generator, backbone, CODEBOOK_SIZE, config.tokenizer.num_quantizers
    )
    generator = spec.create_model("cpu").eval()
    parameters = sum(parameter.numel() for parameter in generator.parameters())
    rows = benchmark_streaming_horizons(generator, list(HORIZONS), "cpu", warmup_steps=0)
    return [
        {
            "width": width,
            "backbone": backbone.value,
            "parameters": parameters,
            "horizon": row["horizon"],
            "state_bytes": row["state_bytes"],
        }
        for row in rows
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/outputs/qa/width_state_bytes.csv")
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    rows = [row for width in WIDTHS for backbone in Backbone for row in state_rows(width, backbone)]
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
