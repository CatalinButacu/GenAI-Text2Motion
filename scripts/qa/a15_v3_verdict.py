# python scripts/qa/a15_v3_verdict.py --uniform docs/paper/pics/a15_v3_verbs_uniform.csv --rareverb docs/paper/pics/a15_v3_verbs_rareverb.csv --out docs/paper/pics/a15_v3_kill_test.csv
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from a7_screen_verdict import BACKBONES, run_curve
from scipy import stats

SEED = 2026
UNIFORM = "a15_uniform"
RAREVERB = "a15_rareverb"
STOP_EPOCH = 6
RARE_CLIPS = 150
FID_RATIO = 1.1
R1_MARGIN = 0.02
RESAMPLES = 10000


def rare_verb_rates(path: Path, backbone: str) -> dict[str, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rates = {
        row["verb"]: float(row[f"{backbone}_top5"])
        for row in rows
        if int(row["train_clips"]) < RARE_CLIPS
    }
    if not rates:
        raise ValueError(f"no verb with fewer than {RARE_CLIPS} train clips in {path}")
    return rates


def paired_interval(differences: np.ndarray) -> tuple[float, float]:
    result = stats.bootstrap(
        (differences,),
        np.mean,
        confidence_level=0.95,
        n_resamples=RESAMPLES,
        rng=np.random.default_rng(SEED),
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def epoch_metrics(backbone: str, tag: str) -> dict[str, float]:
    curve = run_curve(backbone, tag)
    if STOP_EPOCH not in curve or "fid" not in curve[STOP_EPOCH]:
        raise RuntimeError(f"no validation FID at epoch {STOP_EPOCH} for {tag} {backbone}")
    return curve[STOP_EPOCH]


def twin_row(backbone: str, uniform_csv: Path, rareverb_csv: Path, arm: str) -> dict[str, object]:
    uniform_rates = rare_verb_rates(uniform_csv, backbone)
    rareverb_rates = rare_verb_rates(rareverb_csv, backbone)
    if uniform_rates.keys() != rareverb_rates.keys():
        raise ValueError("the two verb tables hold different rare verbs")
    verbs = sorted(uniform_rates)
    differences = np.array([rareverb_rates[verb] - uniform_rates[verb] for verb in verbs])
    low, high = paired_interval(differences)
    uniform = epoch_metrics(backbone, UNIFORM)
    rareverb = epoch_metrics(backbone, arm)
    verb_pass = low > 0.0
    fid_pass = rareverb["fid"] <= FID_RATIO * uniform["fid"]
    r1_pass = rareverb["r_top1"] >= uniform["r_top1"] - R1_MARGIN
    return {
        "arm": arm,
        "backbone": backbone,
        "rare_verbs": len(verbs),
        "uniform_rare_top5": float(np.mean([uniform_rates[verb] for verb in verbs])),
        "arm_rare_top5": float(np.mean([rareverb_rates[verb] for verb in verbs])),
        "difference": float(differences.mean()),
        "difference_low": low,
        "difference_high": high,
        "uniform_fid": uniform["fid"],
        "arm_fid": rareverb["fid"],
        "uniform_r1": uniform["r_top1"],
        "arm_r1": rareverb["r_top1"],
        "uniform_val_ce": uniform.get("val_ce", float("nan")),
        "arm_val_ce": rareverb.get("val_ce", float("nan")),
        "verb_pass": verb_pass,
        "fid_pass": fid_pass,
        "r1_pass": r1_pass,
        "pass": verb_pass and fid_pass and r1_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform", type=Path, required=True)
    parser.add_argument("--rareverb", type=Path, required=True)
    parser.add_argument("--arm", default=RAREVERB)
    parser.add_argument("--out", type=Path, default=Path("docs/paper/pics/a15_v3_kill_test.csv"))
    arguments = parser.parse_args()
    rows = [
        twin_row(backbone, arguments.uniform, arguments.rareverb, arguments.arm)
        for backbone in BACKBONES
    ]
    with arguments.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"A15 {arguments.arm}: {'PASS' if all(row['pass'] for row in rows) else 'FAIL'}")


if __name__ == "__main__":
    main()
