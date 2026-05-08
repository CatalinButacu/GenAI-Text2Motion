#!/usr/bin/env python
"""Run the ablation study: N prompts x 2 configs -> metrics CSV/JSON.

Ablation matrix (matches current pipeline: SpaCy + L-BFGS-B + MotionSSM + SMPL-X render):
  full   -- SpaCy + L-BFGS-B + MotionSSM  (baseline)
  no_m2  -- random layout instead of L-BFGS-B planner  (M2 ablation)

Usage
-----
  python scripts/evaluation/evaluate_ablation.py --output results/ablation
  python scripts/evaluation/evaluate_ablation.py --output results/ablation --n-prompts 10
  python scripts/evaluation/evaluate_ablation.py --output results/ablation --configs full
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.evaluation.compute_metrics import computeClipMetrics, summarise
from src.pipeline import Pipeline
from src.shared.config import PipelineConfig

log = logging.getLogger(__name__)

#  Test prompt suite
# 50 prompts spanning the full range of motion types from the thesis 4
EVAL_PROMPTS: list[str] = [
    #  Standing / walking (10)
    "a person walks forward",
    "a person walks in a circle",
    "a person walks slowly",
    "a person walks and turns left",
    "a person strolls through a park",
    "a human walks up stairs",
    "a person walks backward",
    "a figure walks with a limp",
    "a person walks confidently",
    "a humanoid walks across the room",
    #  Running / jumping (8)
    "a person runs fast",
    "a person jogs lightly",
    "a person jumps over an obstacle",
    "a person leaps forward",
    "a person runs and jumps",
    "a human sprints",
    "a figure hops on one foot",
    "a person bounds forward",
    #  Kicking / throwing (8)
    "a person kicks a ball",
    "a person kicks with their right foot",
    "a person throws an object overhead",
    "a person throws a ball forward",
    "a person kicks and falls",
    "a person throws something to the side",
    "a human kicks repeatedly",
    "a person punches forward",
    #  Falling / physics interactions (8)
    "a person falls down",
    "a person stumbles and falls",
    "a ball falls onto a cube",
    "a person trips and falls forward",
    "a person falls backward",
    "a sphere bounces on the floor",
    "a person collapses",
    "an object tumbles to the ground",
    #  Sitting / standing / crouching (8)
    "a person sits down on a chair",
    "a person stands up from a chair",
    "a person crouches down",
    "a person kneels",
    "a person bends forward",
    "a person stretches their arms",
    "a person raises their hands",
    "a person turns around",
    #  Dance / complex motion (8)
    "a person dances",
    "a person waves hello",
    "a person claps their hands",
    "a person nods their head",
    "a person shakes their head",
    "a person reaches for something high",
    "a person picks something up from the floor",
    "a person does a spin",
]

assert len(EVAL_PROMPTS) == 50, f"Expected 50 prompts, got {len(EVAL_PROMPTS)}"

# Out-of-distribution (OOD) prompts: phrasing and vocabulary NOT present in HumanML3D.
# Purpose: test whether the model generalises beyond training distribution
# (unusual style, compound actions, non-standard vocabulary).
OOD_PROMPTS: list[str] = [
    # Unusual writing style / formal register
    "the subject ambulates in a forward direction",
    "the individual performs a locomotion sequence",
    "a biped entity executes a vertical displacement",
    # Compound / sequential actions not in training set
    "a person walks, stops, looks around, then continues walking",
    "a figure runs forward, pivots sharply, and sprints back",
    "a human crouches, picks something up, stands, and throws it",
    # Emotion-qualified motion
    "a person walks as if very tired and exhausted",
    "someone jogs happily with arms swinging wide",
    "a figure moves cautiously as if afraid of falling",
    # Domain-shifted vocabulary
    "an actor strides across the stage",
    "a soldier marches in formation",
    "a dancer glides gracefully across the floor",
    # Negation / contrast (typically absent in HumanML3D)
    "a person almost falls but catches themselves",
    "a figure starts to run but slows to a walk",
    # Unusual body-part emphasis
    "a person moves using only their upper body",
    "a human sways their hips while standing still",
    "a figure stretches one arm to the ceiling",
    # Edge-case motion extremes
    "a person stands completely motionless",
    "a person moves as slowly as possible",
]

assert len(OOD_PROMPTS) == 19, f"Expected 19 OOD prompts, got {len(OOD_PROMPTS)}"


#  Config definitions

ALL_CONFIGS = ["full", "no_m2"]


def pipelineConfigFor(configName: str, outputDir: str, duration: float = 3.0,
                      fps: int = 24, device: str = "cpu"):
    """Build PipelineConfig for a given ablation config name."""
    if configName not in ALL_CONFIGS:
        raise ValueError(f"Unknown config: {configName!r}")
    cfg = PipelineConfig(
        fps=fps,
        duration=float(duration),
        device=device,
        outputDir=outputDir,
    )
    cfg.planner.randomLayout = configName == "no_m2"

    return cfg


#  Per-run result


@dataclass
class RunResult:
    prompt: str
    config: str
    success: bool
    error: str = ""
    latencyS: float = 0.0
    # Motion metrics (from MotionClip.smplx_params if available)
    nClips: int = 0
    totalFrames: int = 0
    footSlidingMs: float = float("nan")
    groundPenCm: float = float("nan")
    validity: bool = False
    # Entity / action parsing metrics
    nEntities: int = 0
    nActions: int = 0


def extractMotionMetrics(result: dict) -> dict:
    """Pull motion quality metrics from a pipeline result dict."""
    clips = result.get("motion_clips", [])
    if not clips:
        return {}
    # motion.invoke returns dict[str, MotionClip] (actor_name -> clip).
    # Accept both shapes for backwards compatibility with older callers.
    clipList = list(clips.values()) if isinstance(clips, dict) else list(clips)

    allParams = [c.smplxParams for c in clipList if c.smplxParams is not None]
    if not allParams:
        return {}

    clipMetrics = []
    for i, params in enumerate(allParams):
        try:
            clipMetrics.append(computeClipMetrics(f"clip_{i}", params))
        except Exception as e:
            log.debug("metrics failed for clip %d: %s", i, e)

    if not clipMetrics:
        return {}

    summary = summarise(clipMetrics, label="")
    return {
        "n_clips": len(clips),
        "total_frames": sum(len(p) for p in allParams),
        "foot_sliding_ms": summary["foot_sliding_mean_ms"],
        "ground_pen_cm": summary["ground_penetration_mean_cm"],
        "validity": summary["validity_rate"] > 0.5,
    }


def runSingle(
    prompt: str, configName: str, outputDir: str, duration: float, fps: int,
    device: str = "cpu",
) -> RunResult:
    """Run pipeline for one prompt/config combination."""
    run = RunResult(prompt=prompt, config=configName, success=False)
    t0 = time.perf_counter()
    try:
        cfg = pipelineConfigFor(configName, outputDir=outputDir, duration=duration,
                                fps=fps, device=device)
        safeName = prompt[:40].replace(" ", "_").replace("/", "-")
        outputName = f"{configName}__{safeName}"

        pipeline = Pipeline(cfg)
        result = pipeline.run(prompt, outputName=outputName)

        run.success = True
        run.latencyS = time.perf_counter() - t0

        # Parsing metrics
        parsed = result.get("parsed_scene")
        if parsed:
            run.nEntities = len(getattr(parsed, "entities", []))
            run.nActions = len(getattr(parsed, "actions", []))

        # Motion quality metrics
        m = extractMotionMetrics(result)
        if m:
            run.nClips = m.get("n_clips", 0)
            run.totalFrames = m.get("total_frames", 0)
            run.footSlidingMs = m.get("foot_sliding_ms", float("nan"))
            run.groundPenCm = m.get("ground_pen_cm", float("nan"))
            run.validity = m.get("validity", False)

    except Exception:
        run.latencyS = time.perf_counter() - t0
        run.error = traceback.format_exc(limit=5)
        log.warning("[%s] FAILED: %r - %s", configName, prompt[:40], run.error.splitlines()[-1])

    return run


#  Aggregation


def aggregate(results: list[RunResult]) -> dict:
    """Compute mean +/- std for all numeric metrics across a list of results."""
    successful = [r for r in results if r.success]
    n = len(results)
    ns = len(successful)

    def meanStd(vals):
        arr = np.array([v for v in vals if np.isfinite(v)])
        if len(arr) == 0:
            return float("nan"), float("nan")
        return float(arr.mean()), float(arr.std())

    foot_mean, foot_std = meanStd([r.footSlidingMs for r in successful])
    gp_mean, gp_std = meanStd([r.groundPenCm for r in successful])
    latency_mean, _ = meanStd([r.latencyS for r in successful])
    validityRate = float(sum(r.validity for r in successful) / max(ns, 1))

    return {
        "n_total": n,
        "n_success": ns,
        "success_rate": ns / max(n, 1),
        "validity_rate": validityRate,
        "foot_sliding_mean_ms": foot_mean,
        "foot_sliding_std_ms": foot_std,
        "ground_pen_mean_cm": gp_mean,
        "ground_pen_std_cm": gp_std,
        "mean_latency_s": latency_mean,
    }


#  Save helpers


def saveCsv(allResults: list[RunResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(allResults[0]).keys()))
        writer.writeheader()
        for r in allResults:
            writer.writerow(asdict(r))
    log.info("CSV saved to %s", path)


def saveSummary(summaryByConfig: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(summaryByConfig, f, indent=2)
    log.info("Summary JSON saved to %s", path)


def printTable(summaryByConfig: dict) -> None:
    header = (
        f"{'Config':<20} {'Success':>8} {'Valid%':>8} {'FtSlide':>10} "
        f"{'GndPen':>10} {'Latency':>10}"
    )
    print(f"\n{'' * len(header)}")
    print(header)
    print(f"{'' * len(header)}")
    for cfg, s in summaryByConfig.items():
        foot = (
            f"{s['foot_sliding_mean_ms']:.4f}" if np.isfinite(s["foot_sliding_mean_ms"]) else "N/A"
        )
        gp = f"{s['ground_pen_mean_cm']:.3f}" if np.isfinite(s["ground_pen_mean_cm"]) else "N/A"
        lat = f"{s['mean_latency_s']:.1f}s" if np.isfinite(s["mean_latency_s"]) else "N/A"
        print(
            f"  {cfg:<18} {s['success_rate'] * 100:>7.1f}%  {s['validity_rate'] * 100:>7.1f}%  "
            f"{foot:>10}  {gp:>10}  {lat:>10}"
        )
    print(f"{'' * len(header)}\n")


#  Main


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Run dissertation ablation study")
    p.add_argument(
        "--output", default="results/ablation", help="Output directory for CSV/JSON results"
    )
    p.add_argument(
        "--n-prompts", type=int, default=50, help="Number of prompts to evaluate (default: all 50)"
    , dest="nPrompts")
    p.add_argument(
        "--configs",
        nargs="+",
        default=ALL_CONFIGS,
        choices=ALL_CONFIGS,
        help="Which configs to run",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Clip duration per run (shorter = faster evaluation)",
    )
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"],
                   help="Device for motion generation (M4). Render still uses GPU via aitviewer.")
    p.add_argument("--resume", action="store_true", help="Skip runs that already have output files")
    p.add_argument(
        "--ood",
        action="store_true",
        help="Run OOD generalisation set (20 prompts with out-of-distribution phrasing) "
             "instead of the standard 50-prompt eval suite.",
    )
    args = p.parse_args()

    if args.ood:
        prompts = OOD_PROMPTS
        log.info("OOD mode: using %d out-of-distribution prompts", len(prompts))
    else:
        prompts = EVAL_PROMPTS[: args.nPrompts]
    Path(args.output).mkdir(parents=True, exist_ok=True)

    log.info(
        "Ablation: %d prompts x %d configs = %d runs",
        len(prompts),
        len(args.configs),
        len(prompts) * len(args.configs),
    )

    allResults: list[RunResult] = []
    done = 0
    total = len(prompts) * len(args.configs)

    for config_name in args.configs:
        configDir = os.path.join(args.output, config_name)
        Path(configDir).mkdir(parents=True, exist_ok=True)
        configResults: list[RunResult] = []

        for i, prompt in enumerate(prompts):
            done += 1
            log.info("[%d/%d] config=%s  prompt=%r", done, total, config_name, prompt[:50])
            run = runSingle(prompt, config_name, configDir, args.duration, args.fps,
                            device=args.device)
            configResults.append(run)
            allResults.append(run)

            # Save incremental CSV after each run (safe against crashes)
            saveCsv(allResults, os.path.join(args.output, "ablation_results.csv"))

    # Summary per config
    summary: dict[str, dict] = {}
    for config_name in args.configs:
        configRuns = [r for r in allResults if r.config == config_name]
        summary[config_name] = aggregate(configRuns)

    saveSummary(summary, os.path.join(args.output, "ablation_summary.json"))
    printTable(summary)

    log.info("Ablation complete. Results in %s", args.output)


if __name__ == "__main__":
    main()
