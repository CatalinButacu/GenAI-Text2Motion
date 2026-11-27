# .\.venv\Scripts\python.exe scripts/qa/long_stream_probe.py --backbone mamba --ckpt runs/checkpoints/generator/generator_mamba_bs8_e60.pt
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from text2motion.app.runtime.context import ApplicationBootstrap
from text2motion.app.runtime.run_log import log_metrics, start_run
from text2motion.evaluation.contracts import R_PRECISION_POOL
from text2motion.evaluation.embeddings import compute_guo_text_embeddings
from text2motion.evaluation.metrics import (
    jerk_profile,
    motion_quality_metrics,
    own_caption_ranks,
    r_precision,
)
from text2motion.evaluation.transition.statistics import transition_statistics
from text2motion.generation.contracts import (
    Backbone,
    SamplingConfig,
    StreamAnchors,
    StudioSamplingConfig,
)
from text2motion.generation.model.sampling import seeded_sampling_uniforms
from text2motion.motion.contracts import Split
from text2motion.motion.recovery import recover_skeleton
from text2motion.motion.representation import DIM, FPS
from text2motion.motion.storage import FEATURES_DIR, atomic_numpy_save, parse_text_file, sha256_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryPolicy:
    name: str
    anchor_every: int
    anchor_context: int


@dataclass(frozen=True)
class ProbePrompt:
    clip_id: str
    source_family: str
    caption: str
    tokens: tuple[str, ...]


POLICIES = (
    MemoryPolicy("full_state", 0, 0),
    MemoryPolicy("reset_32_keep_16", 32, 16),
    MemoryPolicy("reset_8_keep_16", 8, 16),
    MemoryPolicy("reset_4_keep_16", 4, 16),
    MemoryPolicy("queue_keep_16", 1, 16),
    MemoryPolicy("queue_keep_47", 1, 47),
)
STATISTIC_KEYS = (
    "foot_skate_m_per_s",
    "mean_acceleration_cm_per_frame2",
    "mean_jerk_cm_per_frame3",
    "root_teleports",
)
TRANSITION_METHODS = ("cut", "linear", "minimum_jerk", "generator")
SWITCH_WINDOWS = (("full", 192, 384), ("early", 192, 288), ("late", 288, 384))
SWITCH_CONTEXTS = (16, 8, 0)
COUNTERFACTUAL_SCENARIOS = ("A_to_A", "A_to_B", "B_to_A")


def scenario_selection(selection: list[ProbePrompt], scenario: str) -> list[ProbePrompt]:
    if scenario not in COUNTERFACTUAL_SCENARIOS or len(selection) != 2 * R_PRECISION_POOL:
        raise ValueError("counterfactual selection requires one original32-pair cohort")
    left, right = selection[:R_PRECISION_POOL], selection[R_PRECISION_POOL:]
    if scenario == "A_to_A":
        return left + left
    if scenario == "B_to_A":
        return right + left
    return list(selection)


def review_selection(rows: list[dict]) -> list[dict]:
    rng = np.random.default_rng(2026)
    chosen = {}
    for offset in (0, 64):
        indices = rng.choice(R_PRECISION_POOL, size=8, replace=False)
        for row in rows:
            if row["offset"] == offset and row["pair"] in indices:
                chosen[(row["backbone"], offset, row["pair"])] = {"row": row, "group": "random"}
    for backbone in ("mamba", "transformer"):
        for offset in (0, 64):
            group = [r for r in rows if r["backbone"] == backbone and r["offset"] == offset]
            for metric, descending in (
                ("root_step_delta_m", True),
                ("skate_delta_m_per_s", True),
                ("early_margin_delta", False),
            ):
                valid = [r for r in group if r[metric] is not None]
                for row in sorted(valid, key=lambda r: r[metric], reverse=descending)[:2]:
                    key = (backbone, offset, row["pair"])
                    if key not in chosen:
                        chosen[key] = {"row": row, "group": "outlier"}
    return list(chosen.values())


def review_distributions(rows: list[dict]) -> dict:
    distributions = {}
    for backbone in ("mamba", "transformer"):
        distributions[backbone] = {}
        for offset in (0, 64):
            group = [r for r in rows if r["backbone"] == backbone and r["offset"] == offset]
            statistics = {}
            for metric in (
                "root_step_delta_m",
                "skate_delta_m_per_s",
                "early_margin_delta",
                "early_r1_delta",
            ):
                values = np.asarray([r[metric] for r in group if r[metric] is not None])
                if not values.size:
                    logger.warning("%s/%s: %s has no eligible pairs", backbone, offset, metric)
                    statistics[metric] = {"valid_pairs": 0, "quantiles": None}
                else:
                    statistics[metric] = {
                        "valid_pairs": len(values),
                        "quantiles": np.quantile(values, (0, 0.25, 0.5, 0.75, 0.9, 1)).tolist(),
                        "paired_mean": paired_change(np.stack((np.zeros_like(values), values))),
                    }
            distributions[backbone][str(offset)] = statistics
    return distributions


def review_audit(source: Path, out_path: Path, summary_csv: Path) -> None:
    started = time.perf_counter()
    folder = out_path.with_suffix("")
    if any(p.exists() for p in (out_path, folder, summary_csv)):
        raise FileExistsError("review audit requires fresh outputs")
    report = json.loads(source.read_text(encoding="utf-8"))
    verification = source.with_name("switch_history_verification_20261108.json")
    checked = json.loads(verification.read_text(encoding="utf-8"))
    if checked["status"] != "passed" or checked["source_sha256"] != sha256_file(source):
        raise ValueError("review source does not match its independent audit")
    rows, archives = [], {}
    for backbone, cohorts in report["cases"].items():
        for offset, variants in cohorts.items():
            motions = {}
            for keep in ("16", "0"):
                path = Path(variants[keep]["path"]).with_suffix("") / "generator_joints.npy"
                motions[keep] = np.load(path, allow_pickle=False)
                if motions[keep].shape != (32, 384, 22, 3) or not np.isfinite(motions[keep]).all():
                    raise ValueError(f"{path}: expected finite canonical joints")
                archives[(backbone, int(offset), keep)] = path
            for pair in range(R_PRECISION_POOL):
                base, candidate = variants["16"], variants["0"]
                row = {"backbone": backbone, "offset": int(offset), "pair": pair}
                for keep, case in (("16", base), ("0", candidate)):
                    diagnostics = case["report"]["per_pair_diagnostics"]["generator"][pair]
                    row[f"skate_{keep}_m_per_s"] = diagnostics["foot_skate_m_per_s"]
                    row[f"contact_{keep}"] = diagnostics["foot_contact_fraction"]
                    row[f"raised_feet_{keep}"] = diagnostics["all_feet_above_floor_10cm_fraction"]
                    row[f"penetration_{keep}"] = diagnostics["feet_below_floor_2cm_fraction"]
                    row[f"bone_max_{keep}"] = diagnostics["bone_relative_deviation_max"]
                    joints = motions[keep][pair]
                    row[f"root_step_{keep}_m"] = float(
                        np.linalg.norm(joints[192, 0].astype(float) - joints[191, 0])
                    )
                    row[f"pair_peak_jerk_{keep}"] = float(
                        np.asarray(case["report"]["jerk_profiles"]["generator"])[pair].max()
                    )
                    for window in ("early", "late"):
                        row[f"{window}_r1_{keep}"] = case["semantic"][window]["generator"][
                            "new_r_top1"
                        ][pair]
                        row[f"{window}_margin_{keep}"] = case["semantic"][window]["generator"][
                            "new_caption_margin"
                        ][pair]
                row["root_step_delta_m"] = row["root_step_0_m"] - row["root_step_16_m"]
                row["skate_delta_m_per_s"] = (
                    row["skate_0_m_per_s"] - row["skate_16_m_per_s"]
                    if row["skate_0_m_per_s"] is not None and row["skate_16_m_per_s"] is not None
                    else None
                )
                row["early_margin_delta"] = row["early_margin_0"] - row["early_margin_16"]
                row["early_r1_delta"] = row["early_r1_0"] - row["early_r1_16"]
                rows.append(row)
    if len(rows) != 128:
        raise ValueError("review audit requires both32-pair cohorts on both twins")
    selected = review_selection(rows)
    rng = np.random.default_rng(2026)
    manifest, mapping, distributions = [], [], {}
    for number, selected_case in enumerate(selected, 1):
        row = selected_case["row"]
        backbone, offset, pair = row["backbone"], row["offset"], row["pair"]
        case_id = f"case_{number:03}"
        order = ("16", "0") if rng.integers(2) else ("0", "16")
        paths, hashes = {}, {}
        for label, keep in zip(("A", "B"), order, strict=True):
            path = folder / f"{case_id}_{label}.npy"
            array = np.load(archives[(backbone, offset, keep)], mmap_mode="r", allow_pickle=False)[
                pair
            ]
            atomic_numpy_save(array, path)
            paths[label], hashes[label] = str(path), sha256_file(path)
        captions = report["cases"][backbone][str(offset)]["16"]["report"]["pairs"][pair]
        manifest.append(
            {
                "case": case_id,
                "backbone": backbone,
                "offset": offset,
                "pair": pair,
                "group": selected_case["group"],
                "panels": paths,
                "panel_sha256": hashes,
                "old_caption": captions["left"]["caption"],
                "new_caption": captions["right"]["caption"],
            }
        )
        mapping.append({"case": case_id, "A": order[0], "B": order[1]})
    distributions = review_distributions(rows)
    result = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "seed": 2026,
        "fps": FPS,
        "cases": manifest,
        "distributions": distributions,
        "wall_seconds": time.perf_counter() - started,
        "limitations": "No human judgments. Proxies and distributions are not naturalness acceptance. "
        "Random16 shared pairs balanced across cohorts; outliers are separate purposive selection. "
        "Panel mapping is sealed separately. Pair peak jerk is not cohort Peak Jerk or AUJ.",
    }
    (folder / "sealed_mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote128-pair diagnostic audit and%d blinded review cases", len(manifest))


def raw_motion(resources, clip_id: str) -> np.ndarray:
    path = resources.motion_dataset_dir / FEATURES_DIR / f"{clip_id}.npy"
    features = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        features.ndim != 2
        or features.shape[1] != DIM
        or len(features) < 4
        or not np.issubdtype(features.dtype, np.floating)
        or not np.isfinite(features).all()
    ):
        raise ValueError(f"{path}: expected finite raw HumanML3D-263 frames")
    return features


def test_prompts(resources, count: int, split: Split = Split.TEST) -> list[tuple[str, list[str]]]:
    clip_ids = sorted(resources.clip_ids(split))
    random.Random(2026).shuffle(clip_ids)
    prompts: list[tuple[str, list[str]]] = []
    for clip_id in clip_ids:
        text_path = resources.annotation_dir / f"{clip_id}.txt"
        if not text_path.exists():
            raise FileNotFoundError(text_path)
        lines = text_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"{text_path}: no annotations")
        fields = lines[0].split("#")
        if len(fields) != 4 or not fields[0].strip() or not fields[1].strip():
            raise ValueError(f"{text_path}: invalid first annotation")
        prompt = (fields[0].strip(), fields[1].split())
        if prompt[0] in {caption for caption, _ in prompts}:
            continue
        prompts.append(prompt)
        if len(prompts) == count:
            return prompts
    raise ValueError(f"only {len(prompts)} captioned {split.value} clips found, {count} requested")


def family_prompts(
    resources, count: int, split: Split, index: Path, offset: int, min_frames: int = 0
) -> list[ProbePrompt]:
    families: dict[str, str] = {}
    with index.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            clip_id = Path(row["new_name"]).stem.removeprefix("M")
            source = row["source_path"].strip()
            if not source or (
                clip_id in families and families[clip_id] != str(Path(source)).casefold()
            ):
                raise ValueError(f"{index}: invalid or conflicting source for {clip_id}")
            families[clip_id] = str(Path(source)).casefold()
    ids = sorted(resources.clip_ids(split))
    training_sources = {families[clip_id] for clip_id in resources.clip_ids(Split.TRAIN)}
    random.Random(2026).shuffle(ids)
    chosen: list[ProbePrompt] = []
    used_sources: set[str] = set()
    used_captions: set[str] = set()
    for clip_id in ids:
        family = families[clip_id]
        if family in used_sources or family in training_sources:
            continue
        if min_frames and len(raw_motion(resources, clip_id)) < min_frames:
            logger.info("%s: skipped, shorter than %s reference frames", clip_id, min_frames)
            continue
        annotations = parse_text_file(resources.annotation_dir / f"{clip_id}.txt")
        full = [item for item in annotations if item.start_time == 0 and item.end_time == 0]
        if not full:
            logger.info("%s: skipped, no whole-clip caption", clip_id)
            continue
        annotation = full[0]
        caption = annotation.caption.strip()
        if not caption or not annotation.tokens:
            raise ValueError(f"{clip_id}: empty whole-clip caption or tokens")
        if caption in used_captions:
            continue
        chosen.append(ProbePrompt(clip_id, family, caption, tuple(annotation.tokens)))
        used_sources.add(family)
        used_captions.add(caption)
        if len(chosen) == offset + count:
            return chosen[offset:]
    raise ValueError(
        f"only {len(chosen)} distinct held-out source recordings; need {offset + count}"
    )


def bridge(cut: np.ndarray, boundary: int, frames: int, method: str) -> np.ndarray:
    jerk_profile(cut)
    if method not in ("cut", "linear", "minimum_jerk"):
        raise ValueError(f"unknown positional bridge: {method}")
    if frames < 2 or frames % 2 or boundary <= frames // 2 or boundary + frames // 2 >= len(cut):
        raise ValueError("an even bridge must fit strictly inside the motion")
    output = cut.copy()
    if method == "cut":
        return output
    start, stop = boundary - frames // 2, boundary + frames // 2
    weight = np.linspace(0.0, 1.0, frames + 2)[1:-1]
    if method == "minimum_jerk":
        weight = 10 * weight**3 - 15 * weight**4 + 6 * weight**5
    output[start:stop] = cut[start - 1] + weight[:, None, None] * (cut[stop] - cut[start - 1])
    return output


@torch.no_grad()
def transition_generate(
    pipeline,
    resources,
    pairs,
    tokens: int,
    batch_size: int,
    switch_context: int = 16,
    sampling_pairs=None,
    evidence_path: Path | None = None,
):
    features: dict[str, list[np.ndarray]] = {"cut": [], "generator": []}
    identities = pairs if sampling_pairs is None else sampling_pairs
    if len(identities) != len(pairs):
        raise ValueError("sampling identities must match the generated caption pairs")
    evidence: dict[str, list[np.ndarray]] = {"uniforms": [], "indices": [], "right_indices": []}
    model = pipeline.token_generator
    sampling = StudioSamplingConfig()
    started = time.perf_counter()
    for start in range(0, len(pairs), batch_size):
        group = pairs[start : start + batch_size]
        text = [
            pipeline.text_encoder([pair[action].caption for pair in group]) for action in (0, 1)
        ]
        seeds = [
            zlib.crc32(f"2026:{left.caption}:{right.caption}".encode("utf-8"))
            for left, right in identities[start : start + batch_size]
        ]
        uniforms = seeded_sampling_uniforms(
            seeds, tokens * 2, model.cfg.num_codebooks, resources.device
        )
        kwargs = {
            "sampling": SamplingConfig(
                temperature=sampling.temperature,
                top_p=sampling.top_p,
                cfg_scale=sampling.cfg_scale,
                stop_at_end=False,
            ),
        }
        continuous = torch.stack(
            list(
                model.stream_token_indices(
                    text[0],
                    tokens * 2,
                    uniforms=uniforms,
                    anchors=StreamAnchors(
                        every=8,
                        context=16,
                        text_at=lambda index, text=text: text[int(index >= tokens)],
                        context_at=None
                        if switch_context == 16
                        else lambda index: switch_context if index == tokens else 16,
                    ),
                    **kwargs,
                )
            ),
            dim=1,
        )
        right = torch.stack(
            list(
                model.stream_token_indices(
                    text[1],
                    tokens,
                    uniforms=uniforms[:, tokens:],
                    anchors=StreamAnchors(every=8, context=16),
                    **kwargs,
                )
            ),
            dim=1,
        )
        if evidence_path is not None:
            for name, tensor in (
                ("uniforms", uniforms),
                ("indices", continuous),
                ("right_indices", right),
            ):
                evidence[name].append(tensor.cpu().numpy())
        decoded = {
            "generator": pipeline.tokenizer.network.decode(continuous),
            "cut": torch.cat(
                (
                    pipeline.tokenizer.network.decode(continuous[:, :tokens]),
                    pipeline.tokenizer.network.decode(right),
                ),
                dim=1,
            ),
        }
        for method, tensor in decoded.items():
            array = tensor.cpu().numpy()
            expected = (len(group), tokens * 2 * pipeline.downsample, DIM)
            if array.shape != expected or not np.isfinite(array).all():
                raise RuntimeError(
                    f"{method}: expected finite motion at {expected}, got {array.shape}"
                )
            features[method].extend(np.asarray(resources.denormalize(clip)) for clip in array)
        logger.info("transition generation: %s/%s caption pairs", start + len(group), len(pairs))
    if evidence_path is not None:
        for name, arrays in evidence.items():
            atomic_numpy_save(np.concatenate(arrays), evidence_path / f"{name}.npy")
    return features, time.perf_counter() - started


def transition_scores(profiles: dict[str, np.ndarray], reference: np.ndarray) -> tuple[dict, dict]:
    if (
        reference.ndim != 3
        or reference.shape[1] != 2
        or not len(reference)
        or not reference.shape[2]
    ):
        raise ValueError("reference jerk must have shape (pairs, 2, frames)")
    if not np.isfinite(reference).all() or (reference < 0).any() or "cut" not in profiles:
        raise ValueError("finite reference jerk and a cut baseline are required")
    count = len(reference)
    draws = np.random.default_rng(2026).integers(count, size=(1000, count))
    gt = reference.mean(axis=(1, 2))
    gt_boot = gt[draws].mean(axis=1)
    summary, bootstrap, contrasts = {}, {}, {}
    for method, trace in profiles.items():
        if (
            trace.shape != (count, reference.shape[2])
            or not np.isfinite(trace).all()
            or (trace < 0).any()
        ):
            raise ValueError(f"{method}: jerk profiles do not match reference pairs and frames")
        mean = trace.mean(axis=0)
        boot = trace[draws].mean(axis=1)
        bootstrap[method] = {
            "peak_jerk": boot.max(axis=1),
            "auj": np.abs(boot - gt_boot[:, None]).sum(axis=1),
        }
        points = {
            "peak_jerk": float(mean.max()),
            "auj": float(np.abs(mean - gt.mean()).sum()),
        }
        summary[method] = {}
        for metric, value in points.items():
            low, high = np.quantile(bootstrap[method][metric], (0.025, 0.975))
            summary[method][metric] = {
                "value": value,
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
    for method in profiles:
        if method == "cut":
            continue
        contrasts[method] = {}
        for metric in ("peak_jerk", "auj"):
            low, high = np.quantile(
                bootstrap[method][metric] - bootstrap["cut"][metric], (0.025, 0.975)
            )
            contrasts[method][metric] = {
                "difference": summary[method][metric]["value"] - summary["cut"][metric]["value"],
                "ci95_low": float(low),
                "ci95_high": float(high),
            }
    return summary, contrasts


def transition_retrieval(features, evaluator, pairs, frames: int) -> dict:
    resources = evaluator.resources
    result = {}
    for action in (0, 1):
        text = compute_guo_text_embeddings(
            resources.text_embedder,
            [resources.build_guo_text_inputs(list(pair[action].tokens)) for pair in pairs],
            resources.device,
        )
        result[str(action + 1)] = {}
        for method, clips in features.items():
            motion = evaluator.compute_motion_embeddings(
                [clip[action * frames : (action + 1) * frames] for clip in clips]
            )
            hits = own_caption_ranks(text, motion, np.arange(len(pairs))) == 0
            result[str(action + 1)][method] = {
                "r_top1": float(hits.mean()),
                "text_distance": float(np.linalg.norm(motion - text, axis=-1).mean()),
                "per_pair_hits": hits.astype(int).tolist(),
            }
    return result


def switch_metrics(text: np.ndarray, motion: np.ndarray) -> dict[str, np.ndarray]:
    if (
        text.ndim != 3
        or text.shape[:2] != (2, R_PRECISION_POOL)
        or motion.shape != text.shape[1:]
        or not np.isfinite(text).all()
        or not np.isfinite(motion).all()
    ):
        raise ValueError("caption switching needs finite old/new embeddings in one 32-pair pool")
    old_distance, new_distance = np.linalg.norm(text - motion[None], axis=-1)
    margin = old_distance - new_distance
    return {
        "new_r_top1": (own_caption_ranks(text[1], motion, np.arange(R_PRECISION_POOL)) == 0).astype(
            float
        ),
        "old_r_top1": (own_caption_ranks(text[0], motion, np.arange(R_PRECISION_POOL)) == 0).astype(
            float
        ),
        "new_text_distance": new_distance,
        "old_text_distance": old_distance,
        "new_caption_margin": margin,
        "new_caption_preferred": (margin > 0).astype(float),
        "caption_distance_tie": (margin == 0).astype(float),
    }


def switch_sources(paths: list[Path]) -> list[dict]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if [report["backbone"] for report in reports] != ["mamba", "transformer"]:
        raise ValueError("pass Mamba then Transformer R6 reports")
    for key in (
        "pairs",
        "config",
        "config_sha256",
        "tokenizer_sha256",
        "split",
        "seed",
        "precision",
        "sampling",
        "fps",
        "action_frames",
        "batch_size",
        "anchor_every",
        "anchor_context",
    ):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f"caption-switch sources disagree on {key}")
    first = reports[0]
    if first["split"] != "val" or first["seed"] != 2026 or first["fps"] != FPS:
        raise ValueError("caption-switch sources must use validation, seed2026 and canonical FPS")
    if first["action_frames"] != 192 or len(first["pairs"]) != R_PRECISION_POOL:
        raise ValueError("caption-switch sources need 32 pairs with 192 frames per action")
    for report in reports:
        for key in ("checkpoint", "tokenizer_checkpoint", "config"):
            hash_key = "tokenizer_sha256" if key == "tokenizer_checkpoint" else f"{key}_sha256"
            if sha256_file(report[key]) != report[hash_key]:
                raise ValueError(f"{report['backbone']}: changed {key} provenance")
    return reports


def switch_contrasts(measured: dict) -> dict:
    contrasts = {}
    for backbone, windows in measured.items():
        for window, methods in windows.items():
            contrasts[f"{backbone}/{window}/generator_minus_cut"] = {
                metric: paired_change(np.stack((methods["cut"][metric], values)))
                for metric, values in methods["generator"].items()
            }
        for method, scores in windows["late"].items():
            contrasts[f"{backbone}/{method}/late_minus_early"] = {
                metric: paired_change(np.stack((windows["early"][method][metric], values)))
                for metric, values in scores.items()
            }
    return contrasts


def score_switch_archive(source: Path, report: dict, evaluator, text: np.ndarray):
    measured, embeddings, hashes = {}, [], {}
    features = {}
    for method in ("cut", "generator"):
        path = source.with_suffix("") / f"{method}_features.npy"
        array = np.load(path, allow_pickle=False)
        if (
            array.shape != (R_PRECISION_POOL, 384, DIM)
            or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"{path}: expected finite raw features at (32, 384, 263)")
        features[method], hashes[method] = array, sha256_file(path)
    for window, start, stop in SWITCH_WINDOWS:
        measured[window], window_embeddings = {}, []
        for method, clips in features.items():
            motion = evaluator.compute_motion_embeddings([clip[start:stop] for clip in clips])
            scores = switch_metrics(text, motion)
            if window == "full":
                previous = report["retrieval"]["2"][method]
                if not np.array_equal(
                    scores["new_r_top1"], previous["per_pair_hits"]
                ) or not np.isclose(
                    scores["new_text_distance"].mean(),
                    previous["text_distance"],
                    atol=1e-5,
                    rtol=1e-5,
                ):
                    raise RuntimeError(
                        f"{source}/{method}: full-action R6 retrieval not reproduced"
                    )
            measured[window][method] = scores
            window_embeddings.append(motion)
        embeddings.append(window_embeddings)
        logger.info("%s: scored %s action2 window (%s:%s)", report["backbone"], window, start, stop)
    return measured, np.asarray(embeddings), hashes


def caption_switch(paths: list[Path], out_path: Path, summary_csv: Path, index: Path) -> None:
    folder = out_path.with_suffix("")
    if any(path.exists() for path in (out_path, summary_csv, folder)):
        raise FileExistsError("caption-switch outputs already exist; choose new paths")
    reports = switch_sources(paths)
    app = ApplicationBootstrap.from_config(reports[0]["config"], seed=2026)
    if app.config.precision.value != reports[0]["precision"]:
        raise ValueError("scoring precision differs from R6")
    matcher = app.config.paths.eval_matcher
    if matcher is None:
        raise ValueError("caption-switch scoring requires the frozen Guo matcher")
    evaluator = app.create_humanml3d_generation_evaluator()
    resources = evaluator.resources
    selection = family_prompts(resources, R_PRECISION_POOL * 2, Split.VALIDATION, index, 0, 60)
    expected = [
        {"left": asdict(left), "right": asdict(right)}
        for left, right in zip(
            selection[:R_PRECISION_POOL], selection[R_PRECISION_POOL:], strict=True
        )
    ]
    if json.loads(json.dumps(expected)) != reports[0]["pairs"]:
        raise ValueError("R6 caption pairs no longer match official recording-disjoint selection")
    text = np.stack(
        [
            compute_guo_text_embeddings(
                resources.text_embedder,
                [
                    resources.build_guo_text_inputs(list(pair[side]["tokens"]))
                    for pair in reports[0]["pairs"]
                ],
                resources.device,
            )
            for side in ("left", "right")
        ]
    )
    measured, embeddings, hashes, rows = {}, [], {}, []
    for source, report in zip(paths, reports, strict=True):
        backbone = report["backbone"]
        scores, motion, hashes[backbone] = score_switch_archive(source, report, evaluator, text)
        measured[backbone] = scores
        embeddings.append(motion)
        for window, start, stop in SWITCH_WINDOWS:
            for method, metrics in scores[window].items():
                row = {
                    "backbone": backbone,
                    "window": window,
                    "method": method,
                    "pairs": R_PRECISION_POOL,
                    "split": "val",
                    "seed": 2026,
                    "precision": app.config.precision.value,
                    "fps": FPS,
                    "frame_start": start,
                    "frame_stop": stop,
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "config_sha256": report["config_sha256"],
                }
                for metric, values in metrics.items():
                    estimate = paired_change(np.stack((np.zeros_like(values), values)))
                    row.update(
                        {
                            metric: float(values.mean()),
                            f"{metric}_ci95_low": estimate["ci95_low"],
                            f"{metric}_ci95_high": estimate["ci95_high"],
                        }
                    )
                rows.append(row)
    atomic_numpy_save(text, folder / "text_embeddings.npy")
    atomic_numpy_save(np.asarray(embeddings), folder / "motion_embeddings.npy")
    result = {
        "sources": [str(path) for path in paths],
        "source_sha256": [sha256_file(path) for path in paths],
        "feature_sha256": hashes,
        "pairs": reports[0]["pairs"],
        "split": "val",
        "seed": 2026,
        "generation_precision": reports[0]["precision"],
        "scoring_precision": app.config.precision.value,
        "device": app.device,
        "fps": FPS,
        "matcher": str(matcher),
        "matcher_sha256": sha256_file(matcher),
        "family_index_sha256": sha256_file(index),
        "windows": [
            {
                "name": name,
                "frame_start": start,
                "frame_stop": stop,
                "seconds": (stop - start) / FPS,
            }
            for name, start, stop in SWITCH_WINDOWS
        ],
        "embedding_order": {
            "backbone": ["mamba", "transformer"],
            "window": [name for name, _, _ in SWITCH_WINDOWS],
            "method": ["cut", "generator"],
            "text": ["old", "new"],
        },
        "summary": rows,
        "contrasts": switch_contrasts(measured),
        "per_pair": {
            b: {
                w: {m: {k: v.tolist() for k, v in scores.items()} for m, scores in methods.items()}
                for w, methods in windows.items()
            }
            for b, windows in measured.items()
        },
        "bootstrap_resamples": 1000,
        "limitations": "One fixed32-pair validation cohort, not a generalization or training-seed test. "
        "Exploratory intervals are unadjusted for multiple comparisons. "
        "Intervals condition on fixed distractor pools. Positive old-minus-new distance margin "
        "favors the new caption; ties favor neither. Caption phases and overlapping semantics can "
        "confound early/late comparisons. Independent windows are matched controls. No generator "
        "or tokenizer was loaded. Embedding archives permit independent score verification.",
    }
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote caption-switch evidence %s and %s", out_path, summary_csv)


def transition_screen(args, app, pipeline, evaluator, selection) -> dict:
    resources = evaluator.resources
    pairs = list(zip(selection[: args.prompts], selection[args.prompts :], strict=True))
    frames = args.segment_tokens * pipeline.downsample
    half = args.boundary_frames // 2
    if frames <= half:
        raise ValueError("the boundary window must leave reference frames outside each action")
    features, seconds = transition_generate(
        pipeline,
        resources,
        pairs,
        args.segment_tokens,
        args.batch_size,
        args.switch_context,
        sampling_pairs=getattr(args, "sampling_pairs", None),
        evidence_path=args.out_path.with_suffix("")
        if getattr(args, "sampling_pairs", None) is not None
        else None,
    )
    reference = []
    for pair in pairs:
        profiles = []
        for sample in pair:
            joints = recover_skeleton(raw_motion(resources, sample.clip_id).copy())
            center = len(joints) // 2
            profiles.append(jerk_profile(joints[center - half : center + half]))
        reference.append(profiles)
    reference = np.asarray(reference)
    folder = args.out_path.with_suffix("")
    motion_files, profiles, diagnostics = {}, {}, {}
    raw_joints = {
        method: [recover_skeleton(clip) for clip in clips] for method, clips in features.items()
    }
    for method in TRANSITION_METHODS:
        joints = (
            raw_joints["generator"]
            if method == "generator"
            else [bridge(clip, frames, args.blend_frames, method) for clip in raw_joints["cut"]]
        )
        path = folder / f"{method}_joints.npy"
        atomic_numpy_save(np.stack(joints), path)
        motion_files[method] = str(path)
        profiles[method] = np.stack(
            [jerk_profile(clip[frames - half : frames + half]) for clip in joints]
        )
        diagnostics[method] = [transition_statistics(clip, frames, half) for clip in joints]
    for method, clips in features.items():
        atomic_numpy_save(np.stack(clips), folder / f"{method}_features.npy")
    atomic_numpy_save(reference, folder / "reference_jerk.npy")
    summary, contrasts = transition_scores(profiles, reference)
    for method, values in diagnostics.items():
        summary[method]["diagnostics"] = {}
        for key in values[0]:
            measured = [item[key] for item in values]
            valid = [value for value in measured if value is not None]
            summary[method]["diagnostics"][key] = {
                "mean": float(np.mean(valid)) if valid else None,
                "valid_pairs": len(valid),
            }
    report = {
        "backbone": args.backbone.value,
        "config": args.config,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": sha256_file(args.ckpt),
        "tokenizer_checkpoint": str(args.tokenizer_ckpt),
        "tokenizer_sha256": sha256_file(args.tokenizer_ckpt),
        "config_sha256": sha256_file(args.config),
        "split": args.split.value,
        "seed": 2026,
        "precision": app.config.precision.value,
        "batch_size": args.batch_size,
        "pairs": [{"left": asdict(left), "right": asdict(right)} for left, right in pairs],
        "selection": f"{len({sample.source_family for pair in pairs for sample in pair})} "
        "validation recordings absent from TRAIN; min 60 frames; not subject-disjoint",
        "segment_tokens": args.segment_tokens,
        "action_frames": frames,
        "fps": FPS,
        "blend_frames": args.blend_frames,
        "boundary_frames": args.boundary_frames,
        "sampling": asdict(StudioSamplingConfig()),
        "anchor_every": 8,
        "anchor_context": 16,
        "switch_context": args.switch_context,
        "prompt_offset": args.prompt_offset,
        "ground_truth_jerk_reference": float(reference.mean()),
        "metric_definition": "FlowMDM L1 third difference, max over joints; PJ=max of mean "
        "boundary profile; AUJ=sum absolute deviation of that profile from cohort GT mean. "
        "Raw meters/frame^3; AUJ is a discrete sum. GT reference differs from FlowMDM's test constant.",
        "metric_source": "https://github.com/BarqueroGerman/FlowMDM/blob/main/utils/metrics.py",
        "floor_height_m": 0.0,
        "bootstrap": {"paired_resamples": 1000, "seed": 2026},
        "summary": summary,
        "paired_minus_cut": contrasts,
        "jerk_profiles": {method: trace.tolist() for method, trace in profiles.items()},
        "per_pair_diagnostics": diagnostics,
        "motion_files": motion_files,
        "retrieval": transition_retrieval(features, evaluator, pairs, frames),
        "generation_decode_seconds": seconds,
        "limitations": "Offline noncausal decode, not live latency. Positional bridges have "
        "future endpoint access and no IK. Generator is paired by captions, not endpoint poses. "
        "A zero-velocity minimum-jerk bridge need not match moving boundary derivatives. "
        "Foot slide is conditional on contact at a common Y=0 floor; no contact is null, not zero. "
        "Raised-feet/penetration fields are proxies, not proof of floating or naturalness. "
        "Retrieval only scores raw cut/generator features; bridges are not valid 263 features. "
        "One seed and fixed 32-caption pools; selected validation, not final-test evidence.",
    }
    args.out_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    logger.info("wrote transition screen %s; generation/decode %.1fs", args.out_path, seconds)
    return report


def history_contrasts(cases: dict) -> dict:
    result = {}
    for backbone, cohorts in cases.items():
        result[backbone] = {}
        for offset, variants in cohorts.items():
            reference = np.load(
                Path(variants["16"]["path"]).with_suffix("") / "reference_jerk.npy",
                allow_pickle=False,
            )
            profiles = {
                "cut" if keep == "16" else keep: np.asarray(
                    case["report"]["jerk_profiles"]["generator"]
                )
                for keep, case in variants.items()
            }
            _, physical = transition_scores(profiles, reference)
            comparisons = {}
            for keep in variants:
                if keep == "16":
                    continue
                baseline, candidate = variants["16"], variants[keep]
                semantic = {
                    window: {
                        metric: paired_change(
                            np.stack(
                                (
                                    baseline["semantic"][window]["generator"][metric],
                                    np.asarray(values),
                                )
                            )
                        )
                        for metric, values in methods["generator"].items()
                    }
                    for window, methods in candidate["semantic"].items()
                }
                pairs = [
                    (left["foot_skate_m_per_s"], right["foot_skate_m_per_s"])
                    for left, right in zip(
                        baseline["report"]["per_pair_diagnostics"]["generator"],
                        candidate["report"]["per_pair_diagnostics"]["generator"],
                        strict=True,
                    )
                    if left["foot_skate_m_per_s"] is not None
                    and right["foot_skate_m_per_s"] is not None
                ]
                skate = paired_change(np.asarray(pairs).T) if pairs else None
                if not pairs:
                    logger.warning(
                        "%s/%s/%s: no jointly contacting pairs; skate comparison unavailable",
                        backbone,
                        offset,
                        keep,
                    )
                comparisons[keep] = {
                    "semantic": semantic,
                    "physical": physical[keep],
                    "contact_skate": skate,
                    "jointly_contacting_pairs": len(pairs),
                }
            result[backbone][offset] = comparisons
    return result


def history_case(
    source: dict,
    offset: int,
    keep: int,
    path: Path,
    app,
    pipeline,
    evaluator,
    selection,
    sampling_selection=None,
    scenario: str = "A_to_B",
) -> dict:
    args = argparse.Namespace(
        config=source["config"],
        ckpt=Path(source["checkpoint"]),
        tokenizer_ckpt=Path(source["tokenizer_checkpoint"]),
        backbone=Backbone(source["backbone"]),
        prompts=R_PRECISION_POOL,
        segment_tokens=48,
        batch_size=2,
        split=Split.VALIDATION,
        boundary_frames=60,
        blend_frames=20,
        out_path=path,
        switch_context=keep,
        prompt_offset=offset,
        sampling_pairs=None
        if sampling_selection is None
        else list(
            zip(
                sampling_selection[:R_PRECISION_POOL],
                sampling_selection[R_PRECISION_POOL:],
                strict=True,
            )
        ),
    )
    report = transition_screen(args, app, pipeline, evaluator, selection)
    if sampling_selection is not None:
        report["scenario"] = scenario
        report["sampling_evidence"] = {
            name: {
                "path": str(path.with_suffix("") / f"{name}.npy"),
                "sha256": sha256_file(path.with_suffix("") / f"{name}.npy"),
            }
            for name in ("uniforms", "indices", "right_indices")
        }
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if offset == 0 and keep == 16 and scenario == "A_to_B":
        original = Path(source["motion_files"]["generator"]).parent
        for method in ("cut", "generator"):
            previous = np.load(original / f"{method}_features.npy", allow_pickle=False)
            current = np.load(path.with_suffix("") / f"{method}_features.npy", allow_pickle=False)
            if not np.array_equal(previous, current):
                raise RuntimeError(
                    f"{source['backbone']}: baseline16 no longer reproduces frozen R6 raw motion"
                )
    resources = evaluator.resources
    text = np.stack(
        [
            compute_guo_text_embeddings(
                resources.text_embedder,
                [
                    resources.build_guo_text_inputs(list(pair[side]["tokens"]))
                    for pair in report["pairs"]
                ],
                resources.device,
            )
            for side in ("left", "right")
        ]
    )
    semantic, embeddings, hashes = score_switch_archive(path, report, evaluator, text)
    atomic_numpy_save(text, path.with_suffix("") / "text_embeddings.npy")
    atomic_numpy_save(embeddings, path.with_suffix("") / "semantic_embeddings.npy")
    return {
        "path": str(path),
        "report": report,
        "feature_sha256": hashes,
        "semantic": {
            w: {m: {k: v.tolist() for k, v in scores.items()} for m, scores in methods.items()}
            for w, methods in semantic.items()
        },
    }


def counterfactual_interactions(cases: dict) -> dict:
    result = {}
    for backbone, scenarios in cases.items():
        text = np.load(
            Path(scenarios["A_to_B"]["16"]["path"]).with_suffix("") / "text_embeddings.npy",
            allow_pickle=False,
        )
        result[backbone] = {}
        for index, (window, _, _) in enumerate(SWITCH_WINDOWS):
            deltas = []
            for scenario in ("A_to_A", "A_to_B"):
                scores = []
                for keep in ("16", "0"):
                    embeddings = np.load(
                        Path(scenarios[scenario][keep]["path"]).with_suffix("")
                        / "semantic_embeddings.npy",
                        allow_pickle=False,
                    )
                    scores.append(switch_metrics(text, embeddings[index, 1]))
                deltas.append(
                    {metric: scores[1][metric] - values for metric, values in scores[0].items()}
                )
            result[backbone][window] = {
                metric: paired_change(np.stack((deltas[0][metric], values)))
                for metric, values in deltas[1].items()
            }
    return result


def history_screen(
    paths: list[Path],
    out_path: Path,
    summary_csv: Path,
    index: Path,
    counterfactual: bool = False,
) -> None:
    started = time.perf_counter()
    folder = out_path.with_suffix("")
    if any(path.exists() for path in (folder, out_path, summary_csv)):
        raise FileExistsError("history screen requires new output paths")
    sources = switch_sources(paths)
    app = ApplicationBootstrap.from_config(sources[0]["config"], seed=2026)
    tokenizer = app.models.load_tokenizer(sources[0]["tokenizer_checkpoint"])
    evaluator = app.create_humanml3d_generation_evaluator()
    selection = family_prompts(
        evaluator.resources,
        R_PRECISION_POOL * (2 if counterfactual else 4),
        Split.VALIDATION,
        index,
        0,
        60,
    )
    groups = (
        [
            (scenario, 0, scenario_selection(selection, scenario))
            for scenario in COUNTERFACTUAL_SCENARIOS
        ]
        if counterfactual
        else [
            (str(offset), offset, selection[offset : offset + R_PRECISION_POOL * 2])
            for offset in (0, R_PRECISION_POOL * 2)
        ]
    )
    contexts = (16, 0) if counterfactual else SWITCH_CONTEXTS
    cases, rows = {}, []
    for source in sources:
        backbone = source["backbone"]
        pipeline = app.models.load_text_to_motion_generator(
            Backbone(backbone),
            source["checkpoint"],
            tokenizer,
            tokenizer_checkpoint=source["tokenizer_checkpoint"],
        ).eval()
        cases[backbone] = {}
        for group, offset, cohort in groups:
            cases[backbone][group] = {}
            for keep in contexts:
                label = group if counterfactual else f"offset{offset}"
                path = folder / f"{backbone}_{label}_keep{keep}.json"
                case = history_case(
                    source,
                    offset,
                    keep,
                    path,
                    app,
                    pipeline,
                    evaluator,
                    cohort,
                    sampling_selection=selection if counterfactual else None,
                    scenario=group if counterfactual else "A_to_B",
                )
                cases[backbone][group][str(keep)] = case
                physical = case["report"]["summary"]["generator"]
                row = {
                    "backbone": backbone,
                    "offset": offset,
                    "switch_context": keep,
                    "pairs": R_PRECISION_POOL,
                    "split": "val",
                    "seed": 2026,
                    "precision": app.config.precision.value,
                    "peak_jerk": physical["peak_jerk"]["value"],
                    "auj": physical["auj"]["value"],
                    "checkpoint_sha256": case["report"]["checkpoint_sha256"],
                }
                if counterfactual:
                    row["scenario"] = group
                row.update({key: value["mean"] for key, value in physical["diagnostics"].items()})
                row["skate_valid_pairs"] = physical["diagnostics"]["foot_skate_m_per_s"][
                    "valid_pairs"
                ]
                for window in ("early", "late"):
                    row.update(
                        {
                            f"{window}_{metric}": float(np.mean(values))
                            for metric, values in case["semantic"][window]["generator"].items()
                        }
                    )
                rows.append(row)
        del pipeline
        torch.cuda.empty_cache()
    report = {
        "sources": [str(path) for path in paths],
        "source_sha256": [sha256_file(path) for path in paths],
        "seed": 2026,
        "split": "val",
        "precision": app.config.precision.value,
        "cohort_offsets": [0] if counterfactual else [0, R_PRECISION_POOL * 2],
        "switch_contexts": list(contexts),
        "cases": cases,
        "summary": rows,
        "comparisons_to_keep16": history_contrasts(cases),
        "wall_seconds": time.perf_counter() - started,
        "limitations": "Only token48 changes retained history; discarded tokens never return. Other resets use capacity16. "
        "Offline noncausal tokenizer decode; not live latency. Recording-disjoint validation selection absent TRAIN, "
        "not subjects or training seeds. Exploratory paired1000-resample intervals; no multiplicity correction. "
        "No physical acceptance threshold was selected; report trade-offs, do not adopt a policy.",
    }
    if counterfactual:
        report["scenarios"] = list(COUNTERFACTUAL_SCENARIOS)
        report["sampling_identity_pairs"] = [
            {"left": asdict(left), "right": asdict(right)}
            for left, right in zip(
                selection[:R_PRECISION_POOL], selection[R_PRECISION_POOL:], strict=True
            )
        ]
        report["switch_specific_interactions"] = counterfactual_interactions(cases)
        report["limitations"] += (
            " Original A/B sampling identities fix all uniforms across scenarios/policies/twins. "
            "A-to-A has32 distinct recordings, other scenarios64. "
            "Interactions compare A-to-B versus A-to-A policy deltas in the SAME original A/B "
            "caption pools; A-to-A own-caption margins are tautologically zero. B-to-A is a "
            "direction check only: no matched B-to-B control was run."
        )
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("history screen complete: %.1fs for12 cells", report["wall_seconds"])


def generate(
    pipeline,
    resources,
    captions: list[str],
    tokens: int,
    batch_size: int,
    policy: MemoryPolicy,
    sampling: StudioSamplingConfig,
) -> list[np.ndarray]:
    features: list[np.ndarray] = []
    model = pipeline.token_generator
    for start in range(0, len(captions), batch_size):
        group = captions[start : start + batch_size]
        condition = pipeline.text_encoder(group)
        seeds = [zlib.crc32(f"2026:{caption}".encode("utf-8")) for caption in group]
        uniforms = seeded_sampling_uniforms(
            seeds, tokens, model.cfg.num_codebooks, resources.device
        )
        steps = list(
            model.stream_token_indices(
                condition,
                tokens,
                sampling=SamplingConfig(
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    cfg_scale=sampling.cfg_scale,
                    stop_at_end=sampling.stop_at_end,
                ),
                uniforms=uniforms,
                anchors=StreamAnchors(every=policy.anchor_every, context=policy.anchor_context),
            )
        )
        indices = torch.stack(steps, dim=1)
        decoded = pipeline.tokenizer.network.decode(indices).cpu().numpy()
        if decoded.shape != (len(group), tokens * pipeline.downsample, DIM):
            raise RuntimeError(f"{policy.name}: unexpected decoded shape {decoded.shape}")
        if not np.isfinite(decoded).all():
            raise RuntimeError(f"{policy.name}: generated non-finite motion")
        features.extend(np.asarray(resources.denormalize(clip)) for clip in decoded)
        logger.info("%s: generated %s/%s prompts", policy.name, len(features), len(captions))
    return features


def score(
    features: list[np.ndarray],
    evaluator,
    text_embeddings: np.ndarray,
    segments: int,
    frames: int,
    policy: str,
    run_dir: Path,
    milliseconds: float,
) -> tuple[list[dict[str, float]], dict[str, list[list[float]]]]:
    rows: list[dict[str, float]] = []
    measured: dict[str, list[list[float]]] = {
        key: [] for key in ("text_distance", "r_top1", *STATISTIC_KEYS)
    }
    order = np.arange(len(features))
    for segment in range(segments):
        clips = [clip[segment * frames : (segment + 1) * frames] for clip in features]
        statistics = [motion_quality_metrics(recover_skeleton(clip)) for clip in clips]
        motion_embeddings = evaluator.compute_motion_embeddings(clips)
        distance = np.linalg.norm(motion_embeddings - text_embeddings, axis=-1)
        ranks = own_caption_ranks(text_embeddings, motion_embeddings, order)
        if (ranks < 0).any():
            raise RuntimeError("incomplete R-precision pool")
        top_hits = r_precision(text_embeddings, motion_embeddings, seed=2026)
        measured["text_distance"].append(distance.tolist())
        measured["r_top1"].append((ranks == 0).astype(float).tolist())
        row = {
            "segment": segment + 1,
            "seconds_from": segment * frames / FPS,
            "text_distance": float(distance.mean()),
            "r_top1": float(top_hits[0]),
            "r_top3": float(top_hits[2]),
            "ms_per_token": milliseconds,
        }
        for key in STATISTIC_KEYS:
            values = [statistic[key] for statistic in statistics]
            measured[key].append(values)
            row[key] = float(np.mean(values))
        rows.append(row)
        log_metrics(run_dir, {"policy": policy, **row})
        logger.info(
            "%s segment %s: distance %.3f R@1 %.3f skate %.3f jerk %.3f",
            policy,
            segment + 1,
            row["text_distance"],
            row["r_top1"],
            row["foot_skate_m_per_s"],
            row["mean_jerk_cm_per_frame3"],
        )
    return rows, measured


def paired_change(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("paired changes need at least two windows and one prompt")
    if not np.isfinite(values).all():
        raise ValueError("paired changes need finite values")
    difference = values[-1] - values[0]
    rng = np.random.default_rng(2026)
    draws = rng.integers(0, len(difference), size=(1000, len(difference)))
    low, high = np.quantile(difference[draws].mean(axis=1), [0.025, 0.975])
    return {
        "last_minus_first": float(difference.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def policy_fits(policy: MemoryPolicy, positions: int | None, prefix: int, tokens: int) -> bool:
    if positions is None:
        return True
    available = positions - prefix
    if policy.anchor_every == 0:
        return tokens <= available
    return policy.anchor_context + policy.anchor_every <= available


def validate_args(args: argparse.Namespace) -> None:
    if args.backbone is None or args.ckpt is None:
        raise ValueError("--backbone and --ckpt are required for generation")
    if args.prompts < R_PRECISION_POOL or args.prompts % R_PRECISION_POOL:
        raise ValueError(f"--prompts must be a positive multiple of {R_PRECISION_POOL}")
    if args.segments < 2 or args.segment_tokens < 10 or args.segment_tokens > 49:
        raise ValueError("use at least two segments, each with 10..49 tokens (40..196 frames)")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if args.split == Split.TRAIN:
        raise ValueError("use held-out val or test captions, not train")
    if args.out_path is not None and args.out_path.exists():
        raise FileExistsError(f"{args.out_path}: choose a new output path")
    if args.prompt_offset < 0 or args.prompt_offset % R_PRECISION_POOL:
        raise ValueError("--prompt_offset must be a nonnegative multiple of 32")
    if (args.prompt_offset or args.save_motion) and args.family_index is None:
        raise ValueError("--prompt_offset and --save_motion require --family_index")
    if args.save_motion and (args.out_path is None or args.out_path.with_suffix("").exists()):
        raise FileExistsError("--save_motion requires a new explicit output path")
    if args.transition_screen:
        validate_transition_args(args)


def validate_transition_args(args: argparse.Namespace) -> None:
    if args.split != Split.VALIDATION or args.family_index is None or not args.save_motion:
        raise ValueError("transition screen requires --split val, --family_index and --save_motion")
    if args.segments != 2 or args.segment_tokens % 8:
        raise ValueError("transition screen needs two actions, each a multiple of eight tokens")
    if (
        args.blend_frames < 2
        or args.blend_frames % 2
        or args.boundary_frames % 2
        or args.boundary_frames < args.blend_frames + 6
    ):
        raise ValueError("use an even bridge and an even boundary window with three margin frames")
    if args.policies is not None:
        raise ValueError("transition screen fixes reset8/keep16; omit --policies")


def policy_contrasts(left: dict, right: dict) -> dict:
    contrasts = (
        ("mamba_restart_minus_full_late", left, "reset_8_keep_16", left, "full_state"),
        ("mamba_minus_transformer_restart_late", left, "reset_8_keep_16", right, "reset_8_keep_16"),
        ("mamba_32_minus_8_late", left, "reset_32_keep_16", left, "reset_8_keep_16"),
        ("transformer_32_minus_8_late", right, "reset_32_keep_16", right, "reset_8_keep_16"),
        ("mamba_minus_transformer_32_late", left, "reset_32_keep_16", right, "reset_32_keep_16"),
    )
    comparisons = {}
    for name, candidate, candidate_policy, baseline, baseline_policy in contrasts:
        if (
            candidate_policy not in candidate["policies"]
            or baseline_policy not in baseline["policies"]
        ):
            continue
        for metric in candidate["per_prompt"][candidate_policy]:
            if metric not in baseline["per_prompt"][baseline_policy]:
                raise ValueError(f"missing baseline per-prompt metric {metric}")
            new = np.asarray(candidate["per_prompt"][candidate_policy][metric])[-1]
            old = np.asarray(baseline["per_prompt"][baseline_policy][metric])[-1]
            comparisons.setdefault(metric, {})[name] = paired_change(np.stack([old, new]))
    if not comparisons:
        raise ValueError("no supported paired contrasts in reports")
    return comparisons


def compare(paths: list[Path], out_path: Path, summary_csv: Path) -> None:
    if out_path.exists() or summary_csv.exists():
        raise FileExistsError("comparison outputs already exist; choose new paths")
    left, right = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if left["backbone"] != "mamba" or right["backbone"] != "transformer":
        raise ValueError("pass Mamba then Transformer reports")
    for key in (
        "captions",
        "prompts",
        "split",
        "seed",
        "sampling",
        "config",
        "tokenizer_checkpoint",
        "precision",
        "batch_size",
        "segment_tokens",
    ):
        if left[key] != right[key]:
            raise ValueError(f"reports disagree on {key}")
    if left["split"] != "val" or left["seed"] != 2026:
        raise ValueError("model-selection comparison requires validation and seed 2026")
    if left.get("selection") != right.get("selection"):
        raise ValueError("reports disagree on prompt provenance")
    first_policy = next(iter(left["policies"]))
    expected_times = [row["seconds_from"] for row in left["policies"][first_policy]]
    evidence = []
    for report in (left, right):
        for policy, rows in report["policies"].items():
            if [row["seconds_from"] for row in rows] != expected_times:
                raise ValueError(f"{policy}: mismatched window horizons")
            for metric in report["per_prompt"][policy]:
                values = np.asarray(report["per_prompt"][policy][metric])
                if values.shape != (len(rows), report["prompts"]) or not np.isfinite(values).all():
                    raise ValueError(f"invalid {policy}/{metric} per-prompt evidence")
                if not np.allclose(values.mean(axis=1), [row[metric] for row in rows]):
                    raise ValueError(f"aggregate mismatch for {policy}/{metric}")
            evidence.extend(
                {"backbone": report["backbone"], "policy": policy, **row} for row in rows
            )
    comparisons = policy_contrasts(left, right)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "sources": [str(path) for path in paths],
                "bootstrap_resamples": 1000,
                "seed": 2026,
                "interval_scope": "prompt pairs conditional on fixed distractor pools; not training seeds",
                "comparisons": comparisons,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evidence[0]))
        writer.writeheader()
        writer.writerows(evidence)
    logger.info(f"verified {len(evidence)} windows; comparisons {comparisons}")
    logger.info(f"wrote {out_path} and {summary_csv}")


def compare_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.out_path is None or args.summary_csv is None:
        parser.error("--compare requires --out_path and --summary_csv")
    if len(args.compare) != (1 if args.review_audit else 2):
        parser.error("--compare needs one report for review audit, otherwise two reports")
    logging.basicConfig(level=logging.INFO)
    if args.review_audit:
        review_audit(args.compare[0], args.out_path, args.summary_csv)
    elif args.caption_switch or args.history_screen or args.counterfactual:
        if args.family_index is None:
            parser.error("archived transition screens require --family_index")
        if args.counterfactual:
            history_screen(
                args.compare,
                args.out_path,
                args.summary_csv,
                args.family_index,
                counterfactual=True,
            )
        else:
            comparison = history_screen if args.history_screen else caption_switch
            comparison(args.compare, args.out_path, args.summary_csv, args.family_index)
    else:
        compare(args.compare, args.out_path, args.summary_csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generator/gen_pilot_fsq8x1024.yaml")
    parser.add_argument("--backbone", type=Backbone)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument(
        "--tokenizer_ckpt",
        type=Path,
        default=Path("runs/checkpoints/tokenizer/fsq_g8_v1024.pt"),
    )
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--segment_tokens", type=int, default=49)
    parser.add_argument("--split", type=Split, choices=list(Split), default=Split.TEST)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--policies", nargs="+", choices=[p.name for p in POLICIES])
    parser.add_argument("--out_path", type=Path)
    parser.add_argument("--compare", nargs="+", type=Path)
    parser.add_argument("--summary_csv", type=Path)
    parser.add_argument("--family_index", type=Path)
    parser.add_argument("--prompt_offset", type=int, default=0)
    parser.add_argument("--save_motion", action="store_true")
    screens = parser.add_mutually_exclusive_group()
    screens.add_argument("--transition_screen", action="store_true")
    screens.add_argument("--history_screen", action="store_true")
    screens.add_argument("--counterfactual", action="store_true")
    screens.add_argument("--review_audit", action="store_true")
    parser.add_argument("--switch_context", type=int, choices=SWITCH_CONTEXTS, default=16)
    screens.add_argument(
        "--caption_switch",
        action="store_true",
        help="Score archived R6 --compare reports without generation",
    )
    parser.add_argument("--blend_frames", type=int, default=20)
    parser.add_argument("--boundary_frames", type=int, default=60)
    args = parser.parse_args()
    if args.compare is not None:
        compare_mode(args, parser)
        return
    if args.caption_switch or args.history_screen or args.review_audit or args.counterfactual:
        parser.error("archived transition screens require --compare")
    validate_args(args)

    app = ApplicationBootstrap.from_config(args.config, seed=2026)
    tokenizer = app.models.load_tokenizer(args.tokenizer_ckpt)
    pipeline = app.models.load_text_to_motion_generator(
        args.backbone, args.ckpt, tokenizer, tokenizer_checkpoint=args.tokenizer_ckpt
    ).eval()
    evaluator = app.create_humanml3d_generation_evaluator()
    resources = evaluator.resources
    run_dir = start_run(f"long_stream_{args.backbone.value}", app.config, app.config.paths.logs_dir)

    selection = []
    if args.family_index is not None:
        selection = family_prompts(
            resources,
            args.prompts * (2 if args.transition_screen else 1),
            args.split,
            args.family_index,
            args.prompt_offset,
            args.boundary_frames if args.transition_screen else 0,
        )
        prompts = [(sample.caption, list(sample.tokens)) for sample in selection]
    else:
        prompts = test_prompts(resources, args.prompts, args.split)
    if args.transition_screen:
        transition_screen(args, app, pipeline, evaluator, selection)
        return
    captions = [caption for caption, _ in prompts]
    text_inputs = [resources.build_guo_text_inputs(tokens) for _, tokens in prompts]
    text_embeddings = compute_guo_text_embeddings(
        resources.text_embedder, text_inputs, resources.device
    )
    tokens_total = args.segments * args.segment_tokens
    token_generator = pipeline.token_generator
    sampling = StudioSamplingConfig()
    segment_frames = args.segment_tokens * pipeline.downsample
    report: dict[str, list[dict[str, float]]] = {}
    changes: dict[str, dict[str, dict[str, float]]] = {}
    per_prompt: dict[str, dict[str, list[list[float]]]] = {}
    unsupported: dict[str, str] = {}
    motion_files: dict[str, str] = {}
    costs: dict[str, float] = {}
    selected = [
        policy for policy in POLICIES if args.policies is None or policy.name in args.policies
    ]

    with torch.no_grad():
        prefix_length = token_generator.project_text_prefix(
            pipeline.text_encoder(captions[:1])
        ).size(1)
        for policy in selected:
            positions = token_generator.backbone.max_positions
            if positions is not None and not isinstance(positions, int):
                raise TypeError(f"max_positions must be an int or None, got {type(positions)}")
            if not policy_fits(policy, positions, prefix_length, tokens_total):
                logger.info(f"{policy.name}: skipped, {positions} learned positions are too few")
                unsupported[policy.name] = (
                    f"{positions} learned positions cannot fit {tokens_total} tokens"
                )
                continue
            started = time.perf_counter()
            features = generate(
                pipeline, resources, captions, tokens_total, args.batch_size, policy, sampling
            )
            costs[policy.name] = time.perf_counter() - started
            rows, measured = score(
                features,
                evaluator,
                text_embeddings,
                args.segments,
                segment_frames,
                policy.name,
                run_dir,
                costs[policy.name] * 1000 / tokens_total,
            )
            report[policy.name] = rows
            per_prompt[policy.name] = measured
            changes[policy.name] = {
                "text_distance": paired_change(np.asarray(measured["text_distance"])),
                "r_top1_fixed_pool": paired_change(np.asarray(measured["r_top1"])),
            }
            if args.save_motion:
                motion_path = args.out_path.with_suffix("") / f"{policy.name}.npy"
                atomic_numpy_save(np.stack(features), motion_path)
                motion_files[policy.name] = str(motion_path)
            logger.info(f"{policy.name}: paired last-first changes {changes[policy.name]}")

    if not report:
        raise RuntimeError(f"no supported policies: {unsupported}")
    out_path = (
        args.out_path
        or app.config.paths.outputs_dir / "qa" / f"long_stream_{args.backbone.value}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "backbone": args.backbone.value,
        "checkpoint": str(args.ckpt),
        "prompts": len(captions),
        "captions": captions,
        "split": args.split.value,
        "seed": 2026,
        "sampling": {
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "cfg_scale": sampling.cfg_scale,
            "stop_at_end": False,
        },
        "config": args.config,
        "tokenizer_checkpoint": str(args.tokenizer_ckpt),
        "precision": app.config.precision.value,
        "batch_size": args.batch_size,
        "selection": {
            "method": "source-recording-disjoint" if selection else "legacy-distinct-captions",
            "index": str(args.family_index) if args.family_index is not None else None,
            "offset": args.prompt_offset,
            "excludes_training_sources": bool(selection),
            "samples": [asdict(sample) for sample in selection],
        },
        "motion_files": motion_files,
        "motion_representation": "raw HumanML3D-263 at 20 FPS",
        "generation_decode_seconds": costs,
        "diagnostic": "Fixed-caption continuation, not standard whole-clip evaluation. "
        "R@1 intervals condition on fixed 32-motion distractor pools. "
        "Wall time includes all prompt batches and offline decoding, not streaming latency.",
        "segment_tokens": args.segment_tokens,
        "policies": report,
        "paired_changes": changes,
        "per_prompt": per_prompt,
        "unsupported": unsupported,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
