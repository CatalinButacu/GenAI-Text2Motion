# python scripts/qa/verb_clusters.py --min_clips 50 --pick 10
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from action_vocabulary import ACTION, ROOT, clip_groups
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from verb_generation_probe import CONFIG, MAX_FRAMES, MIN_FRAMES, SEED

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.evaluation.embeddings import compute_guo_text_embeddings

VOCABULARY = Path("doc/notes/verb_vocabulary.csv")
CHUNK = 512
BOTTOM_UP = "bottom_up"
TOP_DOWN = "top_down"
MOTION = "motion"
TEXT = "text"


def action_verbs() -> list[str]:
    with VOCABULARY.open(encoding="utf-8") as handle:
        return [row["verb"] for row in csv.DictReader(handle) if row["status"] == ACTION]


def clip_motion(clip_id: str) -> np.ndarray | None:
    raw = np.load(ROOT / "new_joint_vecs" / f"{clip_id}.npy").astype(np.float32)[:MAX_FRAMES]
    return raw if len(raw) >= MIN_FRAMES and np.isfinite(raw).all() else None


def clip_embeddings(evaluator, clip_ids: list[str]) -> dict[str, np.ndarray]:
    embedded: dict[str, np.ndarray] = {}
    for start in range(0, len(clip_ids), CHUNK):
        motions = {clip: clip_motion(clip) for clip in clip_ids[start : start + CHUNK]}
        kept = [clip for clip, motion in motions.items() if motion is not None]
        if not kept:
            continue
        vectors = evaluator.compute_motion_embeddings([motions[clip] for clip in kept])
        embedded.update(zip(kept, vectors, strict=True))
    return embedded


def union_size(members: frozenset[int], clip_sets: list[frozenset[str]]) -> int:
    return len(frozenset().union(*(clip_sets[verb] for verb in members)))


def weighted_center(points: np.ndarray, weights: np.ndarray, members: frozenset[int]) -> np.ndarray:
    rows = sorted(members)
    return np.average(points[rows], axis=0, weights=weights[rows])


def weighted_spread(points: np.ndarray, weights: np.ndarray, members: frozenset[int]) -> float:
    rows = sorted(members)
    center = np.average(points[rows], axis=0, weights=weights[rows])
    return float((weights[rows] * ((points[rows] - center) ** 2).sum(axis=1)).sum())


def silhouette(distances: np.ndarray, labels: np.ndarray) -> float:
    clusters = labels.max() + 1
    if clusters < 2 or clusters >= len(labels):
        return float("nan")
    return float(silhouette_score(distances, labels, metric="precomputed"))


def cluster_labels(clusters: list[frozenset[int]], verbs: int) -> np.ndarray:
    labels = np.empty(verbs, dtype=int)
    ordered = sorted(clusters, key=min)
    for label, members in enumerate(ordered):
        labels[sorted(members)] = label
    return labels


def step_record(
    clusters: list[frozenset[int]],
    clip_sets: list[frozenset[str]],
    distances: np.ndarray,
    min_clips: int,
) -> dict:
    sizes = [union_size(members, clip_sets) for members in clusters]
    labels = cluster_labels(clusters, len(clip_sets))
    return {
        "clusters": len(clusters),
        "min_clips": min(sizes),
        "median_clips": float(np.median(sizes)),
        "max_clips": max(sizes),
        "below_min": sum(size < min_clips for size in sizes),
        "silhouette": silhouette(distances, labels),
        "valid": min(sizes) >= min_clips,
        "labels": labels,
        "members": sorted(clusters, key=min),
        "sizes": [sizes[clusters.index(members)] for members in sorted(clusters, key=min)],
    }


def ward_costs(centers: np.ndarray, masses: np.ndarray) -> np.ndarray:
    squared = (centers**2).sum(axis=1)
    distance = np.maximum(squared[:, None] + squared[None, :] - 2 * centers @ centers.T, 0.0)
    costs = masses[:, None] * masses[None, :] / (masses[:, None] + masses[None, :]) * distance
    np.fill_diagonal(costs, np.inf)
    return costs


def merge_pair(
    clusters: list[frozenset[int]],
    points: np.ndarray,
    weights: np.ndarray,
    clip_sets: list[frozenset[str]],
    min_clips: int,
) -> tuple[int, int]:
    centers = np.stack([weighted_center(points, weights, members) for members in clusters])
    masses = np.array([weights[sorted(members)].sum() for members in clusters])
    costs = ward_costs(centers, masses)
    sizes = [union_size(members, clip_sets) for members in clusters]
    smallest = int(np.argmin(sizes))
    if sizes[smallest] < min_clips:
        return smallest, int(np.argmin(costs[smallest]))
    first, second = np.unravel_index(int(np.argmin(costs)), costs.shape)
    return int(first), int(second)


def bottom_up(
    points: np.ndarray,
    weights: np.ndarray,
    clip_sets: list[frozenset[str]],
    distances: np.ndarray,
    min_clips: int,
) -> list[dict]:
    clusters = [frozenset({verb}) for verb in range(len(points))]
    steps = [step_record(clusters, clip_sets, distances, min_clips)]
    while len(clusters) > 1:
        first, second = merge_pair(clusters, points, weights, clip_sets, min_clips)
        merged = clusters[first] | clusters[second]
        clusters = [members for row, members in enumerate(clusters) if row not in (first, second)]
        clusters.append(merged)
        steps.append(step_record(clusters, clip_sets, distances, min_clips))
    return steps


def best_split(
    members: frozenset[int],
    points: np.ndarray,
    weights: np.ndarray,
    clip_sets: list[frozenset[str]],
    min_clips: int,
) -> tuple[float, frozenset[int], frozenset[int]] | None:
    rows = np.array(sorted(members))
    if len(rows) < 2:
        return None
    labels = KMeans(n_clusters=2, random_state=SEED, n_init="auto").fit_predict(
        points[rows], sample_weight=weights[rows]
    )
    left = frozenset(rows[labels == 0].tolist())
    right = frozenset(rows[labels == 1].tolist())
    if not left or not right:
        return None
    if min(union_size(left, clip_sets), union_size(right, clip_sets)) < min_clips:
        return None
    gain = weighted_spread(points, weights, members)
    gain -= weighted_spread(points, weights, left) + weighted_spread(points, weights, right)
    return gain, left, right


def top_down(
    points: np.ndarray,
    weights: np.ndarray,
    clip_sets: list[frozenset[str]],
    distances: np.ndarray,
    min_clips: int,
) -> list[dict]:
    clusters = [frozenset(range(len(points)))]
    splits = {}
    steps = [step_record(clusters, clip_sets, distances, min_clips)]
    while True:
        for members in clusters:
            if members not in splits:
                splits[members] = best_split(members, points, weights, clip_sets, min_clips)
        candidates = [(splits[members], members) for members in clusters if splits[members]]
        if not candidates:
            return steps
        (_, left, right), parent = max(candidates, key=lambda candidate: candidate[0][0])
        clusters = [members for members in clusters if members != parent] + [left, right]
        steps.append(step_record(clusters, clip_sets, distances, min_clips))


def picked_step(steps: list[dict], clusters: int) -> dict:
    matches = [step for step in steps if step["valid"] and step["clusters"] == clusters]
    if not matches:
        raise RuntimeError(f"no valid step with {clusters} clusters")
    return matches[0]


def best_step(steps: list[dict]) -> dict:
    scored = [step for step in steps if step["valid"] and not np.isnan(step["silhouette"])]
    if not scored:
        raise RuntimeError("no valid step with a silhouette score")
    return max(scored, key=lambda step: step["silhouette"])


def write_steps(path: Path, trees: dict[tuple[str, str], list[dict]]) -> None:
    columns = ["clusters", "min_clips", "median_clips", "max_clips", "below_min", "silhouette"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["space", "direction", "step", *columns, "valid"])
        for (space, direction), steps in trees.items():
            for number, step in enumerate(steps):
                values = [step[column] for column in columns]
                writer.writerow([space, direction, number, *values, step["valid"]])


def write_best(
    path: Path,
    trees: dict[tuple[str, str], list[dict]],
    verbs: list[str],
    weights: np.ndarray,
    pick: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["space", "direction", "clusters", "cluster", "clips", "verbs"])
        for (space, direction), steps in trees.items():
            step = picked_step(steps, pick) if pick else best_step(steps)
            for number, (members, size) in enumerate(
                zip(step["members"], step["sizes"], strict=True)
            ):
                names = sorted(members, key=lambda verb: -weights[verb])
                words = " ".join(verbs[verb] for verb in names)
                writer.writerow([space, direction, step["clusters"], number, size, words])


def write_agreement(path: Path, trees: dict[tuple[str, str], list[dict]]) -> None:
    by_count = {
        key: {step["clusters"]: step["labels"] for step in steps if step["valid"]}
        for key, steps in trees.items()
    }
    shared = sorted(set.intersection(*(set(labels) for labels in by_count.values())) - {1})
    pairs = (
        ((MOTION, BOTTOM_UP), (MOTION, TOP_DOWN)),
        ((TEXT, BOTTOM_UP), (TEXT, TOP_DOWN)),
        ((MOTION, BOTTOM_UP), (TEXT, BOTTOM_UP)),
        ((MOTION, TOP_DOWN), (TEXT, TOP_DOWN)),
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["clusters", *("_vs_".join("_".join(side) for side in pair) for pair in pairs)]
        )
        for count in shared:
            row = [
                adjusted_rand_score(by_count[first][count], by_count[second][count])
                for first, second in pairs
            ]
            writer.writerow([count, *(f"{value:.4f}" for value in row)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_clips", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("doc/notes"))
    parser.add_argument("--pick", type=int, default=0)
    arguments = parser.parse_args()

    app = ApplicationBootstrap.from_config(CONFIG, seed=SEED)
    evaluator = app.create_humanml3d_generation_evaluator()
    resources = evaluator.resources
    verbs = action_verbs()
    groups = clip_groups({verb: verb for verb in verbs})
    clip_ids = sorted(frozenset().union(*groups.values()))
    embedded = clip_embeddings(evaluator, clip_ids)
    verb_clips = {
        verb: frozenset(clip for clip in groups.get(verb, ()) if clip in embedded) for verb in verbs
    }
    kept = [verb for verb in verbs if verb_clips[verb]]
    print(f"action verbs {len(verbs)}, with clips {len(kept)}")
    print(
        f"clips {len(clip_ids)}, embedded {len(embedded)}, dropped {len(clip_ids) - len(embedded)}"
    )

    clip_sets = [verb_clips[verb] for verb in kept]
    weights = np.array([len(clips) for clips in clip_sets], dtype=float)
    motion = np.stack([np.mean([embedded[clip] for clip in clips], axis=0) for clips in clip_sets])
    text = compute_guo_text_embeddings(
        resources.text_embedder,
        [
            resources.build_guo_text_inputs(["a/DET", "person/NOUN", f"{verb}/VERB"])
            for verb in kept
        ],
        resources.device,
    )
    trees = {}
    for space, points in ((MOTION, motion), (TEXT, text)):
        distances = cdist(points, points)
        trees[space, BOTTOM_UP] = bottom_up(
            points, weights, clip_sets, distances, arguments.min_clips
        )
        trees[space, TOP_DOWN] = top_down(
            points, weights, clip_sets, distances, arguments.min_clips
        )
    write_steps(arguments.out / "verb_cluster_steps.csv", trees)
    write_best(arguments.out / "verb_cluster_best.csv", trees, kept, weights, arguments.pick)
    write_agreement(arguments.out / "verb_cluster_agreement.csv", trees)
    for (space, direction), steps in trees.items():
        valid = [step["clusters"] for step in steps if step["valid"]]
        best = best_step(steps)
        print(
            f"{space:6s} {direction:9s} valid clusters {min(valid)}..{max(valid)}  "
            f"best {best['clusters']} (silhouette {best['silhouette']:.3f}, "
            f"clips {best['min_clips']}..{best['max_clips']})"
        )


if __name__ == "__main__":
    main()
