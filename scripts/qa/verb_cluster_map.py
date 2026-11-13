# python scripts/qa/verb_cluster_map.py --min_clips 50
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from verb_clusters import bottom_up, motion_points, picked_step, verb_clip_embeddings
from verb_generation_probe import CONFIG, SEED

from text2motion.app.bootstrap import ApplicationBootstrap

MAP_LEVELS = (10, 20, 50, 100)
TEST_LEVELS = (5, 10, 15, 20, 30, 40, 50, 60, 80, 100)
FOLDS = 5
NAME_VERBS = 2
OUT = Path("doc/paper/pics")


def level_labels(steps: list[dict], levels: tuple[int, ...]) -> dict[int, np.ndarray]:
    return {level: picked_step(steps, level)["labels"] for level in levels}


def class_names(labels: np.ndarray, verbs: list[str], weights: np.ndarray) -> dict[int, str]:
    members: dict[int, list[int]] = defaultdict(list)
    for verb, label in enumerate(labels):
        members[int(label)].append(verb)
    return {
        label: "/".join(
            verbs[verb] for verb in sorted(rows, key=lambda row: -weights[row])[:NAME_VERBS]
        )
        for label, rows in members.items()
    }


def single_class_clips(labels: np.ndarray, clip_sets: list[frozenset[str]]) -> dict[str, int]:
    classes: dict[str, set[int]] = defaultdict(set)
    for verb, clips in enumerate(clip_sets):
        for clip in clips:
            classes[clip].add(int(labels[verb]))
    return {clip: next(iter(found)) for clip, found in sorted(classes.items()) if len(found) == 1}


def class_recall(
    clip_classes: dict[str, int], embedded: dict[str, np.ndarray]
) -> tuple[dict[int, float], float, int]:
    counts = np.bincount(list(clip_classes.values()))
    tested = {clip: label for clip, label in clip_classes.items() if counts[label] >= FOLDS}
    features = np.stack([embedded[clip] for clip in tested])
    targets = np.array(list(tested.values()))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    predicted = cross_val_predict(model, features, targets, cv=folds)
    labels = np.unique(targets)
    recalls = np.diag(confusion_matrix(targets, predicted, labels=labels, normalize="true"))
    balanced = float(balanced_accuracy_score(targets, predicted))
    return dict(zip(labels.tolist(), recalls.tolist(), strict=True)), balanced, len(tested)


def write_map(
    path: Path,
    verbs: list[str],
    weights: np.ndarray,
    flat: dict[str, np.ndarray],
    labels: dict[int, np.ndarray],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["verb", "clips", *flat, *(f"class_{level}" for level in labels)])
        for row, verb in enumerate(verbs):
            coordinates = [f"{flat[name][row]:.4f}" for name in flat]
            classes = [int(level_label[row]) for level_label in labels.values()]
            writer.writerow([verb, int(weights[row]), *coordinates, *classes])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_clips", type=int, default=50)
    arguments = parser.parse_args()

    app = ApplicationBootstrap.from_config(CONFIG, seed=SEED)
    evaluator = app.create_humanml3d_generation_evaluator()
    verbs, clip_sets, embedded = verb_clip_embeddings(evaluator)
    weights = np.array([len(clips) for clips in clip_sets], dtype=float)
    points = motion_points(clip_sets, embedded)
    steps = bottom_up(points, weights, clip_sets, cdist(points, points), arguments.min_clips)

    pca = PCA(n_components=2, random_state=SEED)
    pca_xy = pca.fit_transform(points)
    tsne_xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(
        points
    )
    print(f"PCA 2D keeps {pca.explained_variance_ratio_.sum():.1%} of the variance")
    flat = {
        "pca_x": pca_xy[:, 0],
        "pca_y": pca_xy[:, 1],
        "tsne_x": tsne_xy[:, 0],
        "tsne_y": tsne_xy[:, 1],
    }
    write_map(OUT / "verb_cluster_map.csv", verbs, weights, flat, level_labels(steps, MAP_LEVELS))

    class_rows, level_rows = [], []
    for level, labels in level_labels(steps, TEST_LEVELS).items():
        clip_classes = single_class_clips(labels, clip_sets)
        recalls, balanced, tested = class_recall(clip_classes, embedded)
        names = class_names(labels, verbs, weights)
        sizes = picked_step(steps, level)["sizes"]
        for label, name in sorted(names.items()):
            single = sum(found == label for found in clip_classes.values())
            members = " ".join(verbs[row] for row in np.flatnonzero(labels == label))
            recall = recalls.get(label, float("nan"))
            class_rows.append([level, label, name, sizes[label], single, f"{recall:.4f}", members])
        values = np.array(list(recalls.values()))
        level_rows.append(
            [
                level,
                len(recalls),
                tested,
                len(clip_classes),
                f"{balanced:.4f}",
                f"{1 / len(recalls):.4f}",
                f"{np.median(values):.4f}",
                int((values >= 0.5).sum()),
                int((values < 2 / len(recalls)).sum()),
            ]
        )
        print(level_rows[-1])
    with (OUT / "verb_cluster_classes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["level", "class", "name", "clips", "single_class_clips", "recall", "verbs"]
        )
        writer.writerows(class_rows)
    with (OUT / "verb_cluster_separability.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "classes",
                "classes_tested",
                "clips_tested",
                "single_class_clips",
                "balanced_accuracy",
                "chance",
                "median_recall",
                "classes_recall_at_least_half",
                "classes_under_twice_chance",
            ]
        )
        writer.writerows(level_rows)


if __name__ == "__main__":
    main()
