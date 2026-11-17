# .\.venv\Scripts\python.exe scripts/extra_motion/reconstruction_gate.py aist style100
import csv
import pathlib
import sys

import numpy as np
import torch

from text2motion.app.bootstrap import required_path
from text2motion.app.config import load_config
from text2motion.motion.contracts import Split
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import JOINTS, recover_from_ric
from text2motion.motion.storage import FEATURES_DIR
from text2motion.tokenization.metrics import select_reconstruction_evaluation_clip_ids
from text2motion.tokenization.nn_model import build_tokenizer_network

REPO = pathlib.Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "generator" / "a7" / "width_320_step50.yaml"
TOKENIZER = REPO / "artifacts" / "checkpoints" / "tokenizer" / "fsq_g8_v1024.pt"
FEATURES_ROOT = REPO / "data" / "extra_motion" / "features20"
OUT_CSV = REPO / "doc" / "notes" / "extra_motion_gate.csv"
WINDOW_FRAMES = 196
CLIPS_PER_SOURCE = 200
SEED = 2026
RATIO_LIMIT = 2.0
LOCAL_RATIO_LIMIT = 2.0
MAX_WINDOWS_PER_CLIP = 6
COMPAT_CSV = REPO / "doc" / "notes" / "extra_motion_compat.csv"
COMPAT_COLUMNS = ("source", "clips", "hours_total", "local_limit_mm", "hours_compatible", "share")
FINITE_SHARE_MIN = 0.9
METRES_TO_MM = 1000.0
COLUMNS = (
    "source",
    "clips",
    "finite_share",
    "median_mpjpe_mm",
    "median_feature_l2",
    "ratio_to_val_mpjpe",
    "pass",
)


def __window(features: np.ndarray, downsample: int, rng: np.random.Generator) -> np.ndarray:
    length = (min(features.shape[0], WINDOW_FRAMES) // downsample) * downsample
    if length < 2 * downsample:
        raise ValueError(f"clip of {features.shape[0]} frames is too short")
    start = int(rng.integers(0, features.shape[0] - length + 1))
    return features[start : start + length]


@torch.no_grad()
def __clip_errors(module, scaler: MotionScaler, window: np.ndarray) -> tuple[float, float, float]:
    normalized = torch.from_numpy(scaler.normalize(window[None]).astype(np.float32))
    reconstruction = np.asarray(scaler.denormalize(module(normalized)[0].numpy()), dtype=np.float32)
    feature_l2 = float(np.sqrt(((window[None] - reconstruction) ** 2).sum(-1)).mean())
    reference = recover_from_ric(torch.from_numpy(window[None]), JOINTS)
    rebuilt = recover_from_ric(torch.from_numpy(reconstruction), JOINTS)
    mpjpe = float(np.sqrt(((reference - rebuilt).numpy() ** 2).sum(-1)).mean()) * METRES_TO_MM
    offsets = (reference - reference[:, :, :1]) - (rebuilt - rebuilt[:, :, :1])
    local = float(np.sqrt((offsets.numpy() ** 2).sum(-1)).mean()) * METRES_TO_MM
    return mpjpe, feature_l2, local


def __window_starts(frames: int, downsample: int) -> list[int]:
    count = frames // WINDOW_FRAMES
    if count == 0:
        return [0]
    step = max(1, count // MAX_WINDOWS_PER_CLIP)
    return [index * WINDOW_FRAMES for index in range(0, count, step)][:MAX_WINDOWS_PER_CLIP]


def __local_errors(module, scaler: MotionScaler, paths: list[pathlib.Path]) -> list[float]:
    errors = []
    for path in paths:
        features = np.load(path).astype(np.float32)
        for start in __window_starts(features.shape[0], module.downsample):
            window = features[start : start + WINDOW_FRAMES]
            window = window[: (len(window) // module.downsample) * module.downsample]
            errors.append(__clip_errors(module, scaler, window)[2])
    return errors


def __compatibility_rows(module, scaler, val_paths, sources) -> list[tuple]:
    limit = LOCAL_RATIO_LIMIT * float(np.median(__local_errors(module, scaler, val_paths)))
    rows = []
    for source in sources:
        files = sorted((FEATURES_ROOT / source).glob("*.npy"))
        kept_frames, total_frames = 0, 0
        for path in files:
            frames = np.load(path, mmap_mode="r").shape[0]
            errors = __local_errors(module, scaler, [path])
            share_ok = float(np.mean([error <= limit for error in errors]))
            kept_frames += int(frames * share_ok)
            total_frames += frames
        rows.append(
            (
                source,
                len(files),
                round(total_frames / 20 / 3600, 2),
                round(limit, 1),
                round(kept_frames / 20 / 3600, 2),
                round(kept_frames / total_frames, 3),
            )
        )
    return rows


def __score(module, scaler, paths: list[pathlib.Path], rng) -> tuple[float, float, float]:
    mpjpe, l2, finite = [], [], 0
    for path in paths:
        features = np.load(path).astype(np.float32)
        if not np.isfinite(features).all():
            continue
        finite += 1
        errors = __clip_errors(module, scaler, __window(features, module.downsample, rng))
        mpjpe.append(errors[0])
        l2.append(errors[1])
    if not mpjpe:
        raise ValueError("no finite clip to score")
    return float(np.median(mpjpe)), float(np.median(l2)), finite / len(paths)


def main() -> None:
    sources = [name for name in sys.argv[1:] if not name.startswith("--")]
    compat = "--compat" in sys.argv
    if not sources:
        raise SystemExit("give at least one source folder name under features20")
    config = load_config(CONFIG)
    module = build_tokenizer_network(config.tokenizer, config.rvq_baseline)
    module.load_state_dict(torch.load(TOKENIZER, map_location="cpu", weights_only=True))
    module.eval()
    root = required_path(config.paths.hml3d_out_dir, "hml3d_out_dir")
    scaler = MotionScaler.load(root, dim=263)
    ids = select_reconstruction_evaluation_clip_ids(
        root, Split.VALIDATION, max_clips=CLIPS_PER_SOURCE, shuffle_seed=SEED
    )
    rng = np.random.default_rng(SEED)
    val_paths = [root / FEATURES_DIR / f"{clip_id}.npy" for clip_id in ids]
    if compat:
        compat_rows = __compatibility_rows(module, scaler, val_paths, sources)
        with open(COMPAT_CSV, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\r\n")
            writer.writerow(COMPAT_COLUMNS)
            writer.writerows(compat_rows)
        for compat_row in compat_rows:
            print(dict(zip(COMPAT_COLUMNS, compat_row, strict=True)))
        return
    val_mpjpe, val_l2, val_finite = __score(module, scaler, val_paths, rng)
    rows: list[tuple[str, int, float, float, float, float, str | bool]] = [
        (
            "humanml3d_val",
            len(val_paths),
            round(val_finite, 3),
            round(val_mpjpe, 2),
            round(val_l2, 4),
            1.0,
            "reference",
        )
    ]
    for source in sources:
        files = sorted((FEATURES_ROOT / source).glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"no features under {FEATURES_ROOT / source}")
        order = np.random.default_rng(SEED).permutation(len(files))[:CLIPS_PER_SOURCE]
        mpjpe, l2, finite = __score(module, scaler, [files[i] for i in order], rng)
        ratio = mpjpe / val_mpjpe
        passed = ratio <= RATIO_LIMIT and finite >= FINITE_SHARE_MIN
        rows.append(
            (
                source,
                len(order),
                round(finite, 3),
                round(mpjpe, 2),
                round(l2, 4),
                round(ratio, 3),
                passed,
            )
        )
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\r\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    for row in rows:
        print(dict(zip(COLUMNS, row, strict=True)))


if __name__ == "__main__":
    main()
