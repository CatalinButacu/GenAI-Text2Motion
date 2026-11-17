# .\.venv\Scripts\python.exe scripts/dataset/prepare_source.py motionx_official --fps 60
import argparse
import csv
import pathlib
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from text2motion.dataset.amass import AmassReader, amass_files_outside_hml3d
from text2motion.dataset.contracts import (
    MIN_CLIP_SECONDS,
    MIN_UPRIGHT_SHARE,
    MOTIONX_FPS,
    MOTIONX_SUBSETS,
    DatasetPaths,
    clip_name,
)
from text2motion.dataset.features import clip_features, encoder_params
from text2motion.dataset.motionx import motionx_body_clips, motionx_captions
from text2motion.dataset.quality import upright_share
from text2motion.dataset.resample import cubic_resample, downsample, lowpass
from text2motion.motion.representation import FeatureParams

REPO = pathlib.Path(__file__).resolve().parents[2]
PATHS = DatasetPaths(REPO, pathlib.Path(r"D:\Facultate\dissertation\data\amass"))
REFERENCE_JOINTS = REPO / "data" / "HumanML3D_263" / "joints" / "000021.npy"
SOURCES = ("amass_extra", "motionx_official")
FPS_CHOICES = (20, 60)
INVENTORY_COLUMNS = (
    "name",
    "subset",
    "frames",
    "seconds",
    "upright_share",
    "up_axis_ok",
    "caption",
)


############################################################
#       RawClip
############################################################
@dataclass(frozen=True)
class RawClip:
    name: str
    subset: str
    joints: np.ndarray
    fps: float
    caption: str

    def inventory_row(self) -> tuple:
        share = upright_share(self.joints)
        frames = self.joints.shape[0]
        seconds = round(frames / self.fps, 2)
        upright = share >= MIN_UPRIGHT_SHARE
        return (self.name, self.subset, frames, seconds, round(share, 3), upright, self.caption)


############################################################
#       Target
############################################################
@dataclass(frozen=True)
class Target:
    folder: str
    fps: int
    cutoff: float | None
    params: FeatureParams
    joints_dir: pathlib.Path
    features_dir: pathlib.Path


############################################################
#       Tally
############################################################
@dataclass
class Tally:
    inventory: list[tuple] = field(default_factory=list)
    non_finite: list[str] = field(default_factory=list)
    written: int = 0
    seconds: float = 0.0


def __arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=SOURCES)
    parser.add_argument("--fps", type=int, choices=FPS_CHOICES, required=True)
    parser.add_argument("--lowpass_hz", type=float, default=None)
    return parser.parse_args()


def __amass_clips(fps: int) -> Iterator[RawClip]:
    reader = AmassReader(PATHS.smplx_models, fps)
    files = amass_files_outside_hml3d(PATHS.amass_raw, PATHS.hml3d_official)
    for path in tqdm(files, desc="amass"):
        joints = reader.clip_joints(path, PATHS.amass_raw)
        if joints is None:
            continue
        parts = path.relative_to(PATHS.amass_raw).with_suffix("").parts
        yield RawClip(clip_name(*parts), parts[0], joints, fps, "")


def __motionx_clips() -> Iterator[RawClip]:
    for subset in MOTIONX_SUBSETS:
        captions = motionx_captions(PATHS.motionx / "semantic_label" / f"{subset}.zip")
        clips = motionx_body_clips(PATHS.motionx / "joint" / f"{subset}.zip")
        for stem, joints in tqdm(clips, desc=subset):
            caption = captions.get(stem, "")
            yield RawClip(clip_name(subset, stem), subset, joints, MOTIONX_FPS, caption)


def __at_target(clip: RawClip, fps: int, cutoff: float | None) -> np.ndarray:
    smooth = clip.joints if cutoff is None else lowpass(clip.joints, clip.fps, cutoff)
    if clip.fps < fps:
        return cubic_resample(smooth, clip.fps, fps)
    return downsample(smooth, int(clip.fps), fps)


def __long_enough(frames: int, fps: float) -> bool:
    return frames >= MIN_CLIP_SECONDS * fps


def __write_inventory(path: pathlib.Path, rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(INVENTORY_COLUMNS)
        writer.writerows(rows)


def __add_clip(clip: RawClip, target: Target, tally: Tally) -> None:
    row = clip.inventory_row()
    tally.inventory.append(row)
    if not row[5] or not __long_enough(clip.joints.shape[0], clip.fps):
        return
    joints = __at_target(clip, target.fps, target.cutoff)
    np.save(target.joints_dir / f"{clip.name}.npy", joints.astype(np.float32))
    if not __long_enough(joints.shape[0], target.fps):
        return
    features = clip_features(joints, target.params)
    if features is None:
        tally.non_finite.append(clip.name)
        return
    np.save(target.features_dir / f"{clip.name}.npy", features)
    tally.written += 1
    tally.seconds += features.shape[0] / target.fps


def __target(source: str, fps: int, cutoff: float | None) -> Target:
    folder = source if cutoff is None else f"{source}_lp{cutoff:g}"
    target = Target(
        folder=folder,
        fps=fps,
        cutoff=cutoff,
        params=encoder_params(REFERENCE_JOINTS, fps),
        joints_dir=PATHS.joints(fps, source),
        features_dir=PATHS.features(fps, folder),
    )
    target.joints_dir.mkdir(parents=True, exist_ok=True)
    target.features_dir.mkdir(parents=True, exist_ok=True)
    return target


def main() -> None:
    arguments = __arguments()
    target = __target(arguments.source, arguments.fps, arguments.lowpass_hz)
    amass = arguments.source == "amass_extra"
    clips = __amass_clips(target.fps) if amass else __motionx_clips()
    tally = Tally()
    for clip in clips:
        __add_clip(clip, target, tally)
    suffix = f"{target.fps}fps"
    __write_inventory(PATHS.extra / f"inventory_{arguments.source}_{suffix}.csv", tally.inventory)
    rejected = PATHS.extra / f"rejected_{target.folder}_{suffix}.txt"
    rejected.write_text("".join(f"{name}\n" for name in tally.non_finite), encoding="utf-8")
    print(
        f"{target.folder}@{suffix}: clips {len(tally.inventory)}  features {tally.written}"
        f"  hours {tally.seconds / 3600:.2f}  non-finite {len(tally.non_finite)}"
    )


if __name__ == "__main__":
    main()
