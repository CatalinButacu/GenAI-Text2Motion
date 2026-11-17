from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from text2motion.dataset.contracts import (
    CONTACT_HEIGHT_M,
    FEET,
    FOOT_JOINTS,
    GRAVITY_M_PER_S2,
    HEAD_JOINT,
    SEED,
    SEGMENTS,
    STEP0_AMASS_SUBSETS,
    NewtonLimits,
    QualityLimits,
)
from text2motion.dataset.features import feature_joints

UP_AXIS = 1
GROUND_AXES = (0, 2)
QUALITY_METRICS = ("skating_ratio", "float_mm", "jerk_m_per_s3", "yaw_p99_rad_per_s")
METRES_TO_MM = 1000.0
SECONDS_PER_HOUR = 3600
ALL_SUBSETS = "all"
AMASS_EXTRA = "amass_extra"


############################################################
#       NewtonResult
############################################################
@dataclass(frozen=True)
class NewtonResult:
    hover_share: float
    unbalanced_share: float

    @property
    def violation_share(self) -> float:
        return self.hover_share + self.unbalanced_share


def long_runs(mask: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1).tolist()
    ends = np.flatnonzero(edges == -1).tolist()
    return [
        (start, end) for start, end in zip(starts, ends, strict=True) if end - start >= min_frames
    ]


def frames(seconds: float, fps: int) -> int:
    return round(seconds * fps)


def hover_frames(
    acceleration: np.ndarray, contact: np.ndarray, fps: int, limits: NewtonLimits
) -> np.ndarray:
    hover = np.zeros(len(contact), dtype=bool)
    gravity = np.array([0.0, -GRAVITY_M_PER_S2, 0.0])
    limit = limits.hover_fraction_of_g * GRAVITY_M_PER_S2
    for start, end in long_runs(~contact, frames(limits.min_flight_seconds, fps)):
        departure = np.linalg.norm(acceleration[start:end] - gravity, axis=1).mean()
        hover[start:end] = departure >= limit
    return hover


def whole_feet(touching: np.ndarray) -> np.ndarray:
    support = touching.copy()
    for ankle, toe in FEET:
        foot_down = touching[:, ankle] | touching[:, toe]
        support[:, ankle] = foot_down
        support[:, toe] = foot_down
    return support


def unbalanced_frames(
    joints: np.ndarray,
    com: np.ndarray,
    acceleration: np.ndarray,
    touching: np.ndarray,
    fps: int,
    limits: NewtonLimits,
) -> np.ndarray:
    support = whole_feet(touching)
    plane = joints[..., list(GROUND_AXES)]
    margin = limits.support_margin_m
    low = np.where(support[..., None], plane, np.inf).min(axis=1) - margin
    high = np.where(support[..., None], plane, -np.inf).max(axis=1) + margin
    centre = com[:, list(GROUND_AXES)]
    outside = ((centre < low) | (centre > high)).any(axis=1)
    still = np.linalg.norm(acceleration, axis=1) < limits.quasi_static_m_per_s2
    candidate = outside & still & touching.any(axis=1)
    unbalanced = np.zeros(len(candidate), dtype=bool)
    for start, end in long_runs(candidate, frames(limits.min_unbalanced_seconds, fps)):
        unbalanced[start:end] = True
    return unbalanced


def newton_violations(joints: np.ndarray, fps: int, limits: NewtonLimits) -> NewtonResult:
    def centre_of_mass(joints: np.ndarray) -> np.ndarray:
        masses = np.array([mass for _, _, mass in SEGMENTS])
        centres = np.stack([(joints[:, a] + joints[:, b]) / 2 for a, b, _ in SEGMENTS], axis=1)
        return np.einsum("s,fsc->fc", masses, centres) / masses.sum()

    heights = joints[..., UP_AXIS] - joints[..., UP_AXIS].min()
    touching = heights < np.array(CONTACT_HEIGHT_M)
    com = centre_of_mass(joints)
    acceleration = np.gradient(np.gradient(com, axis=0), axis=0) * fps * fps
    hover = hover_frames(acceleration, touching.any(axis=1), fps, limits)
    unbalanced = unbalanced_frames(joints, com, acceleration, touching, fps, limits)
    return NewtonResult(float(hover.mean()), float(unbalanced.mean()))


def flight_accelerations(
    joints: np.ndarray, fps: int, clearance_m: float, min_flight_seconds: float
) -> list[float]:
    heights = joints[..., UP_AXIS]
    feet = heights[:, [ankle for pair in FEET for ankle in pair]].min(axis=1)
    floor = np.percentile(feet, 2)
    centre = heights.mean(axis=1)
    accelerations = []
    for start, end in long_runs(feet - floor > clearance_m, frames(min_flight_seconds, fps)):
        times = np.arange(end - start) / fps
        accelerations.append(2.0 * float(np.polyfit(times, centre[start:end], 2)[0]))
    return accelerations


def quality_metrics(features: np.ndarray, fps: int, limits: QualityLimits) -> tuple[float, ...]:
    joints = feature_joints(features).astype(np.float64)
    feet = joints[:, list(FOOT_JOINTS)]
    contact = feet[:-1, :, 1] < limits.contact_height_m
    step = np.linalg.norm(np.diff(feet[..., [0, 2]], axis=0), axis=-1)
    skid = step > limits.skid_m_per_s / fps
    skating = float((contact & skid).any(axis=-1).mean())
    lowest = feet[..., 1].min(axis=-1)
    floating = float(np.maximum(0.0, lowest - limits.float_tolerance_m).mean() * METRES_TO_MM)
    jerk = np.linalg.norm(np.diff(joints, n=3, axis=0), axis=-1) * fps**3
    yaw = float(np.percentile(np.abs(features[:, 0]), limits.yaw_percentile)) * fps
    return skating, floating, float(jerk.mean()), yaw


def fit_quality_forest(train_metrics: np.ndarray, limits: QualityLimits) -> IsolationForest:
    forest = IsolationForest(n_estimators=limits.trees, contamination="auto", random_state=SEED)
    return forest.fit(train_metrics)


def quality_inliers(forest: IsolationForest, metrics: np.ndarray) -> np.ndarray:
    return forest.predict(metrics) == 1


############################################################
#       ClipVerdict
############################################################
@dataclass(frozen=True)
class ClipVerdict:
    source: str
    subset: str
    name: str
    seconds: float
    violation_share: float
    inlier: bool

    def kept(self, limits: NewtonLimits) -> bool:
        return self.violation_share < limits.reject_violation_share and self.inlier


def in_step0(source: str, subset: str) -> bool:
    return source != AMASS_EXTRA or subset in STEP0_AMASS_SUBSETS


def summary_table(verdicts: Sequence[ClipVerdict], limits: NewtonLimits) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[ClipVerdict]] = defaultdict(list)
    for verdict in verdicts:
        groups[(verdict.source, ALL_SUBSETS)].append(verdict)
        if verdict.subset != verdict.source:
            groups[(verdict.source, verdict.subset)].append(verdict)
    table = []
    for (source, subset), members in sorted(groups.items()):
        hours = sum(member.seconds for member in members) / SECONDS_PER_HOUR
        newton = [
            member for member in members if member.violation_share < limits.reject_violation_share
        ]
        kept = [member for member in members if member.kept(limits)]
        table.append(
            {
                "source": source,
                "subset": subset,
                "clips": len(members),
                "hours": round(hours, 2),
                "newton_kept_clips": len(newton),
                "t5g_inlier_clips": sum(member.inlier for member in members),
                "kept_clips": len(kept),
                "kept_hours": round(sum(m.seconds for m in kept) / SECONDS_PER_HOUR, 2),
                "in_step0": in_step0(source, subset),
            }
        )
    return table


def upright_share(joints: np.ndarray) -> float:
    head_above_feet = joints[:, HEAD_JOINT] - joints[:, list(FOOT_JOINTS)].mean(axis=1)
    return float((head_above_feet[:, 1] > 0).mean())
