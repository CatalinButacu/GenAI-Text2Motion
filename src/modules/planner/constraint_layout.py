import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

log = logging.getLogger(__name__)

# Maps canonical relation names (from SpatialRelation.relation) to cost-function keys.
# M1 already resolves raw text -> canonical, so no text re-parsing needed here.
CANON_TO_CTYPE: dict[str, str] = {
    "ON": "ON",
    "ABOVE": "ABOVE",
    "UNDER": "BELOW",
    "IN_FRONT_OF": "FRONT",
    "BEHIND": "BEHIND",
    "BESIDE": "BESIDE",
    "NEAR": "NEAR",
    "LEFT_OF": "LEFT",
    "RIGHT_OF": "RIGHT",
    "INSIDE": "INSIDE",
}

WEIGHTS: dict[str, float] = {
    "ON": 10.0,
    "ABOVE": 5.0,
    "BELOW": 5.0,
    "FRONT": 5.0,
    "BEHIND": 5.0,
    "BESIDE": 5.0,
    "NEAR": 3.0,
    "LEFT": 5.0,
    "RIGHT": 5.0,
    "INSIDE": 3.0,
}

STACK_GAP = 0.25
SIDE_OFFSET = 0.5
NEAR_DIST = 0.7
REPULSION = 2.0
MIN_DIST = 0.3
GROUND_Z = 0.15


@dataclass(slots=True, frozen=True)
class SpatialConstraint:
    subject: str
    object: str
    ctype: str


def parseOne(rel) -> SpatialConstraint | None:
    subj = getattr(rel, "subject", "").strip().lower()
    obj = getattr(rel, "object", "").strip().lower()
    ctype = CANON_TO_CTYPE.get(getattr(rel, "relation", ""))

    return (
        SpatialConstraint(subject=subj, object=obj, ctype=ctype)
        if (subj and obj and ctype)
        else None
    )


def buildConstraints(relations: list) -> list[SpatialConstraint]:
    return [c for rel in relations if (c := parseOne(rel)) is not None]


def costOn(sx, sy, sz, ox, oy, oz):
    return ((sx - ox) ** 2 + (sy - oy) ** 2) + (sz - oz - STACK_GAP) ** 2


def costAbove(sx, sy, sz, ox, oy, oz):
    return max(0, oz + STACK_GAP - sz) ** 2


def costBelow(sx, sy, sz, ox, oy, oz):
    return max(0, sz - oz + STACK_GAP) ** 2


def costFront(sx, sy, sz, ox, oy, oz):
    return max(0, sy - oy + SIDE_OFFSET) ** 2


def costBehind(sx, sy, sz, ox, oy, oz):
    return max(0, oy - sy + SIDE_OFFSET) ** 2


def costBeside(sx, sy, sz, ox, oy, oz):
    return ((sx - ox) ** 2 - SIDE_OFFSET**2) ** 2 / (SIDE_OFFSET**2)


def costNear(sx, sy, sz, ox, oy, oz):
    d2 = (sx - ox) ** 2 + (sy - oy) ** 2 + (sz - oz) ** 2

    return (d2 - NEAR_DIST**2) ** 2


def costLeft(sx, sy, sz, ox, oy, oz):
    return max(0, sx - ox + SIDE_OFFSET) ** 2


def costRight(sx, sy, sz, ox, oy, oz):
    return max(0, ox - sx + SIDE_OFFSET) ** 2


def costInside(sx, sy, sz, ox, oy, oz):
    return (sx - ox) ** 2 + (sy - oy) ** 2 + (sz - oz) ** 2


COST_FN: dict[str, Callable] = {
    "ON": costOn,
    "ABOVE": costAbove,
    "BELOW": costBelow,
    "FRONT": costFront,
    "BEHIND": costBehind,
    "BESIDE": costBeside,
    "NEAR": costNear,
    "LEFT": costLeft,
    "RIGHT": costRight,
    "INSIDE": costInside,
}


def repulsion(pos: np.ndarray, n: int) -> float:
    if n < 2:
        return 0.0

    dists = pdist(pos) + 1e-6
    mask = dists < MIN_DIST

    return float(REPULSION * np.sum((MIN_DIST / dists[mask] - 1.0) ** 2)) if mask.any() else 0.0


def groundPenalty(pos: np.ndarray) -> float:
    below = GROUND_Z - pos[:, 2]
    below = below[below > 0]

    return float(2.0 * np.sum(below**2)) if below.size else 0.0


def buildEnergy(
    entityNames: list[str],
    constraints: list[SpatialConstraint],
    nameToIdx: dict[str, int],
) -> Callable[[np.ndarray], float]:
    n = len(entityNames)

    def energy(x: np.ndarray) -> float:
        pos = x.reshape(n, 3)
        cost = 0.0

        for c in constraints:
            si, oi = nameToIdx.get(c.subject), nameToIdx.get(c.object)

            if si is None or oi is None:
                continue

            fn = COST_FN.get(c.ctype)

            if fn is None:
                continue

            cost += WEIGHTS.get(c.ctype, 3.0) * fn(*pos[si], *pos[oi])
        cost += repulsion(pos, n)
        cost += groundPenalty(pos)

        return cost

    return energy


def solveLayout(
    entityNames: list[str],
    relations: list,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    if not entityNames:
        return {}

    constraints = buildConstraints(relations)

    if not constraints:
        log.info("No spatial constraints found")

        return {}

    nameToIdx = {name.lower(): i for i, name in enumerate(entityNames)}
    n = len(entityNames)
    x0 = initPositions(n, seed)
    energyFn = buildEnergy(entityNames, constraints, nameToIdx)
    result = minimize(energyFn, x0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-8})

    if not result.success:
        log.warning("Layout solver did not converge: %s", result.message)
    log.info(
        "Constraint solver: %d constraints, %d entities, cost=%.4f",
        len(constraints),
        n,
        result.fun,
    )

    return extractPositions(result.x, entityNames)


def initPositions(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0 = np.zeros(n * 3)

    for i in range(n):
        x0[i * 3] = (i % 3 - 1) * SIDE_OFFSET + rng.uniform(-0.1, 0.1)
        x0[i * 3 + 1] = (i // 3) * SIDE_OFFSET + rng.uniform(-0.1, 0.1)
        x0[i * 3 + 2] = GROUND_Z + rng.uniform(0, 0.05)

    return x0


def extractPositions(
    flat: np.ndarray, entityNames: list[str]
) -> dict[str, tuple[float, float, float]]:
    positions = flat.reshape(len(entityNames), 3)

    return {
        name: (
            round(float(positions[i, 0]), 3),
            round(float(positions[i, 1]), 3),
            round(float(max(positions[i, 2], 0.05)), 3),
        )
        for i, name in enumerate(entityNames)
    }
