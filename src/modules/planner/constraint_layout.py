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


def parse_one(rel) -> SpatialConstraint | None:
    subj = getattr(rel, "subject", "").strip().lower()
    obj = getattr(rel, "object", "").strip().lower()
    ctype = CANON_TO_CTYPE.get(getattr(rel, "relation", ""))

    return (
        SpatialConstraint(subject=subj, object=obj, ctype=ctype)
        if (subj and obj and ctype)
        else None
    )


def build_constraints(relations: list) -> list[SpatialConstraint]:
    return [c for rel in relations if (c := parse_one(rel)) is not None]


def cost_on(sx, sy, sz, ox, oy, oz):
    return ((sx - ox) ** 2 + (sy - oy) ** 2) + (sz - oz - STACK_GAP) ** 2


def cost_above(sx, sy, sz, ox, oy, oz):
    return max(0, oz + STACK_GAP - sz) ** 2


def cost_below(sx, sy, sz, ox, oy, oz):
    return max(0, sz - oz + STACK_GAP) ** 2


def cost_front(sx, sy, sz, ox, oy, oz):
    return max(0, sy - oy + SIDE_OFFSET) ** 2


def cost_behind(sx, sy, sz, ox, oy, oz):
    return max(0, oy - sy + SIDE_OFFSET) ** 2


def cost_beside(sx, sy, sz, ox, oy, oz):
    return ((sx - ox) ** 2 - SIDE_OFFSET**2) ** 2 / (SIDE_OFFSET**2)


def cost_near(sx, sy, sz, ox, oy, oz):
    d2 = (sx - ox) ** 2 + (sy - oy) ** 2 + (sz - oz) ** 2

    return (d2 - NEAR_DIST**2) ** 2


def cost_left(sx, sy, sz, ox, oy, oz):
    return max(0, sx - ox + SIDE_OFFSET) ** 2


def cost_right(sx, sy, sz, ox, oy, oz):
    return max(0, ox - sx + SIDE_OFFSET) ** 2


def cost_inside(sx, sy, sz, ox, oy, oz):
    return (sx - ox) ** 2 + (sy - oy) ** 2 + (sz - oz) ** 2


COST_FN: dict[str, Callable] = {
    "ON": cost_on,
    "ABOVE": cost_above,
    "BELOW": cost_below,
    "FRONT": cost_front,
    "BEHIND": cost_behind,
    "BESIDE": cost_beside,
    "NEAR": cost_near,
    "LEFT": cost_left,
    "RIGHT": cost_right,
    "INSIDE": cost_inside,
}


def repulsion(pos: np.ndarray, n: int) -> float:
    if n < 2:
        return 0.0

    dists = pdist(pos) + 1e-6
    mask = dists < MIN_DIST

    return float(REPULSION * np.sum((MIN_DIST / dists[mask] - 1.0) ** 2)) if mask.any() else 0.0


def ground_penalty(pos: np.ndarray) -> float:
    below = GROUND_Z - pos[:, 2]
    below = below[below > 0]

    return float(2.0 * np.sum(below**2)) if below.size else 0.0


def build_energy(
    entity_names: list[str],
    constraints: list[SpatialConstraint],
    name_to_idx: dict[str, int],
) -> Callable[[np.ndarray], float]:
    n = len(entity_names)

    def energy(x: np.ndarray) -> float:
        pos = x.reshape(n, 3)
        cost = 0.0

        for c in constraints:
            si, oi = name_to_idx.get(c.subject), name_to_idx.get(c.object)

            if si is None or oi is None:
                continue

            fn = COST_FN.get(c.ctype)

            if fn is None:
                continue

            cost += WEIGHTS.get(c.ctype, 3.0) * fn(*pos[si], *pos[oi])
        cost += repulsion(pos, n)
        cost += ground_penalty(pos)

        return cost

    return energy


def solve_layout(
    entity_names: list[str],
    relations: list,
    seed: int = 42,
) -> dict[str, tuple[float, float, float]]:
    if not entity_names:
        return {}

    constraints = build_constraints(relations)

    if not constraints:
        log.info("No spatial constraints found")

        return {}

    name_to_idx = {name.lower(): i for i, name in enumerate(entity_names)}
    n = len(entity_names)
    x0 = init_positions(n, seed)
    energy_fn = build_energy(entity_names, constraints, name_to_idx)
    result = minimize(energy_fn, x0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-8})

    if not result.success:
        log.warning("Layout solver did not converge: %s", result.message)
    log.info(
        "Constraint solver: %d constraints, %d entities, cost=%.4f",
        len(constraints),
        n,
        result.fun,
    )

    return extract_positions(result.x, entity_names)


def init_positions(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x0 = np.zeros(n * 3)

    for i in range(n):
        x0[i * 3] = (i % 3 - 1) * SIDE_OFFSET + rng.uniform(-0.1, 0.1)
        x0[i * 3 + 1] = (i // 3) * SIDE_OFFSET + rng.uniform(-0.1, 0.1)
        x0[i * 3 + 2] = GROUND_Z + rng.uniform(0, 0.05)

    return x0


def extract_positions(
    flat: np.ndarray, entity_names: list[str]
) -> dict[str, tuple[float, float, float]]:
    positions = flat.reshape(len(entity_names), 3)

    return {
        name: (
            round(float(positions[i, 0]), 3),
            round(float(positions[i, 1]), 3),
            round(float(max(positions[i, 2], 0.05)), 3),
        )
        for i, name in enumerate(entity_names)
    }
