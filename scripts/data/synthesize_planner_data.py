"""Synthesize (instruction -> action_list) pairs for fine-tuning the
GPT-2-small planner LM (decision #14).

Strategy (decision #17a): combinatorial composition over a curated seed
list of HumanML3D-style action verbs + scene targets + linguistic
templates. We do NOT redistribute the HumanML3D dataset; we use ~30
representative motion verbs drawn from the published HumanML3D action
taxonomy and combine them programmatically into single, two-action and
three-action instructions. The seed list lives inline in this file so the
script is fully reproducible without pulling any data.

Output: JSONL at ``data/planner/{train,val}.jsonl`` with one example per
line in the schema the planner LM trains on:

    {"instruction": "walk forward and then sit down",
     "actions": [
       {"action": "walk forward", "until": "duration(40)"},
       {"action": "sit down", "until": "completed"}
     ]}

Every emitted ``until`` string is guaranteed to pass
:func:`src.modules.agent.conditions.parse_condition` -- ``tests/test_planner_dataset.py``
enforces that invariant at the dataset level.

Run from repo root:
    python scripts/data/synthesize_planner_data.py --num-train 6000 --num-val 600
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# --------------------------------------------------------------------- #
# Seed vocabulary -- representative HumanML3D-style verbs and modifiers. #
# Not the HumanML3D dataset itself; this is a hand-curated list of       #
# common motion primitives the trained MotionSSM can render reliably.    #
# --------------------------------------------------------------------- #

ACTION_VERBS = [
    "walks", "runs", "jogs", "strolls",
    "steps", "moves", "marches", "tiptoes",
    "jumps", "hops", "leaps",
    "kicks", "punches", "claps",
    "waves", "bows", "stretches",
    "sits down", "stands up", "lies down",
    "turns", "spins", "rotates",
    "raises arms", "crosses arms",
    "picks up an object", "drops an object",
    "dances", "swings arms",
    "stops", "pauses",
]

DIRECTIONS = ["forward", "backward", "to the left", "to the right",
              "in a circle", "diagonally"]

OBJECT_NAMES = ["tree", "ball", "chair", "wall", "door",
                "rock", "table", "fence", "ladder", "box"]

CONNECTORS = ["then", "and then", "next", ",then", "after that",
              ", followed by", "and"]

INSTRUCTION_PREFIXES = [
    "the person", "a person", "the avatar",
    "the character", "they",
]


def _atomic_action(verb: str | None = None, target: str | None = None) -> dict:
    """One self-contained action with a duration- or completed-style termination."""
    v = verb or random.choice(ACTION_VERBS)
    direction = random.choice(DIRECTIONS) if v in {"walks", "runs", "jogs", "strolls",
                                                    "steps", "moves", "marches", "tiptoes",
                                                    "jumps", "hops", "leaps"} else ""
    action_str = f"{v} {direction}".strip()
    # Atomic gestures use 'completed'; locomotion uses a duration drawn from
    # the empirical HumanML3D clip-length distribution (40-200 frames).
    is_locomotion = v in {"walks", "runs", "jogs", "strolls", "steps", "moves",
                          "marches", "tiptoes", "jumps", "hops", "leaps", "dances",
                          "spins", "rotates"}

    if is_locomotion and random.random() < 0.5:
        until = f"duration({random.choice([40, 60, 80, 100, 120])})"
    else:
        until = "completed"

    return {"action": action_str, "until": until}


def _distance_action(target: str, verb: str | None = None) -> dict:
    """Locomotion until distance threshold to a scene target is met."""
    v = verb or random.choice(["walks", "runs", "strolls", "moves"])
    direction = random.choice(["forward", "toward " + target, ""])
    action_str = f"{v} {direction}".strip()
    op = random.choice(["<", ">"])
    threshold = random.choice([0.5, 1.0, 1.5, 2.0, 3.0])
    until = f"distance({target}) {op} {threshold}"

    return {"action": action_str, "until": until}


def _rotation_action(verb: str | None = None) -> dict:
    """Turn until a rotation threshold is met."""
    v = verb or random.choice(["turns", "rotates", "spins"])
    direction = random.choice(["", "to the left", "to the right", "around"])
    action_str = f"{v} {direction}".strip()
    deg = random.choice([45, 90, 135, 180, 270, 360])

    return {"action": action_str, "until": f"rotated({deg})"}


def _build_instruction(actions: list[dict], targets: list[str]) -> str:
    """Render the action list back into a natural-language instruction.

    The mapping is deliberately many-to-one: the same instruction can be
    plausibly generated from slightly different action lists. This forces
    the LM to learn the canonical decomposition rather than memorising
    surface forms.
    """
    parts: list[str] = []

    for i, act in enumerate(actions):
        text = act["action"]
        u = act["until"]
        # Annotate with a natural-language termination phrase ~half the time
        if u.startswith("distance("):
            obj = u.split("(")[1].split(")")[0]
            op = "<" if "<" in u else ">"
            phrase = f"until reaching the {obj}" if op == "<" else f"until past the {obj}"
            text = f"{text} {phrase}"
        elif u.startswith("rotated(") and random.random() < 0.5:
            deg = u.split("(")[1].split(")")[0]
            text = f"{text} until rotated {deg} degrees"

        if i == 0:
            prefix = random.choice(INSTRUCTION_PREFIXES)
            parts.append(f"{prefix} {text}")
        else:
            parts.append(f"{random.choice(CONNECTORS)} {text}")

    return " ".join(parts).replace("  ", " ").strip()


def _sample_example(rng: random.Random) -> dict:
    """Draw one (instruction, action_list) example with random structure."""
    random.seed(rng.random())  # propagate global seed to module-level random calls
    structure = rng.choices(
        ["atomic", "two_seq", "three_seq", "distance", "rotation", "distance_then_atomic"],
        weights=[0.15, 0.25, 0.20, 0.15, 0.10, 0.15],
        k=1,
    )[0]
    targets_in_scene = rng.sample(OBJECT_NAMES, k=rng.randint(1, 3))

    if structure == "atomic":
        actions = [_atomic_action()]
    elif structure == "two_seq":
        actions = [_atomic_action(), _atomic_action()]
    elif structure == "three_seq":
        actions = [_atomic_action() for _ in range(3)]
    elif structure == "distance":
        actions = [_distance_action(rng.choice(targets_in_scene))]
    elif structure == "rotation":
        actions = [_rotation_action()]
    else:  # distance_then_atomic
        actions = [
            _distance_action(rng.choice(targets_in_scene)),
            _atomic_action(),
        ]
    instruction = _build_instruction(actions, targets_in_scene)

    return {"instruction": instruction, "actions": actions}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train", type=int, default=6000)
    parser.add_argument("--num-val", type=int, default=600)
    parser.add_argument("--output-dir", default="data/planner")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for split, n in [("train", args.num_train), ("val", args.num_val)]:
        path = out / f"{split}.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for _ in range(n):
                ex = _sample_example(rng)
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {n} examples to {path}")
    print()
    print("Sample (first 3 from train):")

    with (out / "train.jsonl").open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  {line.rstrip()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
