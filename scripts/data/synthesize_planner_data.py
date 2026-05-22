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
#
# Each verb has a CANONICAL form (what goes into the JSON action_text)
# and a list of INSTRUCTION-SURFACE paraphrases that map onto it. The
# planner LM trains to recognise every paraphrase as the same action.
# This is what gives us "real NLP" -- the test set includes paraphrases
# that never appeared verbatim in training.

VERB_PARAPHRASES: dict[str, list[str]] = {
    "walks":      ["walks", "goes", "moves", "heads", "proceeds", "ambles"],
    "runs":       ["runs", "sprints", "dashes", "races"],
    "jogs":       ["jogs", "jogs along", "runs at a jog"],
    "strolls":    ["strolls", "saunters", "wanders", "ambles"],
    "steps":      ["steps", "takes a step"],
    "marches":    ["marches", "strides"],
    "tiptoes":    ["tiptoes", "sneaks", "moves quietly"],
    "jumps":      ["jumps", "leaps", "hops up"],
    "hops":       ["hops", "skips"],
    "kicks":      ["kicks", "swings a leg", "punts"],
    "punches":    ["punches", "throws a punch", "jabs"],
    "claps":      ["claps", "applauds", "claps hands"],
    "waves":      ["waves", "waves a hand", "raises a hand to wave"],
    "bows":       ["bows", "takes a bow"],
    "stretches":  ["stretches", "reaches out", "extends arms"],
    "sits down":  ["sits down", "takes a seat", "lowers themselves", "is seated"],
    "stands up":  ["stands up", "rises", "gets up", "stands"],
    "lies down":  ["lies down", "lies flat", "lays down"],
    "turns":      ["turns", "pivots", "rotates"],
    "spins":      ["spins", "twirls", "rotates"],
    "raises arms":["raises arms", "lifts arms", "holds arms up"],
    "crosses arms":["crosses arms", "folds arms"],
    "picks up an object":["picks up an object", "grabs an object", "lifts an object"],
    "drops an object":["drops an object", "lets go of an object", "puts down an object"],
    "dances":     ["dances", "moves to music", "does a dance"],
    "swings arms":["swings arms", "moves arms back and forth"],
    "stops":      ["stops", "halts", "comes to a stop"],
    "pauses":     ["pauses", "stays still", "freezes"],
}
# Canonical form list, derived for backwards compatibility with the old
# atomic / distance / rotation paths.
ACTION_VERBS = list(VERB_PARAPHRASES.keys())

DIRECTION_PARAPHRASES: dict[str, list[str]] = {
    "forward":      ["forward", "straight ahead", "ahead", "to the front"],
    "backward":     ["backward", "back", "in reverse"],
    "to the left":  ["to the left", "left", "leftward"],
    "to the right": ["to the right", "right", "rightward"],
    "in a circle":  ["in a circle", "around in a circle", "around"],
    "diagonally":   ["diagonally", "at an angle", "off-axis"],
}
DIRECTIONS = list(DIRECTION_PARAPHRASES.keys())

OBJECT_NAMES = ["tree", "ball", "chair", "wall", "door",
                "rock", "table", "fence", "ladder", "box"]

# Paraphrases of "until you reach the X" (distance-LT) and "until past X" (distance-GT)
DIST_LT_PHRASES = ["until reaching the {obj}", "until they reach the {obj}",
                   "until close to the {obj}", "until at the {obj}",
                   "until in front of the {obj}", "until near the {obj}"]
DIST_GT_PHRASES = ["until past the {obj}", "until away from the {obj}",
                   "until clear of the {obj}", "until they pass the {obj}"]
ROTATED_PHRASES = ["until rotated {deg} degrees", "until they have turned {deg} degrees",
                   "for {deg} degrees", "after turning {deg} degrees"]

CONNECTORS = ["then", "and then", "next", ", then", "after that",
              ", followed by", "and", "; then"]

INSTRUCTION_PREFIXES = [
    "the person", "a person", "the avatar",
    "the character", "the figure", "the human",
]
# Note: prefixes are all 3rd-person singular to match the verb conjugation
# in VERB_PARAPHRASES ("walks", "runs", ...). "they" was removed for grammar
# safety.


def paraphrase_action(canonical_verb: str, canonical_direction: str = "") -> tuple[str, str]:
    """Return (canonical_action_text, paraphrased_action_text).

    The canonical text is what goes into the JSON's action field; the
    paraphrased text is what appears in the natural-language instruction.
    Both share the same direction so the LM can learn the mapping.
    """
    verb_paras = VERB_PARAPHRASES.get(canonical_verb, [canonical_verb])
    canonical_text = f"{canonical_verb} {canonical_direction}".strip()
    paraphrased_verb = random.choice(verb_paras)

    if canonical_direction:
        dir_paras = DIRECTION_PARAPHRASES.get(canonical_direction, [canonical_direction])
        paraphrased_dir = random.choice(dir_paras)
        paraphrased_text = f"{paraphrased_verb} {paraphrased_dir}".strip()
    else:
        paraphrased_text = paraphrased_verb

    return canonical_text, paraphrased_text


def atomic_action(verb: str | None = None) -> tuple[dict, str]:
    """One self-contained action with a duration- or completed-style termination.

    Returns (action_dict, paraphrased_surface) so the caller can build the
    instruction with paraphrased words while keeping the JSON canonical.
    """
    canonical_verb = verb or random.choice(ACTION_VERBS)
    is_locomotion = canonical_verb in {"walks", "runs", "jogs", "strolls", "steps",
                                        "marches", "tiptoes", "jumps", "hops",
                                        "dances", "spins"}
    canonical_dir = random.choice(DIRECTIONS) if is_locomotion else ""
    canonical_text, paraphrased = paraphrase_action(canonical_verb, canonical_dir)

    if is_locomotion and random.random() < 0.5:
        until = f"duration({random.choice([40, 60, 80, 100, 120])})"
    else:
        until = "completed"

    return {"action": canonical_text, "until": until}, paraphrased


def distance_action(target: str, verb: str | None = None) -> tuple[dict, str]:
    """Locomotion until distance threshold to a scene target is met."""
    canonical_verb = verb or random.choice(["walks", "runs", "strolls"])
    canonical_dir = random.choice(["forward", ""])
    canonical_text, paraphrased = paraphrase_action(canonical_verb, canonical_dir)
    op = random.choice(["<", ">"])
    threshold = random.choice([0.5, 1.0, 1.5, 2.0, 3.0])
    until = f"distance({target}) {op} {threshold}"
    phrase_pool = DIST_LT_PHRASES if op == "<" else DIST_GT_PHRASES
    phrase = random.choice(phrase_pool).format(obj=target)

    return {"action": canonical_text, "until": until}, f"{paraphrased} {phrase}"


def rotation_action(verb: str | None = None) -> tuple[dict, str]:
    """Turn until a rotation threshold is met."""
    canonical_verb = verb or random.choice(["turns", "rotates", "spins"])
    canonical_dir = random.choice(["", "to the left", "to the right"])
    canonical_text, paraphrased = paraphrase_action(canonical_verb, canonical_dir)
    deg = random.choice([45, 90, 135, 180, 270, 360])
    phrase = random.choice(ROTATED_PHRASES).format(deg=deg)

    return {"action": canonical_text, "until": f"rotated({deg})"}, f"{paraphrased} {phrase}"


def build_instruction(parts: list[str]) -> str:
    """Render a list of paraphrased action surface forms into a single
    natural-language instruction by gluing with random connectors and a
    random instruction prefix."""
    if not parts:
        return ""
    prefix = random.choice(INSTRUCTION_PREFIXES)
    joined = [f"{prefix} {parts[0]}"]

    for part in parts[1:]:
        joined.append(f"{random.choice(CONNECTORS)} {part}")

    return " ".join(joined).replace("  ", " ").strip()


def sample_example(rng: random.Random) -> dict:
    """Draw one (instruction, action_list) example with random structure.

    The instruction uses paraphrased surface forms drawn from
    VERB_PARAPHRASES, DIRECTION_PARAPHRASES, and the DIST_*/ROTATED_PHRASES
    pools; the action_list keeps canonical text. The LM learns the
    paraphrase-to-canonical mapping that gives us robustness to wording
    the user has never seen.
    """
    random.seed(rng.random())
    structure = rng.choices(
        ["atomic", "two_seq", "three_seq", "distance", "rotation", "distance_then_atomic"],
        weights=[0.15, 0.25, 0.20, 0.15, 0.10, 0.15],
        k=1,
    )[0]
    targets_in_scene = rng.sample(OBJECT_NAMES, k=rng.randint(1, 3))
    actions: list[dict] = []
    surface_parts: list[str] = []

    def add(builder, *args) -> None:
        act, surface = builder(*args)
        actions.append(act)
        surface_parts.append(surface)

    if structure == "atomic":
        add(atomic_action)
    elif structure == "two_seq":
        add(atomic_action)
        add(atomic_action)
    elif structure == "three_seq":
        add(atomic_action)
        add(atomic_action)
        add(atomic_action)
    elif structure == "distance":
        add(distance_action, rng.choice(targets_in_scene))
    elif structure == "rotation":
        add(rotation_action)
    else:  # distance_then_atomic
        add(distance_action, rng.choice(targets_in_scene))
        add(atomic_action)
    instruction = build_instruction(surface_parts)

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
                ex = sample_example(rng)
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
