# python scripts/qa/action_vocabulary.py
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from types import MappingProxyType

from text2motion.motion.storage import base_clip_ids, parse_text_file

ROOT = Path("data/HumanML3D_official")
INVENTORY = Path("doc/notes/verb_inventory.csv")
LABELS = (Path("doc/notes/verb_labels_data.csv"), Path("doc/notes/verb_labels_eval.csv"))
ACTION = "action"
TYPO = "typo"
RARE_CLIPS = 150
DECISIONS = MappingProxyType(
    {
        "make": ("not_action", ""),
        "perform": ("not_action", ""),
        "act": ("not_action", ""),
        "outstretche": ("tagger_error", ""),
        "outstreche": ("tagger_error", ""),
        "rest": (ACTION, "rest"),
        "lead": ("not_action", ""),
        "hand": ("tagger_error", ""),
        "show": ("not_action", ""),
        "talk": ("not_action", ""),
        "say": ("not_action", ""),
        "call": ("not_action", ""),
        "shout": ("not_action", ""),
        "speak": ("not_action", ""),
        "lose": (ACTION, "stumble"),
        "prepare": ("not_action", ""),
        "change": ("not_action", ""),
        "work": ("not_action", ""),
        "slow": ("not_action", ""),
        "hurt": ("not_action", ""),
        "relax": ("not_action", ""),
        "meet": ("not_action", ""),
        "form": ("not_action", ""),
        "search": (ACTION, "look"),
        "examine": (ACTION, "look"),
        "inspect": (ACTION, "look"),
        "focus": ("not_action", ""),
        "wake": (ACTION, "wake"),
        "elbow": ("tagger_error", ""),
        "cry": ("not_action", ""),
        "laugh": ("not_action", ""),
        "bump": (ACTION, "bump"),
        "row": (ACTION, "row"),
        "rope": ("tagger_error", ""),
        "double": (ACTION, "bend"),
        "organize": ("not_action", ""),
        "itch": (ACTION, "touch"),
    }
)


def read_labels(path: Path) -> dict[str, dict[str, str]]:
    return {row["verb"]: row for row in csv.DictReader(path.open(encoding="utf-8"))}


def has_group(row: dict[str, str] | None) -> bool:
    return row is not None and row["status"] in (ACTION, TYPO) and bool(row["group"])


def merged_labels(verbs: list[str]) -> dict[str, tuple[str, str, str]]:
    first, second = (read_labels(path) for path in LABELS)
    merged: dict[str, tuple[str, str, str]] = {}
    for verb in verbs:
        left, right = first.get(verb), second.get(verb)
        if verb in DECISIONS:
            status, group = DECISIONS[verb]
            merged[verb] = (status, group, "decided")
            continue
        if left is None or right is None or left["status"] != right["status"]:
            raise ValueError(f"{verb}: the two label lists disagree and no decision exists")
        if not has_group(left):
            merged[verb] = (left["status"], "", "agree")
            continue
        if left["group"] == right["group"]:
            merged[verb] = (left["status"], left["group"], "agree")
            continue
        merged[verb] = (left["status"], f"{left['group']}|{right['group']}", "split")
    return merged


def named_groups(merged: dict[str, tuple[str, str, str]], clips: Counter) -> dict[str, str]:
    members: dict[str, list[str]] = defaultdict(list)
    for verb, (_, key, _) in merged.items():
        if key:
            members[key].append(verb)
    names = {}
    for verbs in members.values():
        actions = [verb for verb in verbs if merged[verb][0] == ACTION] or verbs
        name = max(actions, key=clips.__getitem__)
        for verb in verbs:
            names[verb] = name
    return names


def clip_groups(names: dict[str, str]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for clip in base_clip_ids(ROOT, "train"):
        for annotation in parse_text_file(ROOT / "texts" / f"{clip}.txt"):
            for token in annotation.tokens:
                word, _, tag = token.rpartition("/")
                if tag == "VERB" and word.lower() in names:
                    groups[names[word.lower()]].add(clip)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocabulary", type=Path, default=Path("doc/notes/verb_vocabulary.csv"))
    parser.add_argument("--groups", type=Path, default=Path("doc/notes/action_groups.csv"))
    arguments = parser.parse_args()

    rows = [row for row in csv.DictReader(INVENTORY.open(encoding="utf-8"))]
    clips = Counter({row["verb"]: int(row["train_clips"]) for row in rows})
    verbs = [row["verb"] for row in rows if clips[row["verb"]] >= 5]
    merged = merged_labels(verbs)
    names = named_groups(merged, clips)
    with arguments.vocabulary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["verb", "train_clips", "status", "group", "source"])
        for verb in verbs:
            status, _, source = merged[verb]
            writer.writerow([verb, clips[verb], status, names.get(verb, ""), source])

    group_clips = clip_groups(names)
    members: dict[str, list[str]] = defaultdict(list)
    for verb, name in names.items():
        members[name].append(verb)
    ordered = sorted(group_clips, key=lambda name: len(group_clips[name]))
    with arguments.groups.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "train_clips", "rare", "members"])
        for name in ordered:
            count = len(group_clips[name])
            writer.writerow([name, count, count < RARE_CLIPS, " ".join(sorted(members[name]))])
    rare = [name for name in ordered if len(group_clips[name]) < RARE_CLIPS]
    sources = Counter(source for _, _, source in merged.values())
    print(
        f"verbs {len(verbs)}; sources {dict(sources)}; groups {len(ordered)}; rare groups {len(rare)}"
    )


if __name__ == "__main__":
    main()
