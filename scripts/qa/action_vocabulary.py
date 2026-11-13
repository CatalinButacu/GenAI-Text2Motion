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
TIEBREAK = Path("doc/notes/verb_labels_tiebreak.csv")
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


def agreed_statuses(verbs: list[str], lists: tuple[dict, dict]) -> dict[str, str]:
    statuses = {}
    for verb in verbs:
        if verb in DECISIONS:
            statuses[verb] = DECISIONS[verb][0]
            continue
        left, right = lists[0].get(verb), lists[1].get(verb)
        if left is None or right is None or left["status"] != right["status"]:
            raise ValueError(f"{verb}: the two label lists disagree and no decision exists")
        statuses[verb] = left["status"]
    return statuses


def typo_targets(statuses: dict[str, str], lists: tuple[dict, dict]) -> dict[str, str]:
    targets = {}
    for verb, status in statuses.items():
        if status != TYPO:
            continue
        left, right = lists[0][verb]["canonical"], lists[1][verb]["canonical"]
        if left != right:
            raise ValueError(f"{verb}: the two label lists give different correct verbs")
        targets[verb] = left
    return targets


def member_sets(labels: dict, statuses: dict[str, str]) -> dict[str, frozenset[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for verb, row in labels.items():
        if statuses.get(verb) == ACTION and verb not in DECISIONS and row["group"]:
            groups[row["group"]].add(verb)
    return {verb: frozenset(members) for members in groups.values() for verb in members}


def merged_labels(verbs: list[str]) -> dict[str, tuple[str, str, str]]:
    lists = (read_labels(LABELS[0]), read_labels(LABELS[1]))
    statuses = agreed_statuses(verbs, lists)
    left_sets, right_sets = (member_sets(labels, statuses) for labels in lists)
    merged: dict[str, tuple[str, str, str]] = {}
    for verb, status in statuses.items():
        if status != ACTION or verb in DECISIONS:
            continue
        if verb not in left_sets or verb not in right_sets:
            merged[verb] = (ACTION, f"alone:{verb}", "split")
            continue
        key = f"{lists[0][verb]['group']}|{lists[1][verb]['group']}"
        source = "agree" if left_sets[verb] == right_sets[verb] else "split"
        merged[verb] = (ACTION, key, source)
    for verb, (status, group) in DECISIONS.items():
        if verb not in statuses:
            continue
        anchor = merged.get(group)
        key = anchor[1] if status == ACTION and anchor else (f"decided:{group}" if group else "")
        merged[verb] = (status, key, "decided")
    for verb, target in typo_targets(statuses, lists).items():
        key = merged[target][1] if target in merged and merged[target][0] == ACTION else ""
        merged[verb] = (TYPO, key, "typo")
    for verb, status in statuses.items():
        merged.setdefault(verb, (status, "", "agree"))
    return merged


def resolved_keys(merged: dict[str, tuple[str, str, str]], tiebreak: bool) -> dict[str, str]:
    keys = {verb: key for verb, (_, key, _) in merged.items() if key}
    if not tiebreak:
        return keys
    lists = (read_labels(LABELS[0]), read_labels(LABELS[1]))
    choices = {
        row["verb"]: row["choice"] for row in csv.DictReader(TIEBREAK.open(encoding="utf-8"))
    }
    for verb, (status, _, source) in merged.items():
        if source != "split" or status != ACTION or choices.get(verb, "0") == "0":
            continue
        labels = lists[int(choices[verb]) - 1]
        core = [
            keys[other]
            for other, (_, _, other_source) in merged.items()
            if other_source == "agree"
            and other in keys
            and labels.get(other, {}).get("group") == labels[verb]["group"]
        ]
        keys[verb] = core[0] if core else f"worker{choices[verb]}:{labels[verb]['group']}"
    return keys


def named_groups(
    merged: dict[str, tuple[str, str, str]], clips: Counter, tiebreak: bool
) -> dict[str, str]:
    members: dict[str, list[str]] = defaultdict(list)
    for verb, key in resolved_keys(merged, tiebreak).items():
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
    parser.add_argument("--tiebreak", action="store_true")
    arguments = parser.parse_args()

    rows = [row for row in csv.DictReader(INVENTORY.open(encoding="utf-8"))]
    clips = Counter({row["verb"]: int(row["train_clips"]) for row in rows})
    verbs = [row["verb"] for row in rows if clips[row["verb"]] >= 5]
    merged = merged_labels(verbs)
    names = named_groups(merged, clips, arguments.tiebreak)
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
