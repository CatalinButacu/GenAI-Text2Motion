from __future__ import annotations

import random
from collections import Counter

CATEGORIES = (
    "single_action",
    "multi_step",
    "style_modifier",
    "negation",
    "object_interaction",
)

DIFFICULTY_LEVELS = ("easy", "medium", "hard")


class PromptSuiteValidationError(ValueError):
    pass


def as_lower_str_list(value: object, field_name: str, entry_id: str) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise PromptSuiteValidationError(
            f"entry {entry_id}: field '{field_name}' must be a list, got {type(value).__name__}"
        )

    out: list[str] = []

    for item in value:
        if not isinstance(item, str):
            raise PromptSuiteValidationError(
                f"entry {entry_id}: field '{field_name}' contains non-string item: {item!r}"
            )

        text = item.strip().lower()

        if text:
            out.append(text)

    return out


def as_bool(value: object, field_name: str, entry_id: str, default: bool = False) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    raise PromptSuiteValidationError(
        f"entry {entry_id}: field '{field_name}' must be bool, got {type(value).__name__}"
    )


VALID_TEMPORAL_RELATIONS = frozenset({"before", "after", "overlap", "immediately_before"})


def as_difficulty(value: object, entry_id: str) -> str:
    if value is None:
        return "medium"

    if isinstance(value, str) and value in DIFFICULTY_LEVELS:
        return value

    raise PromptSuiteValidationError(
        f"entry {entry_id}: difficulty must be one of {DIFFICULTY_LEVELS}, got {value!r}"
    )


def normalize_temporal_relations(value: object, entry_id: str) -> list[dict]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise PromptSuiteValidationError(
            f"entry {entry_id}: 'temporal_relations' must be a list, got {type(value).__name__}"
        )

    result: list[dict] = []

    for i, rel in enumerate(value):
        if not isinstance(rel, dict):
            raise PromptSuiteValidationError(
                f"entry {entry_id}: temporal_relations[{i}] must be a dict"
            )

        action_a = str(rel.get("action_a", "")).strip().lower()
        relation = str(rel.get("relation", "")).strip().lower()
        action_b = str(rel.get("action_b", "")).strip().lower()

        if not action_a or not action_b:
            raise PromptSuiteValidationError(
                f"entry {entry_id}: temporal_relations[{i}] missing action_a or action_b"
            )

        if relation not in VALID_TEMPORAL_RELATIONS:
            raise PromptSuiteValidationError(
                f"entry {entry_id}: temporal_relations[{i}] invalid relation {relation!r}; "
                f"expected one of {sorted(VALID_TEMPORAL_RELATIONS)}"
            )

        result.append({"action_a": action_a, "relation": relation, "action_b": action_b})

    return result


def normalize_entry(entry: dict, index: int) -> dict:
    if not isinstance(entry, dict):
        raise PromptSuiteValidationError(
            f"entry at index {index} must be an object, got {type(entry).__name__}"
        )

    entry_id = str(entry.get("id", f"row_{index + 1:04d}")).strip()
    prompt = str(entry.get("prompt", "")).strip()
    category = str(entry.get("category", "")).strip()

    if not entry_id:
        raise PromptSuiteValidationError(f"entry {index}: missing non-empty 'id'")

    if not prompt:
        raise PromptSuiteValidationError(f"entry {entry_id}: missing non-empty 'prompt'")

    if category not in CATEGORIES:
        raise PromptSuiteValidationError(
            f"entry {entry_id}: invalid category {category!r}; expected one of {CATEGORIES}"
        )

    required_actions = as_lower_str_list(
        entry.get("required_actions", entry.get("expected_actions", [])),
        "required_actions",
        entry_id,
    )
    required_entities = as_lower_str_list(
        entry.get("required_entities", entry.get("expected_entities", [])),
        "required_entities",
        entry_id,
    )
    forbidden_actions = as_lower_str_list(
        entry.get("forbidden_actions", []),
        "forbidden_actions",
        entry_id,
    )
    style_constraints = as_lower_str_list(
        entry.get("style_constraints", []),
        "style_constraints",
        entry_id,
    )
    tags = as_lower_str_list(entry.get("tags", []), "tags", entry_id)

    ordered = as_bool(entry.get("ordered", False), "ordered", entry_id, default=False)
    difficulty = as_difficulty(entry.get("difficulty", "medium"), entry_id)
    temporal_relations = normalize_temporal_relations(
        entry.get("temporal_relations"), entry_id
    )
    notes = str(entry.get("notes", ""))

    normalized = {
        "id": entry_id,
        "prompt": prompt,
        "category": category,
        "required_actions": required_actions,
        "required_entities": required_entities,
        "forbidden_actions": forbidden_actions,
        "ordered": ordered,
        "style_constraints": style_constraints,
        "difficulty": difficulty,
        "tags": tags,
        "temporal_relations": temporal_relations,
        "notes": notes,
        # Backward-compatible aliases
        "expected_actions": required_actions,
        "expected_entities": required_entities,
    }

    return normalized


def validate_suite(entries: list[dict], min_size: int = 1) -> list[dict]:
    if not isinstance(entries, list):
        raise PromptSuiteValidationError(
            f"prompt suite must be a list, got {type(entries).__name__}"
        )

    normalized = [normalize_entry(entry, i) for i, entry in enumerate(entries)]

    if len(normalized) < min_size:
        raise PromptSuiteValidationError(
            f"prompt suite too small: got {len(normalized)}, need at least {min_size}"
        )

    ids = [entry["id"] for entry in normalized]
    prompts = [entry["prompt"].strip().lower() for entry in normalized]

    duplicate_ids = [item for item, c in Counter(ids).items() if c > 1]
    duplicate_prompts = [item for item, c in Counter(prompts).items() if c > 1]

    if duplicate_ids:
        raise PromptSuiteValidationError(f"duplicate ids in suite: {duplicate_ids[:5]}")

    if duplicate_prompts:
        raise PromptSuiteValidationError(
            f"duplicate prompts in suite: {duplicate_prompts[:5]}"
        )

    return normalized


def category_counts(entries: list[dict]) -> dict[str, int]:
    counts = Counter(entry["category"] for entry in entries)
    return {cat: counts.get(cat, 0) for cat in CATEGORIES}


def compute_bucket_quota(
    bucket_sizes: dict[str, int], total_pick: int
) -> dict[str, int]:
    if total_pick < 0:
        raise ValueError("total_pick must be non-negative")

    total = sum(bucket_sizes.values())

    if total_pick > total:
        raise ValueError(f"Cannot pick {total_pick} from only {total} entries")

    if total == 0:
        return dict.fromkeys(bucket_sizes, 0)

    raw = {key: total_pick * size / total for key, size in bucket_sizes.items()}
    floor_quota = {key: int(value) for key, value in raw.items()}
    remainder = total_pick - sum(floor_quota.values())

    ranked = sorted(
        raw.keys(),
        key=lambda key: (raw[key] - floor_quota[key], bucket_sizes[key]),
        reverse=True,
    )

    for key in ranked:
        if remainder <= 0:
            break

        if floor_quota[key] < bucket_sizes[key]:
            floor_quota[key] += 1
            remainder -= 1

    return floor_quota


def split_golden_canary(
    entries: list[dict], canary_size: int = 40, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    normalized = validate_suite(entries, min_size=1)

    if canary_size <= 0:
        raise ValueError("canary_size must be > 0")

    if canary_size >= len(normalized):
        raise ValueError(
            f"canary_size must be smaller than suite size ({len(normalized)})"
        )

    grouped: dict[str, list[dict]] = {cat: [] for cat in CATEGORIES}

    for entry in normalized:
        grouped[entry["category"]].append(entry)

    for cat_entries in grouped.values():
        cat_entries.sort(key=lambda row: row["id"])

    quota = compute_bucket_quota(
        {cat: len(grouped[cat]) for cat in CATEGORIES},
        total_pick=canary_size,
    )
    rng = random.Random(seed)
    canary: list[dict] = []
    golden: list[dict] = []

    for cat in CATEGORIES:
        rows = list(grouped[cat])
        rng.shuffle(rows)
        k = quota[cat]
        canary.extend(rows[:k])
        golden.extend(rows[k:])

    canary.sort(key=lambda row: row["id"])
    golden.sort(key=lambda row: row["id"])
    return golden, canary
