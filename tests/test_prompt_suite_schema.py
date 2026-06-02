from __future__ import annotations

from scripts.evaluation.prompt_suite_schema import (
    PromptSuiteValidationError,
    split_golden_canary,
    validate_suite,
)


def test_split_golden_canary_sizes_and_disjoint():
    rows = []

    for i in range(50):
        rows.append(
            {
                "id": f"sa_{i}",
                "prompt": f"single action prompt {i}",
                "category": "single_action",
                "required_actions": ["walk"],
                "required_entities": ["person"],
                "difficulty": "easy",
            }
        )

    for i in range(50):
        rows.append(
            {
                "id": f"ms_{i}",
                "prompt": f"multi step prompt {i}",
                "category": "multi_step",
                "required_actions": ["walk", "run"],
                "required_entities": ["person"],
                "ordered": True,
                "difficulty": "medium",
            }
        )

    for i in range(35):
        rows.append(
            {
                "id": f"st_{i}",
                "prompt": f"style prompt {i}",
                "category": "style_modifier",
                "required_actions": ["walk"],
                "required_entities": ["person"],
                "difficulty": "hard",
            }
        )

    for i in range(35):
        rows.append(
            {
                "id": f"ng_{i}",
                "prompt": f"negation prompt {i}",
                "category": "negation",
                "required_actions": ["walk"],
                "required_entities": ["person"],
                "forbidden_actions": ["run"],
                "difficulty": "hard",
            }
        )

    for i in range(30):
        rows.append(
            {
                "id": f"oi_{i}",
                "prompt": f"object interaction prompt {i}",
                "category": "object_interaction",
                "required_actions": ["kick"],
                "required_entities": ["person", "ball"],
                "difficulty": "medium",
            }
        )

    validated = validate_suite(rows, min_size=200)
    golden, canary = split_golden_canary(validated, canary_size=40, seed=42)

    assert len(golden) == 160
    assert len(canary) == 40

    golden_ids = {row["id"] for row in golden}
    canary_ids = {row["id"] for row in canary}
    assert golden_ids.isdisjoint(canary_ids)


def test_validate_suite_rejects_duplicate_prompts():
    rows = [
        {
            "id": "a",
            "prompt": "same prompt",
            "category": "single_action",
            "required_actions": ["walk"],
            "required_entities": ["person"],
        },
        {
            "id": "b",
            "prompt": "same prompt",
            "category": "single_action",
            "required_actions": ["walk"],
            "required_entities": ["person"],
        },
    ]

    try:
        validate_suite(rows, min_size=1)
        assert False, "Expected PromptSuiteValidationError"
    except PromptSuiteValidationError:
        pass
