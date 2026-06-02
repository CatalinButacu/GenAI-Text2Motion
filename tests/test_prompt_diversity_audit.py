from __future__ import annotations

from scripts.evaluation.audit_prompt_diversity import (
    build_metrics,
    evaluate_diversity,
    prompt_signature,
)


def test_prompt_signature_masks_action_and_entity():
    sig = prompt_signature("A person walks with a ball", ["walk"], ["person", "ball"])
    assert "<act>" in sig
    assert sig.count("<ent>") == 2


def test_build_metrics_returns_expected_fields():
    entries = [
        {
            "prompt": "a person walks quickly",
            "required_actions": ["walk"],
            "required_entities": ["person"],
        },
        {
            "prompt": "a person runs quickly",
            "required_actions": ["run"],
            "required_entities": ["person"],
        },
    ]
    metrics = build_metrics(entries)

    assert metrics["n_prompts"] == 2
    assert "unique_signature_ratio" in metrics
    assert "max_signature_share" in metrics
    assert "pct_nearest_similarity_ge_090" in metrics
    assert "avg_nearest_similarity" in metrics
    assert "top_signatures" in metrics


def test_evaluate_diversity_fails_templated_pool():
    entries = [
        {
            "prompt": f"a person walks quickly pattern {i}",
            "required_actions": ["walk"],
            "required_entities": ["person"],
        }
        for i in range(20)
    ]
    metrics = build_metrics(entries)
    verdict = evaluate_diversity(
        metrics,
        thresholds={
            "max_signature_share": 0.10,
            "min_unique_signature_ratio": 0.80,
            "max_prefix4_share": 0.10,
            "max_nearest_similarity_ge_090_share": 0.10,
        },
    )

    assert verdict["passed"] is False
    assert len(verdict["failures"]) >= 1
