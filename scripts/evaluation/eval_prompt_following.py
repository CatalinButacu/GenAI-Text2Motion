#!/usr/bin/env python
"""Prompt-following evaluation suite.

Measures semantic adherence: what fraction of expected actions and entities
from the prompt actually appear in the parsed scene output.

Usage
-----
  python scripts/evaluation/eval_prompt_following.py \
      --suite data/eval/prompt_following_suite.json \
      --checkpoint checkpoints/motion_ssm/best_model.pt \
      --output results/prompt_following.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import sys
from pathlib import Path

import numpy as np

from scripts.evaluation.action_matcher import match_action
from scripts.evaluation.parse_cache import load_or_run
from scripts.evaluation.prompt_suite_schema import CATEGORIES, validate_suite
from src.pipeline import Pipeline
from src.shared.config import PipelineConfig
from src.shared.run_manifest import build_manifest, save_manifest
from src.shared.stats_report import bootstrap_ci

log = logging.getLogger(__name__)

SUITE_FILE = "data/eval/prompt_following_suite.json"

DEFAULT_GATE = {
    "min_global_combined": 0.85,
    "min_category_combined": 0.75,
    "max_forbidden_violation": 0.10,
    "min_order_accuracy": 0.70,
}


def load_suite(suite_file: str) -> list[dict]:
    raw_suite = json.loads(Path(suite_file).read_text(encoding="utf-8"))
    return validate_suite(raw_suite, min_size=1)


def recall_rate(expected: list[str], detected: list[str]) -> float:
    if not expected:
        return 1.0

    hits = sum(1 for exp in expected if any(match_action(exp, det) for det in detected))
    return hits / len(expected)


def precision_rate(expected: list[str], detected: list[str]) -> float:
    if not detected:
        return 1.0 if not expected else 0.0

    hits = sum(1 for det in detected if any(match_action(exp, det) for exp in expected))
    return hits / len(detected)


def order_accuracy(
    expected_actions: list[str], detected_actions: list[str], ordered: bool
) -> float | None:
    if not ordered or len(expected_actions) <= 1:
        return None

    start = 0

    for expected in expected_actions:
        found = False

        for idx in range(start, len(detected_actions)):
            if match_action(expected, detected_actions[idx]):
                start = idx + 1
                found = True
                break

        if not found:
            return 0.0

    return 1.0


def forbidden_violation_rate(
    forbidden_actions: list[str], detected_actions: list[str]
) -> float:
    if not forbidden_actions:
        return 0.0

    violations = sum(
        1 for forbidden in forbidden_actions
        if any(match_action(forbidden, det) for det in detected_actions)
    )
    return violations / len(forbidden_actions)


def temporal_relation_score(
    temporal_relations: list[dict], detected_actions: list[str]
) -> float | None:
    """Score temporal ordering relations against detected action sequence.

    For each relation {action_a, relation, action_b}, find the first occurrence
    of action_a in detected_actions, then search forward for action_b.
    Supports 'before' and 'immediately_before'; 'after' and 'overlap' are
    scored symmetrically or leniently.

    Returns None when no temporal_relations are defined.
    """
    if not temporal_relations:
        return None

    hits = 0

    for rel in temporal_relations:
        action_a = rel.get("action_a", "")
        relation = rel.get("relation", "before")
        action_b = rel.get("action_b", "")

        pos_a = next(
            (i for i, d in enumerate(detected_actions) if match_action(action_a, d)), None
        )
        pos_b = next(
            (i for i, d in enumerate(detected_actions) if match_action(action_b, d)), None
        )

        if pos_a is None or pos_b is None:
            continue

        if relation in ("before", "immediately_before"):
            satisfied = pos_a < pos_b
        elif relation == "after":
            satisfied = pos_a > pos_b
        else:
            satisfied = True

        if satisfied:
            hits += 1

    return hits / len(temporal_relations)


def score_prompt_following(prompt: str, parsed_scene, entry: dict) -> dict:
    """Score one prompt against its expected actions and entities.

    Parameters
    ----------
    prompt:
        The original text prompt.
    parsedScene:
        ParsedScene returned by the pipeline (has .actions and .entities).
    entry:
        Dataset entry dict with expected_actions and expected_entities.

    Returns
    -------
    dict with keys: detected_actions, expected_actions, action_hit_rate,
    detected_entities, expected_entities, entity_hit_rate, category.
    """
    expected_actions = [a.lower().strip() for a in entry.get("required_actions", [])]
    expected_entities = [e.lower().strip() for e in entry.get("required_entities", [])]
    forbidden_actions = [a.lower().strip() for a in entry.get("forbidden_actions", [])]
    ordered = bool(entry.get("ordered", False))
    difficulty = str(entry.get("difficulty", "medium"))

    detected_actions: list[str] = []
    detected_entities: list[str] = []

    if parsed_scene is not None:
        raw_actions = getattr(parsed_scene, "actions", []) or []
        for act in raw_actions:
            verb = getattr(act, "verb", None) or getattr(act, "action", None) or str(act)
            detected_actions.append(str(verb).lower().strip())

        raw_entities = getattr(parsed_scene, "entities", []) or []
        for ent in raw_entities:
            label = getattr(ent, "label", None) or getattr(ent, "name", None) or str(ent)
            detected_entities.append(str(label).lower().strip())

    temporal_relations = entry.get("temporal_relations") or []
    action_hit_rate = recall_rate(expected_actions, detected_actions)
    entity_hit_rate = recall_rate(expected_entities, detected_entities)
    action_precision = precision_rate(expected_actions, detected_actions)
    entity_precision = precision_rate(expected_entities, detected_entities)
    order_score = order_accuracy(expected_actions, detected_actions, ordered)
    forbidden_rate = forbidden_violation_rate(forbidden_actions, detected_actions)
    temporal_score = temporal_relation_score(temporal_relations, detected_actions)

    return {
        "prompt": prompt,
        "category": entry.get("category", ""),
        "required_actions": expected_actions,
        "expected_actions": expected_actions,
        "detected_actions": detected_actions,
        "action_hit_rate": action_hit_rate,
        "action_precision": action_precision,
        "required_entities": expected_entities,
        "expected_entities": expected_entities,
        "detected_entities": detected_entities,
        "entity_hit_rate": entity_hit_rate,
        "entity_precision": entity_precision,
        "forbidden_actions": forbidden_actions,
        "forbidden_violation_rate": forbidden_rate,
        "ordered": ordered,
        "difficulty": difficulty,
        "order_accuracy": order_score,
        "temporal_relation_accuracy": temporal_score,
        "combined_score": (action_hit_rate + entity_hit_rate) / 2.0,
    }


def evaluate_gate(summary: dict, gate: dict | None = None) -> dict:
    """Evaluate hard gate thresholds against a summary dict.

    If the summary contains bootstrap CI fields (e.g.
    ``global_combined_score_ci95_low``), the lower CI bound is compared
    against the threshold.  Otherwise the point estimate is used directly,
    preserving backward compatibility with tests that build summaries
    without CI fields.
    """
    gate_cfg = dict(DEFAULT_GATE)

    if gate is not None:
        gate_cfg.update(gate)

    checks: dict[str, dict] = {}
    failures: list[str] = []

    global_combined = float(
        summary.get(
            "global_combined_score_ci95_low",
            summary.get("global_combined_score", 0.0),
        )
    )
    checks["global_combined"] = {
        "value": global_combined,
        "threshold": gate_cfg["min_global_combined"],
        "passed": global_combined >= gate_cfg["min_global_combined"],
    }

    if not checks["global_combined"]["passed"]:
        failures.append(
            f"global_combined_score {global_combined:.3f} < {gate_cfg['min_global_combined']:.3f}"
        )

    max_forbidden = float(summary.get("global_forbidden_violation_rate", 0.0))
    checks["forbidden_violation"] = {
        "value": max_forbidden,
        "threshold": gate_cfg["max_forbidden_violation"],
        "passed": max_forbidden <= gate_cfg["max_forbidden_violation"],
    }

    if not checks["forbidden_violation"]["passed"]:
        failures.append(
            f"global_forbidden_violation_rate {max_forbidden:.3f} > {gate_cfg['max_forbidden_violation']:.3f}"
        )

    global_order = summary.get("global_order_accuracy")
    order_passed = global_order is None or global_order >= gate_cfg["min_order_accuracy"]
    checks["order_accuracy"] = {
        "value": global_order,
        "threshold": gate_cfg["min_order_accuracy"],
        "passed": order_passed,
    }

    if not order_passed and global_order is not None:
        failures.append(
            f"global_order_accuracy {float(global_order):.3f} < {gate_cfg['min_order_accuracy']:.3f}"
        )

    category_failures = []

    for cat, values in summary.get("by_category", {}).items():
        # use CI lower bound when present, fall back to point estimate
        score = values.get("combined_score_ci95_low", values.get("combined_score"))

        if score is None:
            continue

        if score < gate_cfg["min_category_combined"]:
            category_failures.append((cat, score))

    checks["category_combined"] = {
        "threshold": gate_cfg["min_category_combined"],
        "failed": [{"category": cat, "value": score} for cat, score in category_failures],
        "passed": len(category_failures) == 0,
    }

    for cat, score in category_failures:
        failures.append(
            f"category {cat} combined_score {score:.3f} < {gate_cfg['min_category_combined']:.3f}"
        )

    return {
        "passed": len(failures) == 0,
        "checks": checks,
        "failures": failures,
        "thresholds": gate_cfg,
    }


def run_suite(
    suite_file: str = SUITE_FILE,
    checkpoint_path: str = "checkpoints/motion_ssm/best_model.pt",
    output_path: str = "results/prompt_following.json",
    device: str = "cpu",
    use_cache: bool = True,
    workers: int = 1,
) -> dict:
    suite = load_suite(suite_file)
    log.info("Loaded %d suite entries from %s", len(suite), suite_file)

    cfg = PipelineConfig(device=device)
    cfg.motion.checkpoint_path = checkpoint_path
    pipeline = Pipeline(cfg)

    def parse_entry(entry: dict) -> dict:
        prompt = entry["prompt"]

        def run_parse(p: str) -> object:
            try:
                pipe_result = pipeline.parse_and_plan(p)
                return pipe_result["parsed_scene"] if pipe_result else None
            except Exception as exc:
                log.warning("parse_and_plan failed for %r: %s", p[:50], exc)
                return None

        parsed_scene = load_or_run(
            prompt, checkpoint_path, device, run_parse, use_cache=use_cache
        )
        score = score_prompt_following(prompt, parsed_scene, entry)
        log.debug(
            "[%s] action=%.2f entity=%.2f  %r",
            entry["id"],
            score["action_hit_rate"],
            score["entity_hit_rate"],
            prompt[:50],
        )
        return score

    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(parse_entry, suite))
    else:
        results = [parse_entry(e) for e in suite]

    summary = aggregate_suite_results(results)
    output = {"summary": summary, "results": results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
    log.info("Prompt-following results saved to %s", output_path)

    manifest_path = str(
        Path(output_path).parent / f"manifest_pf_{Path(output_path).stem}.json"
    )
    manifest = build_manifest(cfg, [e["prompt"] for e in suite], extra={"suite_file": suite_file})
    save_manifest(manifest, manifest_path)

    print_suite_table(summary)
    return output


def accumulateResult(
    r: dict,
    allActionRates: list, allEntityRates: list, allActionPrec: list, allEntityPrec: list,
    allCombined: list, allForbidden: list, allOrderScores: list, allTemporalScores: list,
    categoryScores: dict, categoryForbidden: dict, categoryOrderScores: dict,
    difficultyScores: dict,
) -> None:
    cat = r.get("category", "")
    difficulty = str(r.get("difficulty", "medium"))
    combined = r["combined_score"]
    allActionRates.append(r["action_hit_rate"])
    allEntityRates.append(r["entity_hit_rate"])
    allActionPrec.append(r.get("action_precision", 0.0))
    allEntityPrec.append(r.get("entity_precision", 0.0))
    allCombined.append(combined)

    forbidden = r.get("forbidden_violation_rate", 0.0)
    allForbidden.append(forbidden)

    orderScore = r.get("order_accuracy")

    if orderScore is not None:
        allOrderScores.append(orderScore)

    temporalScore = r.get("temporal_relation_accuracy")

    if temporalScore is not None:
        allTemporalScores.append(temporalScore)

    if cat in categoryScores:
        categoryScores[cat].append(combined)
        categoryForbidden[cat].append(forbidden)

        if orderScore is not None:
            categoryOrderScores[cat].append(orderScore)

    if difficulty in difficultyScores:
        difficultyScores[difficulty].append(combined)


def buildCategoryStats(
    categoryScores: dict, categoryForbidden: dict, categoryOrderScores: dict
) -> dict:
    result: dict = {}

    for cat, scores in categoryScores.items():
        catCiLow, catCiHigh = bootstrap_ci(scores) if scores else (float("nan"), float("nan"))
        result[cat] = {
            "combined_score": float(np.mean(scores)) if scores else None,
            "combined_score_ci95_low": catCiLow,
            "combined_score_ci95_high": catCiHigh,
            "forbidden_violation_rate": (
                float(np.mean(categoryForbidden[cat])) if categoryForbidden[cat] else None
            ),
            "order_accuracy": (
                float(np.mean(categoryOrderScores[cat])) if categoryOrderScores[cat] else None
            ),
        }

    return result


def aggregate_suite_results(results: list[dict]) -> dict:
    categoryScores: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}
    difficultyScores: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}
    allActionRates: list[float] = []
    allEntityRates: list[float] = []
    allActionPrec: list[float] = []
    allEntityPrec: list[float] = []
    allCombined: list[float] = []
    allForbidden: list[float] = []
    allOrderScores: list[float] = []
    allTemporalScores: list[float] = []
    categoryOrderScores: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}
    categoryForbidden: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}

    for r in results:
        accumulateResult(
            r,
            allActionRates, allEntityRates, allActionPrec, allEntityPrec,
            allCombined, allForbidden, allOrderScores, allTemporalScores,
            categoryScores, categoryForbidden, categoryOrderScores,
            difficultyScores,
        )

    globalCombinedMean = float(np.mean(allCombined)) if allCombined else 0.0
    combinedCiLow, combinedCiHigh = (
        bootstrap_ci(allCombined) if allCombined else (float("nan"), float("nan"))
    )

    summary: dict = {
        "n_prompts": len(results),
        "global_action_hit_rate": float(np.mean(allActionRates)) if allActionRates else 0.0,
        "global_entity_hit_rate": float(np.mean(allEntityRates)) if allEntityRates else 0.0,
        "global_action_precision": float(np.mean(allActionPrec)) if allActionPrec else 0.0,
        "global_entity_precision": float(np.mean(allEntityPrec)) if allEntityPrec else 0.0,
        "global_forbidden_violation_rate": (
            float(np.mean(allForbidden)) if allForbidden else 0.0
        ),
        "global_order_accuracy": float(np.mean(allOrderScores)) if allOrderScores else None,
        "global_temporal_relation_accuracy": (
            float(np.mean(allTemporalScores)) if allTemporalScores else None
        ),
        "global_combined_score": globalCombinedMean,
        "global_combined_score_ci95_low": combinedCiLow,
        "global_combined_score_ci95_high": combinedCiHigh,
        "by_category": buildCategoryStats(categoryScores, categoryForbidden, categoryOrderScores),
        "by_difficulty": {},
    }

    for difficulty, scores in difficultyScores.items():
        summary["by_difficulty"][difficulty] = float(np.mean(scores)) if scores else None

    return summary


def print_suite_table(summary: dict) -> None:
    print("\nPrompt-Following Results")
    print("-" * 50)
    print(f"  Prompts evaluated    : {summary['n_prompts']}")
    print(f"  Global action rate   : {summary['global_action_hit_rate']:.3f}")
    print(f"  Global entity rate   : {summary['global_entity_hit_rate']:.3f}")
    print(f"  Global action prec.  : {summary['global_action_precision']:.3f}")
    print(f"  Global entity prec.  : {summary['global_entity_precision']:.3f}")
    print(f"  Forbidden violation  : {summary['global_forbidden_violation_rate']:.3f}")
    order_value = summary["global_order_accuracy"]
    print(f"  Global order acc.    : {order_value:.3f}" if order_value is not None else "  Global order acc.    : N/A")
    print(f"  Global combined      : {summary['global_combined_score']:.3f}")
    print("\n  By category:")

    for cat, row in summary["by_category"].items():
        score = row["combined_score"]
        score_str = f"{score:.3f}" if score is not None else "N/A"
        forbidden = row["forbidden_violation_rate"]
        forbidden_str = f"{forbidden:.3f}" if forbidden is not None else "N/A"
        order_score = row["order_accuracy"]
        order_str = f"{order_score:.3f}" if order_score is not None else "N/A"
        print(f"    {cat:<25} score={score_str} forbid={forbidden_str} order={order_str}")

    print("\n  By difficulty:")

    for difficulty, score in summary.get("by_difficulty", {}).items():
        score_str = f"{score:.3f}" if score is not None else "N/A"
        print(f"    {difficulty:<25} score={score_str}")

    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Prompt-following evaluation suite")
    p.add_argument("--suite", default=SUITE_FILE, help="Path to prompt_following_suite.json")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/motion_ssm/best_model.pt",
        dest="checkpoint",
    )
    p.add_argument("--output", default="results/prompt_following.json")
    p.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    p.add_argument("--enforce-gate", action="store_true", help="Exit non-zero if gate fails")
    p.add_argument("--min-global-combined", type=float, default=DEFAULT_GATE["min_global_combined"])
    p.add_argument("--min-category-combined", type=float, default=DEFAULT_GATE["min_category_combined"])
    p.add_argument("--max-forbidden-violation", type=float, default=DEFAULT_GATE["max_forbidden_violation"])
    p.add_argument("--min-order-accuracy", type=float, default=DEFAULT_GATE["min_order_accuracy"])
    p.add_argument("--workers", type=int, default=1, help="Parallel scoring workers")
    p.add_argument("--no-cache", action="store_true", help="Bypass parse output cache")
    args = p.parse_args()

    output = run_suite(
        suite_file=args.suite,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        use_cache=not args.no_cache,
        workers=args.workers,
    )

    gate_result = evaluate_gate(
        output["summary"],
        gate={
            "min_global_combined": args.min_global_combined,
            "min_category_combined": args.min_category_combined,
            "max_forbidden_violation": args.max_forbidden_violation,
            "min_order_accuracy": args.min_order_accuracy,
        },
    )
    gate_path = Path(args.output).with_name(f"{Path(args.output).stem}_gate.json")
    gate_path.write_text(json.dumps(gate_result, indent=2), encoding="utf-8")
    log.info("Gate report saved to %s", gate_path)

    if args.enforce_gate and not gate_result["passed"]:
        for failure in gate_result["failures"]:
            log.error("Gate failure: %s", failure)

        sys.exit(2)


if __name__ == "__main__":
    main()
