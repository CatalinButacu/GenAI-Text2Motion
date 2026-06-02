#!/usr/bin/env python
from __future__ import annotations

import argparse
import difflib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from scripts.evaluation.prompt_suite_schema import validate_suite

DEFAULT_THRESHOLDS = {
    "max_signature_share": 0.20,
    "min_unique_signature_ratio": 0.40,
    "max_prefix4_share": 0.20,
    "max_nearest_similarity_ge_090_share": 0.60,
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def token_variants(label: str) -> set[str]:
    base = label.lower().strip()

    if not base:
        return set()

    variants = {base}
    variants.add(f"{base}s")
    variants.add(f"{base}es")
    variants.add(f"{base}ed")
    variants.add(f"{base}ing")

    if base.endswith("e") and len(base) > 1:
        variants.add(f"{base[:-1]}ing")
        variants.add(f"{base}d")

    if base.endswith("y") and len(base) > 1:
        variants.add(f"{base[:-1]}ies")

    return variants


def prompt_signature(prompt: str, actions: list[str], entities: list[str]) -> str:
    action_set = set().union(*(token_variants(a) for a in actions)) if actions else set()
    entity_set = set().union(*(token_variants(e) for e in entities)) if entities else set()
    tokens = tokenize(prompt)
    signature_tokens: list[str] = []

    for token in tokens:
        if token in action_set:
            signature_tokens.append("<act>")
        elif token in entity_set:
            signature_tokens.append("<ent>")
        elif token.isdigit():
            signature_tokens.append("<num>")
        else:
            signature_tokens.append(token)

    return " ".join(signature_tokens)


def computeNearestSimilarities(prompts: list[str]) -> list[float]:
    nearest = []

    for i, src in enumerate(prompts):
        best = 0.0

        for j, dst in enumerate(prompts):
            if i == j:
                continue

            score = difflib.SequenceMatcher(a=src.lower(), b=dst.lower()).ratio()

            if score > best:
                best = score

        nearest.append(best)

    return nearest


def build_metrics(entries: list[dict]) -> dict:
    prompts = [entry["prompt"].strip() for entry in entries]
    signatures = [
        prompt_signature(
            entry["prompt"],
            entry.get("required_actions", []),
            entry.get("required_entities", []),
        )
        for entry in entries
    ]

    token_lists = [tokenize(prompt) for prompt in prompts]
    all_tokens = [tok for seq in token_lists for tok in seq]
    unique_prompts = len({prompt.lower() for prompt in prompts})
    unique_signatures = len(set(signatures))
    signature_counts = Counter(signatures)
    prefix4_counts = Counter(" ".join(tokens[:4]) for tokens in token_lists if tokens)

    max_signature_share = 0.0

    if signature_counts:
        max_signature_share = max(signature_counts.values()) / len(entries)

    max_prefix4_share = 0.0

    if prefix4_counts:
        max_prefix4_share = max(prefix4_counts.values()) / len(entries)

    lengths = [len(tokens) for tokens in token_lists]
    nearest_similarity = computeNearestSimilarities(prompts)

    pct_nearest_ge_080 = (
        sum(1 for s in nearest_similarity if s >= 0.80) / len(nearest_similarity)
        if nearest_similarity
        else 0.0
    )
    pct_nearest_ge_090 = (
        sum(1 for s in nearest_similarity if s >= 0.90) / len(nearest_similarity)
        if nearest_similarity
        else 0.0
    )

    return {
        "n_prompts": len(entries),
        "unique_prompt_ratio": unique_prompts / len(entries) if entries else 0.0,
        "unique_signature_ratio": unique_signatures / len(entries) if entries else 0.0,
        "max_signature_share": max_signature_share,
        "max_prefix4_share": max_prefix4_share,
        "avg_prompt_len_tokens": statistics.mean(lengths) if lengths else 0.0,
        "std_prompt_len_tokens": statistics.pstdev(lengths) if lengths else 0.0,
        "type_token_ratio": (len(set(all_tokens)) / len(all_tokens)) if all_tokens else 0.0,
        "avg_nearest_similarity": (
            statistics.mean(nearest_similarity) if nearest_similarity else 0.0
        ),
        "pct_nearest_similarity_ge_080": pct_nearest_ge_080,
        "pct_nearest_similarity_ge_090": pct_nearest_ge_090,
        "top_signatures": [
            {"signature": key, "count": count, "share": count / len(entries)}
            for key, count in signature_counts.most_common(10)
        ],
        "top_prefix4": [
            {"prefix4": key, "count": count, "share": count / len(entries)}
            for key, count in prefix4_counts.most_common(10)
        ],
    }


def evaluate_diversity(metrics: dict, thresholds: dict | None = None) -> dict:
    cfg = dict(DEFAULT_THRESHOLDS)

    if thresholds:
        cfg.update(thresholds)

    failures: list[str] = []

    if metrics["max_signature_share"] > cfg["max_signature_share"]:
        failures.append(
            f"max_signature_share {metrics['max_signature_share']:.3f} > {cfg['max_signature_share']:.3f}"
        )

    if metrics["unique_signature_ratio"] < cfg["min_unique_signature_ratio"]:
        failures.append(
            "unique_signature_ratio "
            f"{metrics['unique_signature_ratio']:.3f} < {cfg['min_unique_signature_ratio']:.3f}"
        )

    if metrics["max_prefix4_share"] > cfg["max_prefix4_share"]:
        failures.append(
            f"max_prefix4_share {metrics['max_prefix4_share']:.3f} > {cfg['max_prefix4_share']:.3f}"
        )

    if (
        metrics["pct_nearest_similarity_ge_090"]
        > cfg["max_nearest_similarity_ge_090_share"]
    ):
        failures.append(
            "pct_nearest_similarity_ge_090 "
            f"{metrics['pct_nearest_similarity_ge_090']:.3f} > "
            f"{cfg['max_nearest_similarity_ge_090_share']:.3f}"
        )

    return {
        "passed": len(failures) == 0,
        "thresholds": cfg,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit lexical/template diversity in prompt suite")
    parser.add_argument("--suite", default="data/eval/prompt_following_suite.json")
    parser.add_argument("--output", default="results/prompt_diversity_report.json")
    parser.add_argument("--max-signature-share", type=float, default=DEFAULT_THRESHOLDS["max_signature_share"])
    parser.add_argument(
        "--min-unique-signature-ratio",
        type=float,
        default=DEFAULT_THRESHOLDS["min_unique_signature_ratio"],
    )
    parser.add_argument("--max-prefix4-share", type=float, default=DEFAULT_THRESHOLDS["max_prefix4_share"])
    parser.add_argument(
        "--max-nearest-similarity-ge-090-share",
        type=float,
        default=DEFAULT_THRESHOLDS["max_nearest_similarity_ge_090_share"],
    )
    parser.add_argument("--enforce", action="store_true", help="Exit non-zero when diversity checks fail")
    args = parser.parse_args()

    suite_path = Path(args.suite)
    entries = validate_suite(json.loads(suite_path.read_text(encoding="utf-8")), min_size=1)
    metrics = build_metrics(entries)
    verdict = evaluate_diversity(
        metrics,
        thresholds={
            "max_signature_share": args.max_signature_share,
            "min_unique_signature_ratio": args.min_unique_signature_ratio,
            "max_prefix4_share": args.max_prefix4_share,
            "max_nearest_similarity_ge_090_share": args.max_nearest_similarity_ge_090_share,
        },
    )
    report = {
        "suite": str(suite_path),
        "metrics": metrics,
        "verdict": verdict,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"diversity_passed={verdict['passed']} output={output_path}")

    if args.enforce and not verdict["passed"]:
        for failure in verdict["failures"]:
            print(f"DIVERSITY FAIL: {failure}")

        sys.exit(2)


if __name__ == "__main__":
    main()
