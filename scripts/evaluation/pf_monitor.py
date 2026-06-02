#!/usr/bin/env python
"""Per-prompt historical trend tracking and flakiness detection.

Appends per-prompt scores to results/pf_history.jsonl after each run,
then reports prompts whose score variance exceeds a threshold over the
last N runs.

Usage
-----
  python scripts/evaluation/pf_monitor.py \
      --results results/prompt_following.json \
      [--history results/pf_history.jsonl] \
      [--window 5] \
      [--variance-threshold 0.04] \
      [--output results/pf_flakiness_report.json]
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timezone
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

HISTORY_FILE = Path("results/pf_history.jsonl")
DEFAULT_WINDOW = 5
DEFAULT_VARIANCE_THRESHOLD = 0.04


def append_run(
    run_results: list[dict],
    run_id: str,
    history_file: Path = HISTORY_FILE,
) -> None:
    """Append per-prompt scores for one evaluation run to the history log."""
    history_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()

    with history_file.open("a", encoding="utf-8") as fh:
        for r in run_results:
            record = {
                "run_id": run_id,
                "timestamp": timestamp,
                "id": r.get("id", r.get("prompt", "")[:40]),
                "prompt": r.get("prompt", "")[:80],
                "combined_score": r.get("combined_score"),
                "action_hit_rate": r.get("action_hit_rate"),
                "entity_hit_rate": r.get("entity_hit_rate"),
                "forbidden_violation_rate": r.get("forbidden_violation_rate"),
                "category": r.get("category", ""),
                "difficulty": r.get("difficulty", ""),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Appended %d records to %s", len(run_results), history_file)


def load_history(history_file: Path) -> list[dict]:
    if not history_file.exists():
        return []

    records = []

    for line in history_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed history line: %s", exc)

    return records


def detect_flaky(
    records: list[dict],
    window: int = DEFAULT_WINDOW,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> list[dict]:
    """Return entries whose combined_score variance exceeds threshold in last N runs.

    Parameters
    ----------
    records:
        Full history list (all runs, all prompts).
    window:
        Number of most-recent runs to consider per prompt.
    variance_threshold:
        Sample variance threshold above which a prompt is flagged.

    Returns
    -------
    List of dicts: {prompt, category, n_runs, variance, scores, flag}.
    """
    prompt_records: dict[str, list[dict]] = {}

    for r in records:
        key = r.get("id") or r.get("prompt", "")[:40]
        prompt_records.setdefault(key, []).append(r)

    flaky = []

    for key, history in prompt_records.items():
        recent = history[-window:]

        scores = [
            float(r["combined_score"]) for r in recent
            if r.get("combined_score") is not None
        ]

        if len(scores) < 2:
            continue

        variance = float(np.var(scores, ddof=1))

        if variance > variance_threshold:
            flaky.append({
                "id": key,
                "prompt": recent[-1].get("prompt", "")[:80],
                "category": recent[-1].get("category", ""),
                "n_runs": len(scores),
                "variance": round(variance, 4),
                "scores": [round(s, 3) for s in scores],
                "flag": "flaky",
            })

    flaky.sort(key=lambda x: x["variance"], reverse=True)
    return flaky


def build_report(
    history_file: Path = HISTORY_FILE,
    window: int = DEFAULT_WINDOW,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> dict:
    records = load_history(history_file)
    flaky_list = detect_flaky(records, window=window, variance_threshold=variance_threshold)
    unique_runs = sorted({r.get("run_id", "") for r in records})

    return {
        "total_records": len(records),
        "unique_runs": len(unique_runs),
        "window": window,
        "variance_threshold": variance_threshold,
        "flaky_count": len(flaky_list),
        "flaky_prompts": flaky_list,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Prompt-following monitor and flakiness detector")
    p.add_argument("--results", default="results/prompt_following.json")
    p.add_argument("--history", default=str(HISTORY_FILE))
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--variance-threshold", type=float, default=DEFAULT_VARIANCE_THRESHOLD)
    p.add_argument("--output", default="results/pf_flakiness_report.json")
    args = p.parse_args()

    results_data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    run_results = results_data.get("results", [])
    summary_run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    history_file = Path(args.history)
    append_run(run_results, run_id=summary_run_id, history_file=history_file)

    report = build_report(
        history_file=history_file,
        window=args.window,
        variance_threshold=args.variance_threshold,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Flaky prompts: {report['flaky_count']} / {report['total_records']}")

    for entry in report["flaky_prompts"][:5]:
        print(f"  {entry['id']}  variance={entry['variance']}  scores={entry['scores']}")

    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
