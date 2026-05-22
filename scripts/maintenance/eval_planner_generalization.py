"""Held-out paraphrase generalization eval for the fine-tuned planner LM.

The training set is synthesised from a fixed list of (verb, paraphrase)
templates -- so quality on the training distribution proves nothing about
generalization. This script feeds the trained planner a HAND-AUTHORED
set of paraphrased instructions whose surface forms do NOT appear in the
synthesis templates, then measures four quantities:

  1. %  valid_json     -- planner output parses as a non-empty JSON list
  2. %  valid_grammar  -- every "until" string parses via parse_condition
  3. %  canonical_hit  -- emitted action_text contains one of the expected
                          canonical verbs for that instruction
  4. mean action_count -- planner emits roughly the right number of steps

These four together are the dissertation's evaluation table for the
"natural-language input is real" claim. The held-out set lives at
tests/fixtures/planner_held_out.jsonl with 25 hand-authored examples
covering: vocabulary the LM has seen, paraphrases it has NOT seen,
multi-action chains, unusual connectors, and one verb (the
canonical_contains list) it must map to.

Run from repo root:
    python scripts/maintenance/eval_planner_generalization.py \
        --checkpoint checkpoints/planner_lm

Output: prints a per-example trace + final summary. Optionally writes
runs/planner_eval/<timestamp>/results.json for thesis Chapter 8.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
HELD_OUT = REPO_ROOT / "tests" / "fixtures" / "planner_held_out.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/planner_lm",
                        help="Fine-tuned planner LM directory")
    parser.add_argument("--held-out", default=str(HELD_OUT),
                        help="JSONL held-out instructions")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature; 0.3 keeps output focused")
    parser.add_argument("--output", default=None,
                        help="Optional JSON dump of per-example results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    from src.modules.agent.planner import ActionPlanner

    log.info("Loading planner from %s", args.checkpoint)
    planner = ActionPlanner(args.checkpoint, temperature=args.temperature)

    held_out = []

    with open(args.held_out, "r", encoding="utf-8") as f:
        for line in f:
            held_out.append(json.loads(line))
    log.info("Loaded %d held-out instructions", len(held_out))
    n_total = len(held_out)
    n_valid_json = 0
    n_valid_grammar = 0
    n_canonical_hit = 0
    sum_action_count = 0
    per_example: list[dict] = []

    for i, item in enumerate(held_out, 1):
        instr = item["instruction"]
        expected_min = item.get("expected_min_actions", 1)
        expected_canon = set(item.get("expected_canonical_contains", []))
        result = {"instruction": instr, "expected_min_actions": expected_min,
                  "expected_canonical_contains": list(expected_canon)}
        t0 = time.perf_counter()

        try:
            plan = planner(instr)
            result["plan"] = [
                {"action": a.action_text, "until": a.until.source} for a in plan
            ]
            result["valid_json"] = True
            result["valid_grammar"] = True   # parser already validated grammar
            result["action_count"] = len(plan)
            n_valid_json += 1
            n_valid_grammar += 1
            sum_action_count += len(plan)
            # Canonical-verb hit: at least one expected verb appears in some
            # action_text (substring match, case-insensitive).
            hit = any(
                any(canon.lower() in a.action_text.lower() for canon in expected_canon)
                for a in plan
            )
            result["canonical_hit"] = bool(hit)

            if hit:
                n_canonical_hit += 1
        except ValueError as e:
            result["plan"] = []
            result["valid_json"] = False
            result["valid_grammar"] = False
            result["action_count"] = 0
            result["canonical_hit"] = False
            result["error"] = str(e)[:200]
        result["elapsed_s"] = time.perf_counter() - t0
        per_example.append(result)
        marker = "OK" if result["canonical_hit"] else ("JSON" if result["valid_json"] else "FAIL")
        log.info("%3d/%-3d  [%s]  %r  ->  %d actions  (%.1fs)",
                 i, n_total, marker, instr[:60],
                 result["action_count"], result["elapsed_s"])
    summary = {
        "n_examples": n_total,
        "valid_json_pct": 100.0 * n_valid_json / n_total,
        "valid_grammar_pct": 100.0 * n_valid_grammar / n_total,
        "canonical_hit_pct": 100.0 * n_canonical_hit / n_total,
        "mean_action_count": sum_action_count / max(n_total, 1),
    }
    print()
    print("=" * 60)
    print("Generalization eval summary")
    print("=" * 60)

    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:24} {v:.1f}")
        else:
            print(f"  {k:24} {v}")

    if args.output:
        out = Path(args.output)
    else:
        out = Path(
            f"runs/planner_eval/{time.strftime('%Y%m%d-%H%M%S')}/results.json"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"summary": summary, "per_example": per_example}, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
