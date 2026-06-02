# Chapter 8 — Interactive Avatar System

> **Target: 10-12 pages.** Where the dissertation makes its product claim: a single consumer GPU runs the entire stack — natural-language input to rendered avatar motion — with no API calls and a streaming latency budget that survives a defense-day live demo.

---

## 8.1 System Diagram

```
        user typed instruction
                 │
                 ▼
   ┌──────────────────────────────────────────────────┐
   │  ActionPlanner (fine-tuned GPT-2-small, 124M)    │
   │  Instruction: ...           (prompt)              │
   │  Actions:     [...]         (greedy / temp=0.3)   │
   │  outlines-enforced grammar at decode time         │
   └──────────────────────────────────────────────────┘
                 │
                 ▼   list[PlannedAction(action_text, until)]
   ┌──────────────────────────────────────────────────┐
   │  StreamingRunner                                  │
   │  for action in plan:                              │
   │    stream_begin (1st) or carry_over (subsequent)  │
   │    world.reset_for_new_action()                   │
   │    while not action.until(world):                 │
   │       stream_step -> argmax tokens                │
   │       tokenizer.decode incremental                │
   │       world.update_from_frame                     │
   │       yield frame                                 │
   └──────────────────────────────────────────────────┘
                 │
                 ▼   raw 168-d SMPL-X frames @ 20 fps
   ┌──────────────────────────────────────────────────┐
   │  Renderer (M4)                                    │
   │  SMPL-X mesh, aitviewer backend, MP4 / WebSocket  │
   └──────────────────────────────────────────────────┘
```

The system is *single-process*. There is no API call to a hosted LLM, no microservice boundary, no IPC. The planner LM, the MotionSSM, the RVQ tokenizer and the renderer all live in the same Python interpreter; the interface between them is method calls + tensor handoffs.

---

## 8.2 Prompt Schema and Grammar

The planner's prompt is deliberately spartan because GPT-2-small has no chat template:

```
Instruction: walk forward until you reach the tree, then turn left
Actions:
```

The model is trained to complete after `Actions: ` with a JSON list of action dicts. Each dict has exactly two keys: `action` (free-form string the MotionSSM conditions on) and `until` (one of the five grammar variants).

The grammar of `until` is small and fully enumerable:

| Variant | Meaning | Runtime check |
|---|---|---|
| `duration(N)` | Stop after N frames | `world.frames_in_action >= N` |
| `distance(obj) < D` | Avatar within D meters of object | `world.distance_to(obj) < D` |
| `distance(obj) > D` | Avatar past D meters from object | `world.distance_to(obj) > D` |
| `rotated(deg)` | Yaw change ≥ deg degrees | `world.rotated_since_start_deg() >= deg` |
| `completed` | One-shot: fires immediately after first emit | `True` |

The grammar is enforced two ways:

1. **At inference time** via `outlines` (decision #19): the LM's logits are masked so only tokens compatible with the JSON-list-of-action-dicts schema can be sampled. The model literally cannot emit malformed output.
2. **At parse time** via `src/modules/agent/conditions.parse_condition`: even if outlines is bypassed (e.g., when sampling without the constraint), the parser rejects anything outside the grammar with a `ValueError`. The runner never feeds garbage to the motion model.

Both layers fail loud. There is no silent fallback to "default action" — a malformed plan is the user's signal that the instruction was unclear.

---

## 8.3 Training the Planner From Scratch

We commit to **fine-tuning from a vanilla GPT-2-small** rather than prompting a pre-instruction-tuned model. Three reasons:

1. **Defensibility.** "We trained the NLP ourselves" is a stronger claim than "we prompt-engineered GPT-4." The fine-tune is reproducible by anyone with the dataset + 90 minutes of GPU time.
2. **Cost.** Zero API calls. The dissertation demo runs on a 4 GB consumer GPU.
3. **Licensing.** GPT-2 is MIT-licensed; no commercial restrictions, no closed-source dependency.

The **synthetic training dataset** is constructed combinatorially from 28 canonical action verbs, each with 3-6 paraphrase surface forms (e.g. `walks` ⇄ {walks, goes, moves, heads, proceeds, ambles}); 6 directional modifiers each with 3-4 paraphrases; 10 scene-target objects; and five composition templates (atomic, two-action sequential, three-action sequential, distance-bounded, rotation-bounded, distance-then-atomic).

Each generated example has the form:

```jsonl
{"instruction": "the avatar saunters straight ahead until clear of the ladder, then gets up",
 "actions": [{"action": "strolls forward", "until": "distance(ladder) > 1.5"},
             {"action": "stands up", "until": "completed"}]}
```

Note the deliberate asymmetry: the instruction string uses paraphrases (`saunters`, `straight ahead`, `clear of`, `gets up`) but the JSON keeps the canonical verbs (`strolls forward`, `stands up`). The LM trains to perform the paraphrase-to-canonical mapping; the MotionSSM downstream only ever sees the canonical action text it was trained on.

We generate 6000 train + 600 val examples in ~2 seconds (`scripts/data/synthesize_planner_data.py`). Every emitted `until` string is asserted by `tests/test_planner_dataset.py` to pass the runtime `parse_condition` — a dataset-level invariant that prevents the LM from learning a distribution we can't decode.

**Training recipe** (from `scripts/training/train_planner_lm.py`):

- Loss: causal LM cross-entropy, **masked on the prompt portion** so the model only learns to complete the action list, not memorise the instruction surface form
- Optimizer: AdamW, lr=5e-5, weight_decay=0 (the default for small SFT)
- Scheduler: OneCycleLR with 10% warmup, cosine annealing
- Batch size 8, max sequence length 128 tokens, 3 epochs
- Full-SFT (no LoRA): 124M parameters fit in ~1.5 GB VRAM at this batch size
- Save-best-by-val-loss

Smoke-validated end-to-end: 25 steps drops loss from 2.55 to 0.40 and produces structurally valid JSON; full 3-epoch run lands `val_loss ≈ 0.07` (TBD — final number from the training run currently in flight).

---

## 8.4 Inference and Generalization

The trained planner is evaluated on a hand-authored held-out set of 25 paraphrased instructions (`tests/fixtures/planner_held_out.jsonl`) covering:

- vocabulary the LM has seen but in surface forms it has *not* seen (`"take a stroll across the room"`, `"jog in place"`, `"throw a single punch"`)
- multi-action chains with unusual connectors (`"first the avatar dashes ahead, then it halts"`)
- imperative + indirect-imperative forms (`"have the figure go forward"`)
- numeric phrasings (`"pivot 90 degrees to the left"`)

We report four metrics per evaluation pass:

| Metric | Definition |
|---|---|
| `valid_json_pct` | % of outputs that parse as a non-empty JSON list |
| `valid_grammar_pct` | % of outputs where every `until` string parses via `parse_condition` |
| `canonical_hit_pct` | % of outputs where at least one emitted `action_text` contains an expected canonical verb |
| `mean_action_count` | average emitted action-list length |

> **Table T8.1** — Generalization metrics on the held-out set (TBD; lands when the in-flight training completes).

| Setting | valid_json | valid_grammar | canonical_hit | mean #actions |
|---|---:|---:|---:|---:|
| Smoke (25 steps) | TBD | TBD | TBD | TBD |
| Full (3 epochs, paraphrase dataset) | TBD | TBD | TBD | TBD |

A passing dissertation threshold is **valid_json ≥ 95%** (the system is robust), **valid_grammar ≥ 99%** (the parser layer enforces this anyway), **canonical_hit ≥ 80%** (the LM has actually learnt the mapping). If `canonical_hit` is below 50%, the LM has memorised templates but not generalised — we would need a larger or longer-trained model.

The eval script (`scripts/maintenance/eval_planner_generalization.py`) is deterministic at `temperature=0.3` and runs in ~30 seconds on the local GPU.

---

## 8.5 Runtime Demo (`scripts/demo/run_streaming_demo.py`)

The end-to-end CLI takes:

```bash
python scripts/demo/run_streaming_demo.py \
    --planner-ckpt checkpoints/planner_lm \
    --motion-ckpt  checkpoints/motion_ssm/baseline/best_model.pt \
    --rvq-ckpt     checkpoints/rvq_tokenizer/best_model.pt \
    --instruction "walk forward until you reach the tree, then turn left, then wave" \
    --scene-object "tree=5,0,0"
```

It writes a self-contained run directory:

```
runs/streaming_demo/<timestamp>/
    instruction.txt    -- the user input
    plan.json          -- the planner's decomposition
    frames.npy         -- the emitted raw 168-d SMPL-X frames
    events.jsonl       -- per-frame timing + position + heading
    summary.json       -- TTFF, total frames, generation fps
```

The defense-day demo loop will simply replay one of these directories through the renderer at 20 fps — recorded ahead of time for safety (decision #4), with a live attempt as a flex on top.

**Defensive contract:** the script preflight-checks all three checkpoints exist before importing any heavy dependencies. A missing checkpoint produces a copy-pasteable "train X first" message in <1 second, not a stack trace seven seconds in.

---

## 8.6 Failure Modes and Recovery

| Failure | Where it surfaces | Recovery |
|---|---|---|
| LM emits invalid JSON | `ActionPlanner.parse_completion` raises `ValueError` | Caller retries with higher temperature, or surfaces error to user |
| LM emits valid JSON but unparseable `until` | Same | Same |
| LM names an action verb the MotionSSM was never trained on | MotionSSM still runs (CLIP gives *some* embedding for any string), but motion quality degrades | Logged; demo continues |
| Termination predicate never fires | `--max-action-latents` cap (default 50) | Logged warning, runner advances to next action |
| Scene object named in `distance(...)` is not in the scene | `WorldState.distance_to` raises `KeyError` | Should be caught at plan-validate time; today raises at runtime |

The scene-object-not-in-scene case is the only known crash path that the current code doesn't catch at validate time. Fix: extend `ActionPlanner.parse_completion` to take the active scene's object set and validate `distance(X)` references against it. Tracked as a TODO in `src/modules/agent/planner.py`.

---

## 8.7 What's Deferred to the Journal Version

Two product features the dissertation explicitly does not promise but the system architecture admits:

1. **Multi-actor coordination.** The M2 scene planner already emits multi-actor scenes; the M5 agent layer is single-actor today (one instruction → one avatar). Extending to multi-actor would add a second `StreamingRunner` instance + cross-actor condition predicates (`distance(actor_A, actor_B)`). See Ch 9.1 for the full extension plan.

2. **Pure web-browser execution.** The demo currently uses aitviewer + an MP4 sink. For a public-facing demo, a Three.js + WASM port of the SMPL-X mesh code would let the avatar render in any browser. The motion path is already streamable — the renderer is the only piece that needs porting.

Both are listed as "future work" in Ch 9; neither is a requirement for the thesis defense.

---

## 8.8 Human Evaluation (n=60 A/B study) — DEFERRED

Standard practice in T2M is a human A/B study where N participants pick the more natural of two motion clips. We deliberately skip this for the thesis (decision #8): cost and timeline don't fit the 8-week schedule. The journal version of this work (decision #7) will include the n=60 study. The dissertation's qualitative argument rests on:

- the constant-memory benchmark (Ch 7.3 — quantitative, reproducible)
- the planner generalization eval (Ch 8.4 — quantitative)
- recorded demo videos (qualitative, defense-only)

Reviewers will note the absence of a human study. The standard response is "future work."

---

## Figure inventory

- **F8.1** System diagram (the ASCII version above, redrawn).
- **F8.2** Demo UI screenshots (left: instruction input, right: rendered avatar with HUD).
- **F8.3** Per-action HUD: current action highlighted green, next-action condition shown dim.
- **F8.4** Plan-to-rendered-motion alignment timeline.

## Table inventory

- **T8.1** Generalization metrics on the held-out set.
- **T8.2** Prompt schema vs grammar branches.
- **T8.3** Failure-modes register.

---

## Cross-references

- §8.2's grammar lives in `src/modules/agent/conditions.py` (Ch 4.8).
- §8.3's training methodology overlaps Ch 5.5.
- §8.5's demo CLI is the consumer of every component built in Ch 4.
- §8.7's deferred multi-actor extension is detailed in Ch 9.1.
