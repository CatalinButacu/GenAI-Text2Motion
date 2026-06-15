# Lessons roadmap — understand the whole training system before resuming compute

Compute is PAUSED (2026-06-15). We resume runs (tokenizer sweep, foundational re-runs, final twin)
only after every component below is explained and understood. Each step = one lesson = one turn,
with definitions + paper notes + check questions, grounded in our actual code.

## Already covered
- **Lesson 1** — a pose: joints, skeleton, rotations (positions vs rotations, FK vs IK).
- **Lesson 2** — pose -> motion: fps, velocity, why 20 fps.
- **Lesson 3** — the 263 vector, slice by slice (root / ric / rot6d / vel / foot).
- **Lesson 4** — the tokenizer (Contribution A): quantizer landscape, FSQ, why 6x1000, log health.
- **Applied A** — the loss + optimizer (term split, FK-consistency, AdamW recipe).

## Remaining steps (the user's questions, mapped)

| Step | Lesson | Covers (user's words) |
|---|---|---|
| 1 | **5 — The architecture** | "what is the architecture" — the generator: token embeddings + text prefix -> causal backbone (Transformer twin vs Mamba) -> per-codebook heads |
| 2 | **6 — What we train on + the output** | "what we training on / what is the output" — inputs (frozen tokens + CLIP text), target (next token), teacher forcing, output (tokens -> decoded motion) |
| 3 | **7 — Streaming inference** | "the output at runtime" — stream_step, bounded SSM state vs growing KV-cache, CFG sampling, END token (the thesis novelty) |
| 4 | **8 — Evaluation** | "the things which matter / how we know" — FID, R-precision, the Guo matcher, what each number means |
| 5 | **9 — What we have + what matters** | "what we have" — the assets (checkpoints, results, the system end-to-end) and the contributions/claims |

## Then (only after step 5)
Resume compute: tokenizer sweep (resume from `_last.pt`), foundational FSQ/RVQ re-runs with
manifests, the g5 canary + final 100M twin run.
