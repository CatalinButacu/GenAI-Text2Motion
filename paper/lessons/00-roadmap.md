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

## Part I — the tokenizer (Contribution A), as beginner theory then confirmed by values
| Lesson | Covers |
|---|---|
| **5 — RVQ tokenizer** | learned-codebook quantization from scratch: VQ -> STE/commitment/EMA/dead-code reset -> residual stacking; how WE train the strong-RVQ baseline |
| **6 — FSQ tokenizer** | mirror: fixed-grid quantization, why it needs almost no machinery, Grouped-FSQ; how WE train it |
| **7 — Tokenizer results** | the single results section: confirm both designs by their values (recon-FID, health), honest caveats, reproducibility |

## Part II — the big training process (the generator, Contribution B) — after Part I
| Lesson | Covers (user's earlier questions) |
|---|---|
| 8 — The generator architecture | "what is the architecture": embeddings + text prefix -> causal backbone (Transformer twin vs Mamba) -> heads |
| 9 — Generator training (value-by-value) | "what we train on / the output": frozen tokens + CLIP -> teacher forcing -> next-token + soft-decode loss |
| 10 — Streaming inference (value-by-value) | "the output at runtime": stream_step, bounded state vs KV-cache, CFG, END (the novelty) |
| 11 — Evaluation | "what matters / how we know": FID, R-precision, the matcher |
| 12 — What we have + what matters | the assets + the contributions/claims |

(The earlier value-by-value training/inference traces live in git history at commit d272992; they
will be rewritten as Lessons 9-10 for the generator.)

## Then (only after Part II)
Resume compute: tokenizer sweep (resume from `_last.pt`), foundational FSQ/RVQ re-runs with
manifests, the g5 canary + final 100M twin run.
