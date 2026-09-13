# Tests

Tests mirror the seven pipeline stages under `src/text2motion/`:

| directory | responsibility |
|---|---|
| `motion/` | HumanML3D-263 representation, kinematics, preparation, datasets, and caches |
| `tokenization/` | FSQ/RVQ behavior, tokenizer contracts, and token corpora |
| `generation/` | Transformer/Mamba parity, text conditioning, losses, and training |
| `evaluation/` | Frozen matcher loading and metric mathematics |
| `streaming/` | Bounded-state decode, service protocol, and stream/batch parity |
| `studio/` | Scene state and viewer composition without requiring a live window |
| `app/` | Checkpoint schemas and cross-stage application gates |

Keep a test with the stage that owns the behavior. Cross-stage composition tests belong in `app/`.
Tests must import production code, never another test module. Mark tests requiring external data or
checkpoints with `slow`, and tests requiring CUDA with `gpu`.

Run the complete suite from the repository root with `pytest`.
