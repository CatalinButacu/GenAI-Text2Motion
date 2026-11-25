# ADR 0006 — Home of the shared step runner and training engine

**Status:** Accepted (2026-11-25). Task A28-T2. The code move is task A28-T3.

**Related:** [ADR 0005](0005-generic-helpers-placement.md) (the `common` package), STATUS 2026-08-16 "PACKAGE RESTRUCTURE" (stage packages, dependencies point inward).

## Context

Epic A28 needs one step runner for the data, tokenizer, generator and evaluate workflows, and one training engine for the three trainers. The step runner exists: `Step` and `run_steps` in `src/text2motion/dataset/core/runner.py`. It uses `recipe_hash` from `dataset/core/fingerprint.py`. The manifest helpers are in `dataset/core/manifest.py`. The three modules import only the standard library.

`tokenization`, `generation` and `app` must use the runner. `generation` may not import `dataset` for infrastructure without a domain reason, and the engine must sit below `tokenization` and `generation`. Two homes were possible:

- A: a new bottom package that imports no `text2motion` package.
- B: inside `motion`, the current bottom package.

## Measurement

One session, 2026-11-25, RTX 4050 laptop, CPU only. Three copies of `src` in a temp folder: baseline, option A (`core/`), option B (`motion/`). In each copy, `runner.py`, `fingerprint.py` and `manifest.py` moved, and the imports were rewritten. Option A added `core` to `ALLOWED`. Import times were interleaved across the three copies, after one warm-up import for each copy.

| measure                                                                                              | baseline               | A: new package         | B: inside `motion`        |
| ---------------------------------------------------------------------------------------------------- | ---------------------- | ---------------------- | ------------------------- |
| `.py` files in target package (limit 10)                                                             | —                      | 4 of 10                | 10 of 10                  |
| target package after the A28-T8 engine file                                                          | —                      | 5 of 10                | 11 of 10, fails           |
| `dataset/core` files                                                                                 | 10                     | 7                      | 7                         |
| `test_package_layers.py` (2 tests) + `tests/dataset/test_fingerprint_runner.py` + `test_manifest.py` | 12 passed, 0.24 s      | 12 passed, 0.24 s      | 12 passed, 1.97 s         |
| `python -X importtime -c "import text2motion.dataset"`, median of 15                                 | 0.621 ms               | 0.610 ms               | 0.632 ms                  |
| same, `text2motion.dataset.pipeline`, median of 15                                                   | 2462 ms                | 2476 ms                | 2477 ms                   |
| same, the runner module alone, 6 runs                                                                | 35.5–40.7 ms, no torch | 34.7–37.0 ms, no torch | 1716–1778 ms, loads torch |

Reading of the table:

- `text2motion.dataset` has an empty `__init__.py`, so its import time does not change with the home.
- `dataset.pipeline` already loads torch, so the home does not change its time.
- In option B, the runner loads torch through `motion/__init__.py`, which imports `MotionScaler`. A workflow script that only runs steps then pays about 1.7 s more. The test time shows the same cost.
- Option B fills `motion` to the limit before the engine exists.

## Decision

Option A. The runner, the recipe hash, the manifest and the training engine go into the bottom package that ADR 0005 created, `common`. No second bottom package is added: one package below all others is enough, and two would hold the same kind of code.

- `common/runner.py`: `Step`, `run_steps`.
- `common/fingerprint.py`: `recipe_hash`, `output_folder`.
- `common/manifest.py`: `check_recipe_reuse`, `write_manifest`.
- The A28-T8 engine is one more file in `common`.

Rules added to the ADR 0005 rules:

- `common/__init__.py` stays empty. Then the runner does not load torch through `checkpointing.py`. The measured 35 ms holds only under this rule.
- The runner, fingerprint and manifest modules import only the standard library.

The `common` file count after ADR 0005 and this ADR: `__init__`, `files`, `checkpointing`, `runner`, `fingerprint`, `manifest`, engine = 7 of 10.

## Alternatives

1. B, inside `motion`. Rejected: 10 of 10 files now and 11 with the engine; the runner then loads torch (about 1.7 s); `motion` should own only the 263-dimension representation (ADR 0005).
2. A new package named `core`, next to `common`. Rejected: two bottom packages with the same import rule. The measured numbers for A hold for any empty-`__init__` bottom package, so the name does not change the result.
3. Keep the runner in `dataset`. Rejected: `generation` and `evaluation` would import `dataset` only for infrastructure, and `evaluation` may not import `dataset` now.

## Consequences

- `ALLOWED` in `tests/test_package_layers.py` gains `"common": frozenset()`, and every other package gains `common`. This lands in the A28-T3 commit with the move.
- `dataset/core` drops from 10 to 7 files.
- If ADR 0005 step 2 has not landed when A28-T3 starts, A28-T3 creates `common` with an empty `__init__.py`.
