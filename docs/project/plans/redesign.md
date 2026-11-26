# Redesign plan: small modules with one role (supersedes the open A29 tasks)

Owner, 2026-11-26: "redesign from ground up ... make a good plan, step by step ... use haiku to implement, then check".

## Method

1. Move steps (M): a JSON spec in `docs/project/plans/moves/` lists `from`, `to`, `names` (and `copy` for loggers).
   `scripts/maintenance/move_symbols.py <spec>` moves the code, rewrites every import in src, tests and scripts,
   renames the size ratchet entries, and deletes the emptied file. No LLM tokens. No logic change.
2. Haiku steps (H): one fresh Haiku session for one file, with an exact spec: which function or class,
   the target shape, the tests to run. The runner adds a code graph query to the prompt.
3. Check after every step: ruff, pyright on the touched files, the tests of the touched packages,
   `tests/test_size_ratchet.py`, `tests/test_package_layers.py`, and the guards
   (`tests/generation/test_twin_guard.py`, `tests/app/test_cli_schema_guard.py`). `make check` after each group.
4. The orchestrator reviews each diff and commits. Frozen `motion` and `tokenization` are not touched.

## Target layout

```
app/                composition root
  cli/              parser.py (from cli.py), requests_*.py (from cli_requests.py, one file per stage group)
  config/           schema.py (dataclasses), loading.py (load_config and helpers), validation.py
  runtime/          bootstrap.py, context.py, checkpoint.py, overfit_gate.py, run_log.py
  stages/           data.py, tokenizer.py, generator.py, overfit.py, pretrain.py, serve.py, evaluate.py
  service/          protocol.py, state.py, stats.py, client.py, handlers.py        (R1)
  workflows/        unchanged
generation/
  model/            text_encoder.py, generator.py, decode_step.py, sampling.py, spec.py, backbones.py
  training/         optimizer.py, trainer.py, epochs.py, selection.py, overfit.py, pretrain.py
  contracts.py, requests.py, losses.py, token_datasets.py, pipeline.py, decoder.py
studio/
  session/          registry.py, buffers.py, debounce.py, state.py
  live/             the generation controller, split by role
  gui/              unchanged (A29-T19)
dataset/, evaluation/   only the files still over the limits (H steps)
```

## Steps

| step | kind | what                                                                                            | check                                   |
| ---- | ---- | ----------------------------------------------------------------------------------------------- | --------------------------------------- |
| R0   | tool | `move_symbols.py` with tests                                                                    | `tests/scripts/test_move_symbols.py`    |
| R1   | M    | `app/service.py` -> `app/service/` (ADR 0007 option A plus `stats.py`)                          | streaming, app tests                    |
| R2   | M    | `app/config.py` -> `app/config/`                                                                | app tests, guards                       |
| R3   | M    | `app/cli.py`, `app/cli_requests.py` -> `app/cli/`; update the entry point in pyproject.toml     | CLI schema guard                        |
| R4   | M    | `app/training.py`, `app/serving.py` -> `app/stages/`                                            | app, workflow tests                     |
| R5   | M    | `app/bootstrap.py`, `app/checkpoint.py`, `app/run_log.py` -> `app/runtime/`                     | app tests                               |
| R6   | M    | `generation/nn_generator.py`, `nn_backbones.py` -> `generation/model/`                          | twin guard, parity                      |
| R7   | M    | `generation/generator_trainer.py`, `pretrain.py` -> `generation/training/`                      | trainer parity                          |
| R8   | M    | `generation/contracts.py` -> `contracts.py` (configs) and `requests.py`                         | twin guard                              |
| R9   | M    | `studio/session.py`, `studio/generation.py` -> `studio/session/`, `studio/live/`                | studio tests                            |
| R10  | H    | each function over 30 lines: split into helpers of at most 30 lines, one Haiku session per file | ratchet: the name leaves the allow list |
| R11  | H    | each class over 10 methods: split by role with composition                                      | ratchet                                 |
| R12  | H    | files still over 250 lines: one more split by role                                              | ratchet                                 |
| R13  | M    | ARCHITECTURE.md, CLAUDE.md package table and naming rules; final `make check`                   | `make check`                            |

## Haiku spec form (R10 to R12)

- File: `<path>`. Target: `<name>` (<lines> lines).
- Shape: helpers of at most 30 lines, each with one role and a name that says it. Keep the public name and signature.
- No logic change. No new comments. No underscore prefixes. Fail loud.
- Run: `<tests>`, `tests/test_size_ratchet.py` (remove the name from ALLOWED), ruff, pyright on the file.
- Commit only the touched files.
