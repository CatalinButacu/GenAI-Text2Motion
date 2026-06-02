# 09 — Configuration management policy

Status: **active**. Last reviewed 2026-05-21.

## TL;DR

- **Current standard:** `@dataclass(slots=True)` configs + `argparse` CLIs.
- **Future:** migrate to `pydantic.BaseModel` when nested validation hurts.
- **Don't:** Hydra. Premature for a single-paper repo; we'll regret the
  file-tree convention and the `os.chdir` to outputs dir.

## Why dataclass + argparse, today

This is what the codebase has. It works at our scale:

- `PipelineConfig`, `MotionConfig`, `PlannerConfig`, `TrainingConfig`,
  `RVQConfig`, `SSMConfig` all live in `src/shared/config.py` and per-module
  `config.py` files.
- Argparse defines the CLI surface in each `scripts/training/*.py` and
  `scripts/inference/*.py`; the parsed namespace is mapped onto the relevant
  dataclass.
- camelCase field names are enforced by the style checker
  (`tests/test_style_rules.py`).

Pros: zero new dependencies, IDE autocomplete works out of the box, dataclass
defaults map 1-to-1 to argparse defaults, beartype + ruff catch most field
mismatches.

Cons that we accept for now:

- No runtime validation of CLI-supplied values (e.g. `--lr -1.0` is accepted
  silently; trainer crashes later).
- Nested configs (`config.motion.ssm.dModel`) require some `getattr`
  navigation when consumed.
- Loading a config from a YAML file requires writing a parser by hand.

## When dataclass falls over (and what to use next)

The next-step migration is **pydantic v2** (`pydantic.BaseModel`), NOT Hydra.
Triggers for adopting pydantic:

1. We grow a config-from-YAML loader for ablations.
2. A bug ships to production because a CLI flag was a typo / wrong type and
   nothing caught it.
3. Nested configs exceed 3 levels deep.

Pydantic gives us, for cheap:

- Field validators (`@field_validator("lr")` returning `lr > 0`).
- YAML/JSON round-trip via `model_dump_json` / `model_validate_json`.
- Free CLI generation with `tyro` or `simple-parsing` (no argparse boilerplate).
- camelCase aliases via `alias_generator=to_camel`.

If we go pydantic, we replace argparse with `tyro` in the same PR — they're
designed to compose.

## Why not Hydra

Hydra is the tempting next step because every CV/NLP paper repo uses it. We've
decided against it for this project:

1. **Working-directory rewriting**: Hydra `os.chdir`s into `outputs/<run_id>/`
   by default. That conflicts with our `runs/` convention, our cwd-relative
   data paths in `src/data/dataset_cache.py`, and the cloud bootstrap scripts.
2. **File-tree convention**: `conf/model/*.yaml`, `conf/data/*.yaml`, etc.
   pushes config out of `src/` and into a parallel directory that drifts from
   the code that consumes it.
3. **Multi-run sweep machinery**: useful when you ablate ~50+ configurations.
   We're at ~5. The cost-benefit doesn't flip yet.
4. **CamelCase**: Hydra's structured configs are awkward to combine with the
   camelCase rule enforced project-wide.

If we ever genuinely need composable config sweeps (e.g. ablating every pair
of {3 SSM widths} × {4 RVQ depths} × {2 text encoders}), Hydra becomes
defensible. Until then, named scripts under `scripts/training/` are clearer.

## What to actively avoid

- **`hydra-core` as a transitive dep** sneaking in via something else. Pin it
  out if it shows up.
- **`omegaconf`** without Hydra — has its own footguns (interpolation in
  string values, structured config drift).
- **JSON for configs**: no comments, awkward to edit by hand. YAML or Python
  dataclass.
- **Class-level mutable defaults** (`fields: list = []`). Use
  `dataclasses.field(default_factory=list)`. Pre-commit catches this anyway.

## Migration path when the trigger fires

1. Convert `PipelineConfig` and `TrainingConfig` to `pydantic.BaseModel`,
   keep field names + defaults identical.
2. Add `@field_validator` for the obvious nonsense (`lr > 0`, `batch_size > 0`,
   `device in {"cpu", "cuda"}`).
3. Replace argparse in `scripts/training/train_motion_ssm.py` with
   `tyro.cli(TrainingConfig)`. Verify the help text is acceptable.
4. Migrate one module's config at a time; the dataclass and pydantic configs
   can coexist during the transition.

No big-bang rewrite.
