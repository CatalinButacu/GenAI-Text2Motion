---
name: repo-scaffold
description: >
  The canonical layout and conventions for this clean repo, so structure stays consistent and
  architecture-agnostic until ADR 0002 lands. Use when creating new modules/dirs, deciding where
  code belongs, or setting up packaging/config.
---

# Repo scaffold & conventions

## Layout (architecture-agnostic until ADR 0002)
```
src/text2motion/
  data/        # loaders, representation, normalization, augmentation, splits
  model/       # generator(s) + tokenizer — POPULATED ONLY AFTER ADR 0002
  train/       # trainer loop, losses, EMA, schedules
  eval/        # FID / R-precision / diversity / streaming benchmark
  stream/      # streaming decode loop + bounded-state contract
  render/      # SMPL-X layer + aitviewer studio
  shared/      # config, constants (SMPL-X spec), seed, paths
configs/       # YAML; one source of truth for hyperparameters
scripts/       # entry points for data prep / train / eval (heavy ops live here)
tests/         # shape/round-trip/sanity tests
.claude/       # agents, skills, hooks, decisions, docs
```

## Conventions
- **Config-driven.** No magic numbers in code — they live in `configs/*.yaml`, loaded into typed
  dataclasses in `shared/config.py`.
- **One representation contract** (`motion-representation`) imported everywhere; never inline a
  second motion layout.
- **Fail loud.** No `strict=False` loads, no `try/except: pass`, no silent fallbacks. Missing
  checkpoints/data raise at construction.
- **Paths via config/env**, never hardcoded; data + model files are license-gated and gitignored.
- Python 3.12, ruff (line length 100), enforced by the `post_edit` hook. Tests with pytest;
  mark slow/gpu tests so the fast suite stays runnable.
- `model/` stays empty until ADR 0002 is accepted — keep the repo honest about what's decided.

## First scaffold actions
1. `pyproject.toml` (deps: torch, smplx, numpy; extras: viewer→aitviewer, dev→ruff+pytest).
2. `shared/constants.py` (SMPL-X spec) + `shared/config.py` + `shared/paths.py` (donor data roots).
3. `.gitignore` (data/, checkpoints/, *.zip, .venv, caches).
4. Stub the package dirs with `__init__.py` and a `README` line each describing its single job.
