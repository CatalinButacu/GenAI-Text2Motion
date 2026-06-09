# Streaming Text-to-Motion (SMPL-X)

Master's thesis, clean rebuild: **real-time, incremental text → whole-body SMPL-X motion**, with our
own trained generator, viewed in a studio viewer (aitviewer).

> The architecture is an open decision until ADR 0002 is accepted. See `.claude/decisions/`.
> Project conventions, the donor-asset map, and the lessons from the prior plateau live in `CLAUDE.md`.

## Layout
```
src/text2motion/{data,model,train,eval,stream,render,shared}
configs/   YAML hyperparameters (single source of truth)
scripts/   data prep / train / eval entry points
tests/     shape / round-trip / sanity tests
.claude/   agents, skills, hooks, decisions
```

## Setup
```bash
uv sync                 # core deps
uv sync --extra viewer  # aitviewer for SMPL-X studio
uv sync --extra dev      # ruff + pytest
```

## Status
Scaffolding stage — see the checklist in `CLAUDE.md`. `model/` is intentionally empty until the
architecture ADR (0002) is accepted.
