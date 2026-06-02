# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Text-to-Motion Video Generation: a natural-language prompt -> MP4 pipeline for a dissertation project. The system parses the prompt into entities/actions, plans a 3D layout, generates SMPL-X human motion via a Mamba state-space model with an RVQ-tokenized output head, and renders the result.

**There is no physics simulation stage and no diffusion (ControlNet / AnimateDiff) stage in the runtime pipeline.** Those modules were explored and removed. The thesis contribution is on the text -> motion side, with a streaming/constant-memory claim (see `stream_step` in `src/modules/motion/`).

## Commands

```bash
# Install (uv is the supported workflow; pip is the fallback)
uv sync                                  # core deps
uv sync --extra viewer                   # adds aitviewer for headless SMPL-X render
uv sync --extra dev                      # lint + tests + pre-commit
uv run python -m spacy download en_core_web_sm

# Run the pipeline
python main.py "a person walks forward"
python main.py "a person walks and kicks a ball" --duration 8 --fps 30
python main.py "a person walks" --no-layout-opt           # ablation: random layout
python main.py "a person dances" --temperature 1.2 --top-p 0.9   # RVQ sampling controls

# Tests
pytest tests/
pytest tests/ -m "not slow"              # skip integration tests (need checkpoints/ffmpeg)
pytest tests/test_pipeline.py            # single test file
pytest tests/test_pipeline.py::test_name # single test
PROFILE_MEMORY=1 pytest tests/...        # memory profiling (needs memory-profiler)

# Benchmarks
python tests/benchmarks/benchmark_m2.py
python tests/benchmarks/benchmark_m4.py
python tests/benchmarks/run_all_benchmarks.py

# Training (two-step: tokenizer, then SSM)
python scripts/training/train_rvq_tokenizer.py --data-dir data/AMASS
python scripts/training/train_motion_ssm.py --data-source amass --use-sbert --bidirectional --use-film

# Evaluation
python scripts/evaluation/compute_fid.py     --checkpoint checkpoints/motion_ssm/best_model.pt
python scripts/evaluation/compute_metrics.py                       # FID + diversity + multimodality
python scripts/evaluation/evaluate_ablation.py                     # sweep over current checkpoints
python scripts/evaluation/eval_rvq.py        --checkpoint checkpoints/rvq_tokenizer/best_model.pt

# Style / validation
python scripts/validation/check_style.py    # enforces the naming rules below across the repo
```

## Architecture

`src/pipeline.py` is a ~40 LOC orchestrator that sequences four module stages:

```
Text Prompt
  -> understanding.invoke()  (M1): parse entities + actions (spaCy parser)
  -> planner.invoke()        (M2): L-BFGS-B spatial layout for objects
  -> motion.invoke()         (M4): generate SMPL-X motion clips via TextToMotionSSM
  -> render.invoke()         (M6): SMPL-X mesh renderer -> MP4
```

**Entry points:**
- `main.py` -- CLI, constructs `PipelineConfig` and calls `Pipeline.run()`
- `src/pipeline.py` -- thin orchestrator
- `src/shared/config.py` -- `PipelineConfig` and per-stage configs
- `src/shared/constants.py` -- canonical values for rendering, checkpoints, SMPL-X body model
- `src/shared/vocab/` -- canonical objects, actions, properties (backed by `data/vocabulary/*.yaml`, tracked in git). Import via `from src.shared.vocab import ACTIONS, OBJECTS, ...`.

**Key design choices:**
- M1 default parser is `SpacyParser` (spaCy NLP). A regex-only `PromptParser` is the zero-ML fallback. Both live in `src/modules/understanding/`.
- M4 production backend is `TextToMotionSSM` in `src/architecture/nn_models.py`: frozen SBERT (`all-MiniLM-L6-v2`) -> FiLM-conditioned BiMamba x4 -> RVQ-tokenized output head (Mogo / MoMask-style residual vector quantization).
- Motion output is 168-dim SMPL-X pose per frame at 30 fps, produced by decoding the predicted RVQ codebook indices through a pre-trained tokenizer.
- SMPL-X constants (55 joints, 10475 vertices, 168-dim pose) are fixed by the body model spec -- do not change them.
- **Streaming contract:** `MotionSSM.stream_step` and the causal RVQ path are the thesis novelty. Keep the constant-memory invariant intact when editing motion code -- the benchmark (`doc-equivalent under .claude/`) shows 36x lower memory at T=4000 vs the non-streaming baseline.
- **Fail-fast policy:** no silent fallbacks, no `strict=False` checkpoint loads, no `try/except` that swallows errors. Missing checkpoints raise `FileNotFoundError` at construction.
- **Coordinate system:** the pipeline assumes Y-up everywhere (matches HumanML3D and aitviewer). Raw AMASS is Z-up -- preprocess upstream of `data/AMASS/`.

**Module locations:**
- `src/modules/understanding/` -- M1: `parser.py`, `strategies/` (spaCy + optional T5), `retriever.py` (SBERT+FAISS KB lookup)
- `src/modules/planner/` -- M2: `planner.py`, `constraint_layout.py` (L-BFGS-B spatial layout)
- `src/architecture/` -- neural net: `nn_models.py` (TextToMotionSSM), `rvq_tokenizer.py`, `ssm.py` (Mamba/BiMamba layers), `streaming.py`, `training/` (base_trainer, trainer, trainer_utils)
- `src/modules/motion/` -- M4 module orchestration: `generator.py` and `ssm_model.py` (wrap architecture), `clip_ops.py`, `blend.py`, `reranker.py`, `retrieval.py`, `models.py`
- `src/modules/render/` -- M6: `smplx_render.py` (SMPL-X OpenCV / aitviewer renderer)

**Data & checkpoints:**
- Training data: `data/AMASS/` (Y-up SMPL-X), `data/humanml3d/` (text-motion pairs), `data/inter-x/` (multi-agent, optional)
- Normalization: `data/stats/` (precomputed via `scripts/training/precompute_stats.py`)
- Vocabulary: `data/vocabulary/actions.yaml`, `data/vocabulary/objects.yaml` (tracked)
- Joblib caches: `data/.cache/` (gitignored; warm via `scripts/data/prebuild_unified_cache.py`)
- Checkpoints: `checkpoints/understanding/scene_extractor_v5`, `checkpoints/motion_ssm/best_model.pt`, `checkpoints/rvq_tokenizer/best_model.pt`
- Older checkpoints from prior worktrees are archived under `_from_worktrees/` -- archived, not live.

## Code Style

- Max line length: **100 chars**, enforced by ruff (config in `pyproject.toml`)
- Python 3.12 is the single supported version (CI tests this exclusively)
- `pyproject.toml` is the single source of truth for ruff + pytest config; ruff excludes `doc/`, `scripts/cloud/`, `*.ipynb`
- Prefer compact, short functions; no docstrings on internal helpers; no backwards-compat shims

### Mandatory naming and structure rules (apply to all new and edited code)

1. **No underscore-prefixed names.** Functions and variables must not start with `_`. Dunder methods (`__init__`, `__len__`, etc.) are exempt.
   - Wrong: `_contact_mask`, `_eval_loop`, `_TRANS_SLICE`
   - Right: `contactMask`, `evalLoop`, `TRANS_SLICE`

2. **camelCase for variables and functions.** Local variables and module-level helpers use lowerCamelCase. Class names stay PascalCase. Constants stay UPPER_SNAKE_CASE. External API parameters (PyTorch, numpy, etc.) are excluded.
   - Wrong: `total_loss`, `run_train_epoch`, `my_var`
   - Right: `totalLoss`, `runTrainEpoch`, `myVar`

3. **Blank line before and after control flow and significant module calls.** One blank line before/after every `if`, `for`, `while`, and before/after significant external calls. Ruff removes extras in some contexts -- don't fight the formatter.

4. **All imports at the top of the file.** No `import` inside functions, conditionals, or class bodies. Exceptions: `if TYPE_CHECKING:` blocks and module-level optional-dependency try/except guards.

5. **Variable isolation -- pass minimum information.** Functions receive only the fields they need, not whole config objects.
   - Wrong: `def computeLoss(config): use config.lr`
   - Right: `def computeLoss(lr: float): ...`

6. **No excessive try/except.** Only wrap true system boundaries (user-supplied files, subprocesses, external network calls). Never wrap internal calls in `try/except Exception: pass`. Fail loudly.
   - Wrong: `try: wandb.log(m) except Exception: pass`
   - Right: `try: np.load(user_path) except (OSError, BadZipFile): log.warning(...)`

7. **Read actual code before acting.** Always grep/read to confirm a name, function, or pattern exists before renaming, removing, or referencing it. Never infer from memory alone.

8. **Use scripts and terminal commands for batch/expensive operations.** Operations spanning multiple files or scanning data dirs go through a script or shell command, not per-file Read calls. When a tool call fails, handle it inline -- do NOT retry the same failing call.
   - Wrong: Read() 15 files one by one to check naming violations
   - Right: `python scripts/validation/check_style.py`

## Documentation policy

This repo keeps the documentation surface minimal at the root:
- `README.md` -- the user-facing entry point (install, quick-start, results table)
- `CLAUDE.md` -- this file, the Claude Code briefing

**Do not create any other `.md` file outside `.claude/`.** Notes, plans, audits, dissertation chapters, research synthesis, runbooks -- all of that belongs under `.claude/` (or stays as conversation context). Do not write `NOTES.md`, `PLAN.md`, `TODO.md`, or dissertation chapters into the project tree. Third-party READMEs under `data/<dataset>/` and auto-generated reports under `checkpoints/.../eval/` are exceptions because they aren't authored here.

## Claude Code Tooling

**Slash commands** (`.claude/commands/`): `/test`, `/benchmark`, `/pipeline`, `/training-status`, `/train`, `/research`, `/deploy`, `/aws-train`, `/gcp-train`.

**Hooks** (`.claude/hooks/post_edit.py`): auto-runs `ruff format` + `ruff check --fix` after every Python file edit.

**Pre-commit** (`.pre-commit-config.yaml`): ruff + trailing-whitespace + large-file checks on `git commit`. Install with `pip install pre-commit && pre-commit install`.

## Testing Notes

- Slow / integration tests are marked `@pytest.mark.slow` -- skip with `-m "not slow"`
- GPU-only tests are marked `@pytest.mark.gpu` -- skip with `-m "not gpu"`
- Integration tests that require trained checkpoints are guarded with `@pytest.mark.skipif`
- `pyproject.toml` sets `log_cli = true` at INFO, so test output includes pipeline logs
- CI (`.github/workflows/test.yml`) runs the fast suite + ruff on every push to main and every PR
