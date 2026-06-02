[![Tests](https://github.com/CatalinButacu/GenAI-Text2Motion/actions/workflows/test.yml/badge.svg)](https://github.com/CatalinButacu/GenAI-Text2Motion/actions/workflows/test.yml)

# Text-to-Motion Video Generation

Dissertation project — turn a natural-language prompt into an MP4 of an SMPL-X
character performing the described motion. The contribution is on the
text → motion side: a Mamba state-space model with an RVQ-tokenized output head,
trained on HumanML3D / AMASS.

> No physics simulation, no diffusion (ControlNet / AnimateDiff). Those modules
> were explored and removed; the runtime pipeline is text → parse → layout →
> motion → render.

## Quick Start

```bash

# uv (fast, hash-pinned, reproducible)
uv sync
uv run python -m spacy download en_core_web_sm
uv run python main.py "a person walks forward"

# Headless rendering needs the optional viewer extras (~1 GB)
uv sync --extra viewer

# Dev install (lint + tests + pre-commit)
uv sync --extra dev

# Or with pip
pip install ".[viewer]"
python -m spacy download en_core_web_sm
```

## Pipeline

```
Text Prompt
  → M1  Scene Understanding   spaCy parser → entities + actions
  → M2  Scene Planner         L-BFGS-B spatial layout (objects in 3D)
  → M4  Motion Generator      TextToMotionSSM: SBERT → BiMamba×4 → RVQ tokens
  → M6  Render Engine         SMPL-X mesh renderer (aitviewer headless) → MP4
```

`src/pipeline.py` is a ~40 LOC orchestrator that calls `invoke()` on each stage.

## Usage

```bash
# Default
python main.py "a person walks forward"

# Cap clip duration / set output fps
python main.py "a person walks and kicks a ball" --max-duration 8 --fps 30

# Sampling controls for the RVQ head
python main.py "a person dances" --temperature 1.2 --top-p 0.9

# Classifier-free guidance (requires an SBERT checkpoint trained with cfg_dropout)
python main.py "a person waves" --cfg-scale 2.0

# Candidate reranking (SBERT picks the best of N samples)
python main.py "a person punches" --rerank --num-candidates 4

# Parse + plan only, skip motion + render
python main.py "a person walks" --dry-run

# CPU only
python main.py "a person waves" --device cpu

python main.py --help
```

### Python API

```python
from src.pipeline import Pipeline
from src.shared.config import PipelineConfig

config = PipelineConfig(duration=8, fps=30)
result = Pipeline(config).render_to_file("a person walks to a ball and kicks it")
print(result["video_path"])
```

## Modules

| # | Module | Role | Key files |
|---|--------|------|-----------|
| M1 | Scene Understanding | text → `ParsedScene` | `understanding/spacy.py`, `actions.py`, `entities.py` |
| M2 | Scene Planner | `ParsedScene` → 3D layout | `planner/planner.py`, `constraint_layout.py` |
| M4 | Motion Generator | actions → SMPL-X motion clip | `motion/generator.py`, `ssm_model.py`, `nn_models.py`, `rvq_tokenizer.py` |
| M6 | Render Engine | SMPL-X poses → MP4 | `render/smplx_render.py` |

**M4 in detail.** `TextToMotionSSM` freezes a SBERT encoder
(`all-MiniLM-L6-v2`), conditions a stack of 4 FiLM-modulated bidirectional
Mamba blocks on the sentence embedding, and predicts residual codebook indices
through an RVQ head (Mogo / MoMask-style). A pre-trained tokenizer decodes the
indices back to 168-dim SMPL-X pose at 30 fps.

## Project Structure

```
main.py                          # CLI entry point
src/
├── pipeline.py                  # Thin orchestrator (~40 LOC)
├── shared/
│   ├── config.py                # PipelineConfig and per-stage configs
│   ├── constants.py             # SMPL-X spec, render and checkpoint paths
│   └── vocabulary.py            # Canonical objects, actions, properties
└── modules/
    ├── understanding/           # M1: spaCy parser → ParsedScene
    ├── planner/                 # M2: L-BFGS-B layout
    ├── motion/                  # M4: BiMamba + RVQ
    │   ├── ssm.py               #   Mamba / BiMamba layers
    │   ├── nn_models.py         #   TextToMotionSSM head
    │   ├── rvq_tokenizer.py     #   Residual VQ tokenizer
    │   └── training/            #   SSM trainers
    └── render/                  # M6: SMPL-X mesh renderer
scripts/
├── training/                    # train_motion_ssm.py, train_rvq_tokenizer.py
├── evaluation/                  # compute_metrics.py, evaluate_ablation.py
├── data/                        # HumanML3D / AMASS preparation
└── cloud/                       # AWS / Colab / SageMaker launchers
tests/
└── benchmarks/                  # benchmark_m2.py, benchmark_m4.py
```

## Data Preparation

The model trains on AMASS (raw SMPL-X) and HumanML3D (text-paired AMASS subset).
Neither is bundled; both must be downloaded under their own licenses.

```bash
# HumanML3D (text-paired) — primary training corpus
python scripts/data/download_humanml3d.py

# Inter-X (multi-person AMASS extension) — optional, for the unified pipeline
python scripts/data/download_interx.py

# Pre-compute per-channel normalization stats from your downloaded AMASS
python scripts/training/precompute_stats.py

# Quality audit (T-pose detection, velocity outliers, NaN frames)
python scripts/data/quality_stats.py

# Build the unified multi-source preprocessing cache (warms ~/.cache for training)
python scripts/data/prebuild_unified_cache.py
```

Expected layout after preparation:
```
data/
├── AMASS/                  # .npz from amass-data.is.tue.mpg.de, preprocessed to Y-up
├── humanml3d/              # text + index files
├── inter-x/                # optional multi-person dataset
├── stats/                  # precomputed normalization .npz
├── vocabulary/             # actions.yaml, objects.yaml (tracked in git)
└── .cache/                 # joblib dataset caches (gitignored)
```

> **Coordinate system note.** The pipeline expects Y-up motion everywhere
> (matches HumanML3D and aitviewer). Raw AMASS is Z-up — preprocess
> upstream before placing files under `data/AMASS/`. See the docstring at
> `src/data/amass/amass_loader.py` for the full contract.

## Training

See [.claude/docs/TRAINING.md](.claude/docs/TRAINING.md) for the full procedure
(loss curves, ablations, cloud / Terraform setup). Short version:

```bash
# Step 1 — RVQ tokenizer on AMASS / HumanML3D (frozen during step 2)
python scripts/training/train_rvq_tokenizer.py --data-dir data/AMASS

# Step 2 — MotionSSM on the tokenized motion (the headline result)
python scripts/training/train_motion_ssm.py --data-source amass \
                                            --use-sbert --bidirectional --use-film
```

Checkpoints land under `checkpoints/{rvq_tokenizer,motion_ssm}/`.
Both the RVQ tokenizer and MotionSSM checkpoints are required for inference;
missing files fail loudly at construction (no silent fallbacks).

## Evaluation

```bash
# FID against held-out HumanML3D test split
python scripts/evaluation/compute_fid.py \
    --checkpoint checkpoints/motion_ssm/best_model.pt

# Batch metrics: FID + diversity + multimodality
python scripts/evaluation/compute_metrics.py

# Ablation sweep (uses currently-trained checkpoints)
python scripts/evaluation/evaluate_ablation.py

# RVQ tokenizer reconstruction quality + codebook analysis
python scripts/evaluation/eval_rvq.py \
    --checkpoint checkpoints/rvq_tokenizer/best_model.pt
```

## Results

Current numbers on the HumanML3D held-out test split (work in progress; these
will be replaced with publication-final values once the next training pass
completes).

| Model | val_ce ↓ | top1 acc ↑ | FID ↓ | Notes |
|---|---|---|---|---|
| MotionSSM (current) | 4.60 | 12.4 % | — | Plateaued; undertrained vs MoMask baseline |
| MoMask (reference)  | ~3.5 | ~25 %  | 0.08 | Published baseline (CVPR 2024) |

See [.claude/docs/planning/](.claude/docs/planning/) for the audit and improvement
plan that followed the plateau finding (2026-05-12).

## Testing & Benchmarks

```bash
pytest tests/                    # full suite (242+ tests)
pytest tests/ -m "not slow"      # fast subset, skips integration
pytest tests/test_pipeline.py    # one-file selection

python tests/benchmarks/benchmark_m2.py
python tests/benchmarks/benchmark_m4.py
python tests/benchmarks/run_all_benchmarks.py
```

CI runs the fast suite + ruff on every push to main and every pull request.

## Requirements

- Python 3.12 (single supported version; CI tests this exclusively)
- CUDA GPU recommended for M4 training and inference
- CPU is sufficient for M1, M2, and rendering

## Methodology

The project follows CRISP-DM (Business Understanding → Data Understanding →
Data Preparation → Modeling → Evaluation → Deployment). Iterations on poor
metrics loop back to data preparation or modeling, not to deployment.

## Further reading

See [.claude/docs/planning/11_REFERENCES.md](.claude/docs/planning/11_REFERENCES.md)
for the curated bibliography (Mamba/SSM, RVQ, motion generation, CLIP,
training tricks, evaluation, visual guides). Start with the visual guides if
any of the architecture choices are unclear.

## Citation

If this work is useful in your research, please cite:

```bibtex
@mastersthesis{butacu2026text2motion,
  author  = {Butacu, Catalin},
  title   = {Text-to-Motion Video Generation with State-Space Models and
             RVQ-Tokenized Pose Output},
  school  = {Politehnica University of Bucharest},
  year    = {2026},
  url     = {https://github.com/CatalinButacu/GenAI-Text2Motion}
}
```

## License

MIT — see [LICENSE](LICENSE) for the full text.
