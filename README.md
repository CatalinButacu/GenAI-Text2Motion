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
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python main.py "a person walks forward"
```

## Pipeline

```
Text Prompt
  → M1  Scene Understanding   spaCy parser → entities + actions
  → M2  Scene Planner         L-BFGS-B spatial layout (objects in 3D)
  → M4  Motion Generator      TextToMotionSSM: SBERT → BiMamba×4 → RVQ tokens
  → M6  Render Engine         SMPL-X mesh renderer (OpenCV) → MP4
```

`src/pipeline.py` is a ~40 LOC orchestrator that calls `invoke()` on each stage.

## Usage

```bash
# Default
python main.py "a person walks forward"

# Custom duration / fps
python main.py "a person walks and kicks a ball" --duration 8 --fps 30

# Ablation: skip layout optimization (random placement)
python main.py "a person walks" --no-layout-opt

# Sampling controls for the RVQ head
python main.py "a person dances" --temperature 1.2 --top-p 0.9

# CPU only
python main.py "a person waves" --device cpu

python main.py --help
```

### Python API

```python
from src.pipeline import Pipeline
from src.shared.config import PipelineConfig

config = PipelineConfig(duration=8, fps=30)
result = Pipeline(config).run("a person walks to a ball and kicks it")
print(result["video_path"])   # outputs/videos/output.mp4
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

## Training

See [TRAINING.md](TRAINING.md) for the full procedure. Short version:

```bash
# M4 step 1 — RVQ tokenizer on AMASS / HumanML3D
python scripts/training/train_rvq_tokenizer.py

# M4 step 2 — MotionSSM with the frozen RVQ codebook
python scripts/training/train_motion_ssm.py

# M1 (optional) — T5 scene parser
python scripts/training/train_m1_t5.py
```

Checkpoints land under `checkpoints/{rvq_tokenizer,motion_ssm,understanding}/`.
Both the RVQ tokenizer and MotionSSM checkpoints are required for inference;
missing files fail loudly at construction (no silent fallbacks).

## Testing & Benchmarks

```bash
pytest tests/                    # full suite
pytest tests/ -m "not slow"      # skip integration tests
pytest tests/test_pipeline.py

python tests/benchmarks/benchmark_m2.py
python tests/benchmarks/benchmark_m4.py
python tests/benchmarks/run_all_benchmarks.py
```

## Requirements

- Python 3.10+
- CUDA GPU recommended for M4 training and inference
- CPU is sufficient for M1, M2, and rendering

## Methodology

The project follows CRISP-DM (Business Understanding → Data Understanding →
Data Preparation → Modeling → Evaluation → Deployment). Iterations on poor
metrics loop back to data preparation or modeling, not to deployment.

## License

MIT
