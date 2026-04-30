# Physics-Constrained Video Generation

Dissertation project -- generate physics-realistic videos from text prompts.

## Quick Start

```bash
pip install -r requirements.txt
python main.py "a red ball falls onto a blue cube"
```

## Pipeline

```
Text Prompt
    v M1  Scene Understanding  -- T5: parse entities, actions, spatial relations
    v M2  Scene Planner        -- L-BFGS-B: place entities in 3D world space
    v M3  Asset Generator      -- Shap-E 3D mesh generation       (optional)
    v M4  Motion Generator     -- MotionSSM + PhysicsSSM           (optional)
    v M5  Physics Simulator    -- PyBullet rigid-body simulation (240 Hz)
    v M6  Render Engine        -- SMPL mesh + effects + MP4 export
    v M7  AI Enhancer          -- ControlNet / AnimateDiff          (optional)
    v MP4 Video
```

## Usage

```bash
# Default
python main.py "a red ball falls onto a blue cube"

# Custom duration / fps
python main.py "a person walks and kicks a ball" --duration 8 --fps 30

# Enable 3-D mesh generation (requires ~4 GB GPU)
python main.py "a wooden chair tips over" --with-assets

# Enable ControlNet enhancement (requires ~8 GB GPU)
python main.py "a sphere bounces" --with-enhance

# Disable motion generation (faster)
python main.py "boxes collide" --no-motion

python main.py --help
```

## Python API

```python
from src.pipeline import Pipeline, PipelineConfig

config   = PipelineConfig(duration=8, fps=30)
pipeline = Pipeline(config)
result   = pipeline.run("A person walks to a ball and kicks it")
print(result["video_path"])   # outputs/videos/output.mp4
```

## Module Reference

| # | Module | Status | Key files |
|---|--------|--------|-----------|
| M1 | Scene Understanding |  Trained | `t5_parser.py`, `prompt_parser.py`, `retriever.py` |
| M2 | Scene Planner |  Active | `planner.py`, `constraint_layout.py` |
| M3 | Asset Generator |  Optional | `generator.py` (Shap-E / TripoSR) |
| M4 | Motion Generator |  Trained | `ssm_generator.py`, `nn_models.py` (MotionSSM + PhysicsSSM) |
| M5 | Physics Simulator |  Active | `scene.py`, `simulator.py`, `humanoid.py` |
| M6 | Render Engine |  Active | `aitviewer_renderer.py`, `engine.py`, `smpl_body.py` |
| M7 | AI Enhancer |  Optional | `controlnet_human.py`, `animatediff_human.py` |

## Project Structure

```
main.py                        # Single entry point
src/
+ pipeline.py                # Thin orchestrator (~170 LOC)
+ stages/                    # Pipeline stages (one per concern)
|   + understanding.py       #   M1+M2: parse prompt + plan layout
|   + motion.py              #   M4: generate + refine + validate motion
|   + physics.py             #   M5: PyBullet simulation loop
|   + rendering.py           #   M6+M7: SMPL mesh rendering + video export
+ shared/
|   + constants.py           # Physics, rendering, checkpoint paths
|   + vocabulary.py          # Canonical objects, actions, properties
+ modules/
    + understanding/         # M1: Text -> ParsedScene + KB enrichment
    |   + t5_parser.py       #   T5 ML parser (default, with rules fallback)
    |   + prompt_parser.py   #   Rules-based parser (fast, no GPU)
    |   + retriever.py       #   SBERT + FAISS knowledge base lookup
    + planner/               # M2: ParsedScene -> PlannedScene
    + assets/                # M3: entity -> 3D mesh (optional)
    + motion/                # M4: action text -> motion clip
    |   + ssm/               #   Mamba / S4 / PhysicsSSM layers
    + physics/               # M5: PlannedScene -> frames + video
    + render/                # M6: SMPL mesh rendering + post-processing
    + diffusion/             # M7: ControlNet / AnimateDiff (optional)
scripts/                       # Training utilities (M1 T5, M4 SSM, PhysicsSSM)
tests/                         # Benchmark suites per module
doc/
+ planning/                  # 6 planning docs (master plan, modules, datasets, etc.)
```

## M1: Two Parser Modes

| Mode | Class | Speed | Accuracy |
|------|-------|-------|----------|
| ML (default) | `T5SceneParser` | Requires checkpoint | Higher accuracy |
| Rules (fallback) | `PromptParser` | Fast, no GPU | Good for common objects/actions |

Train the T5 extractor:

```bash
python scripts/train_m1_t5.py
```

## Requirements

- Python 3.10+
- CUDA GPU for M3 (Shap-E) and M7 (ControlNet)
- CPU-only sufficient for M1 (rules), M2, M5, M6

## License

MIT
