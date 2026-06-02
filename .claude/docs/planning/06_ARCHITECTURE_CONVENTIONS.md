# Pattern 6: Architecture Conventions

This document outlines the strict structural conventions for the `GenAI-Text2Motion` repository to prevent codebase fragmentation.

## 1. Hierarchical Configurations 

**Convention**: Do NOT pass sprawling keyword arguments down the pipeline (`fps`, `duration`, `use_physics`, `max_repairs`). Do NOT use flat configurations objects.

**Rule**: Every major module (e.g. `M4 Motion`, `M5 Physics`) must own its configuration rules via a localized `@dataclass`. These modules are controlled by a single master configuration (`PipelineConfig`). 

### The Structure (`src/shared/config.py`)

```python
from dataclasses import dataclass, field

@dataclass
class MotionConfig:               # Module Config
    validate_motion: bool = True
    max_repairs: int = 3

@dataclass
class PhysicsConfig:              # Module Config
    use_physics_simulation: bool = True
    fixed_camera: bool = False

@dataclass
class PipelineConfig:             # Master Config
    output_dir: str = "outputs"
    fps: int = 24
    
    # Sub-Configs nested directly inside the master
    motion: MotionConfig = field(default_factory=MotionConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
```

## 2. Module Interfaces

When setting up and calling modules from the orchestrator (`src/pipeline.py`), only pass the sub-config that the module owns:

```python
# CORRECT
def generate_motion_clips(parsed, motion_gen, duration, config: MotionConfig):
    if config.validate_motion:
        # ... logic

# INCORRECT (Do not map flat variables from a master object)
def generate_motion_clips(parsed, motion_gen, duration, validate, max_repairs):
    # ... logic
```

## 3. Top-Level Entry Points (`main.py`, `app.py`)

The entry points are the **ONLY** files permitted to interact with the raw user input (e.g., `argparse` or Gradio checkboxes). The entry point's sole responsibility is mapping user inputs into the specific module dataclasses:

```python
# main.py
motion_cfg = MotionConfig(validate_motion=args.validate, max_repairs=args.repairs)
physics_cfg = PhysicsConfig(fixed_camera=args.static_cam)

# Assemble and run
config = PipelineConfig(motion=motion_cfg, physics=physics_cfg)
result = Pipeline(config).run()
```

## Rationale
This prevents "Config Hell" where adding a single PyBullet parameter to M5 (Physics) forces you to rewrite `PipelineConfig`, `Pipeline.run`, `pipeline.setup`, and `simulate()` signatures. Instead, you modify the `PhysicsConfig` and the new variable is instantly available down to the destination function cleanly.
