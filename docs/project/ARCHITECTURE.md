# Architecture -- top-down, measured

Enforced by `tests/test_package_layers.py` (source of truth) and
`tests/app/test_import_boundaries.py` (a wider envelope for `evaluation`).
The source is split into eight packages under `src/text2motion/`.

## The layering

Static dependency inspection gives an acyclic package graph; runtime imports are not checked here.

```
  L6  app          configuration, composition, checkpoints, CLI commands, workflows, service IPC
  L5  evaluation   Guo metrics and streaming benchmarks
  L4  generation   Mamba/transformer twins, text encoder, training, bounded decode, steering
  L3  tokenization FSQ/RVQ tokenizer, corpus encoding, training
  L2  dataset      source preparation, splits, loaders, test fixture
      studio       SMPL-X fitting and the aitviewer studio (imports motion and common only)
  L1  motion       HumanML3D-263 contract, kinematics
  L0  common       step runner, recipe hash, manifest, training engine
```

Allowed dependencies (each package may also import itself):

| package | may import |
|---|---|
| `common` | nothing |
| `motion` | nothing |
| `dataset` | `common`, `motion` |
| `studio` | `common`, `motion` |
| `tokenization` | `common`, `motion`, `dataset` |
| `generation` | `common`, `motion`, `dataset`, `tokenization` |
| `evaluation` | `common`, `motion`, `generation` |
| `app` | all packages; composition root |

Each package holds at most 10 Python files. A subpackage takes the overflow.

`app/` has exactly ten Python modules:
`__init__.py`, `bootstrap.py`, `checkpoint.py`, `cli.py`, `cli_requests.py`,
`config.py`, `run_log.py`, `service.py`, `training.py`, and `serving.py`.
The `app/workflows/` subpackage holds the `run` command.
`service.py` owns the wire protocol, client, server startup, and inference server.
`training.py` owns preparation and training commands; `serving.py` owns serving and evaluation.

## Role responsibilities (one sentence each)

| package | owns | must NOT |
|---|---|---|
| `common` | `Step` and `run_steps`, recipe hash, manifest, the shared training engine | import another package |
| `motion` | 263 representation and joint recovery | know about tokenizers, generators, metrics, or rendering |
| `dataset` | preparation, splits, loaders, the CPU test fixture | know about tokenizers or generators |
| `tokenization` | tokenizer architectures, corpus encoding, reconstruction training/eval | know about generators or application wiring |
| `generation` | twins, encoder, losses, training, decode | import evaluation, studio, or app |
| `evaluation` | frozen Guo matcher, FID/R-precision/Diversity, benchmarks | mutate models or write checkpoints |
| `studio` | aitviewer meshes/skeletons and SMPL-X fitting | duplicate motion representation math |
| `app` | config, checkpoints, composition, CLI, service IPC | leak wiring into lower stages |

## Workflows, training engine and steering (epic A28)

- `text2motion run` runs one workflow: `data`, `tokenizer`, `generator` or `evaluate`.
  `app/workflows/steps.py` builds each as a list of `Step` objects over the CLI handlers.
  One step list serves both twins. `run_steps` skips a finished step by recipe hash (ADR 0006).
- `common/engine.py` `run_epochs` is the one epoch loop of `tokenizer_trainer`,
  `generator_trainer` and `pretrain`. It holds the clean validation cross-entropy history.
  Parity values: `tests/generation/trainer_parity.json`.
- `generation/decoder.py` `steer_stream` changes the prompt inside a stream: `REANCHOR` or
  `CROSSFADE`. The service command `steer` uses re-anchor and reports the joint jump in cm.
  Studio live mode sends `steer` on a prompt change. The choice is provisional: it was measured
  on a tiny random model only (`docs/notes/steering_joint_jump.csv`).

## The ">5 collaborators" rule

Only three classes exceed a fan-out of 5 project classes:

| class | fan-out | verdict |
|---|---:|---|
| `ApplicationConfig` | 9 | ALLOWED -- aggregate root of config dataclasses; that IS its job |
| `GeneratorTrainer` | 7 | BORDERLINE -- coordinates model+tokenizer+encoder+EMA+losses |
| `StreamingStudioViewer` | 6 | ALLOWED -- composition root; 5 collaborators + the `Viewer` base class |

Everything else is at or below 5. The rule holds, with composition roots exempt by design
(see "Composition roots" below): a root's job IS to assemble, so counting its wiring against it
would penalise the very pattern that keeps every other class small.

## Composition roots (the only places allowed to wire many things)

Entrypoints assemble; libraries do not. The 20 `main()` scripts are the composition roots, plus:
- `ApplicationContext` / `ApplicationBootstrap` -- the single place runtime dependencies and
  evaluation resources are built.
- `TextToMotionGenerator` -- bundles the token generator, text encoder, and tokenizer.
- `ApplicationConfig` / `load_config` + `validate_config` -- the single place configuration is resolved.

## The studio viewer (converted from mixins to composition, 2026-07-28)

Was `StreamingStudioViewer(GuiPanels, GenerationControl, Avatar, Display, Viewer)` -- five namespaces
sharing one object's mutable state. Now `StreamingStudioViewer(Viewer)` holding four components plus
one shared state object:

```
  StreamingStudioViewer(Viewer)          composition root; delegates aitviewer hooks
    +-- StudioSession                    all shared mutable state (settings + 4 state objects)
    +-- StreamSceneUpdater(host, state)  scene nodes, follow-cam, frame sync
    +-- SmplxAvatarController(host, state)       SMPL-X fitting + remesh workers
    +-- StudioGenerationController(host, state, avatar) prompt -> stream -> chunks, live regen
    +-- GuiPanels(host, state, avatar, generation)   imgui panels
```

`host` is the aitviewer `Viewer` (scene, toggle_animation, center_view_on_node); `state` is
`StudioSession`. Components reach siblings by explicit reference, never by MRO.

| class | attrs before | attrs after | fan-out |
|---|---:|---:|---:|
| `StreamingStudioViewer` | 68 | **9** | 6 |
| `GuiPanels` | 55 | **13** | 0 |
| `StudioGenerationController` | 48 | **6** | 1 |
| `SmplxAvatarController` | 39 | **9** | 1 |
| `StreamSceneUpdater` | 23 | **7** | 0 |
| `StudioSession` (new) | -- | 29 | 5 |

The components now construct against a stub host, so they are unit-testable without opening a
window (`tests/studio/test_studio_composition.py`). That was impossible under the mixin design.

## Known debt

1. `StudioSession` at 29 attributes is the remaining aggregation -- it is deliberately a data holder,
   but the sampling knobs (temperature/top_p/cfg_scale/steps) could become a `SamplingCfg` and the
   model-registry fields (`models`/`model_idx`/`loading_model`/`load_status`) a `ModelSelection`.
2. 20 entrypoints is a lot; several `scripts/eval/*` are one-off analyses that could fold together.
3. The viewer has never been launched since the refactor -- imports, static resolution and stub-host
   construction are verified, but the live window is not.

## Re-check commands

```powershell
make check
make types
.\.venv\Scripts\python.exe -m pytest tests/test_package_layers.py tests/app/test_import_boundaries.py -q
```
