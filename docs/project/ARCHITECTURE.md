# Architecture -- top-down, measured

Grounded in the code and enforced by `tests/app/test_import_boundaries.py`, not aspirational.
The source is split into six stages under `src/text2motion/`.

## The layering

Static dependency inspection gives an acyclic package graph; runtime imports are not checked here.

```
  L5  app          configuration, composition, checkpoints, CLI commands, service IPC
  L4  evaluation   Guo metrics and streaming benchmarks
      studio       SMPL-X fitting and the aitviewer studio
  L3  generation   Mamba/transformer twins, text encoder, training, bounded decode
  L2  tokenization FSQ/RVQ tokenizer, corpus encoding, training
  L1  motion       HumanML3D-263 contract, data, preparation, kinematics
```

Allowed dependencies:

| package | depends on |
|---|---|
| `motion` | `motion` |
| `tokenization` | `motion`, `tokenization` |
| `generation` | `motion`, `tokenization`, `generation` |
| `evaluation` | `motion`, `tokenization`, `generation`, `evaluation` |
| `studio` | `motion`, `studio` |
| `app` | all stages; composition root |

The table lists the allowed dependency envelope; the boundary test rejects imports outside it.

`app/` is flat, with exactly ten Python modules:
`__init__.py`, `bootstrap.py`, `checkpoint.py`, `cli.py`, `cli_requests.py`,
`config.py`, `run_log.py`, `service.py`, `training.py`, and `serving.py`.
`service.py` owns the wire protocol, client, server startup, and inference server.
`training.py` owns preparation and training commands; `serving.py` owns serving and evaluation.

## Role responsibilities (one sentence each)

| package | owns | must NOT |
|---|---|---|
| `motion` | 263 representation, preparation, datasets, joint recovery | know about tokenizers, generators, metrics, or rendering |
| `tokenization` | tokenizer architectures, corpus encoding, reconstruction training/eval | know about generators or application wiring |
| `generation` | twins, encoder, losses, training, decode | import evaluation, studio, or app |
| `evaluation` | frozen Guo matcher, FID/R-precision/Diversity, benchmarks | mutate models or write checkpoints |
| `studio` | aitviewer meshes/skeletons and SMPL-X fitting | duplicate motion representation math |
| `app` | config, checkpoints, composition, CLI, service IPC | leak wiring into lower stages |

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

```
python - <<'PY'   # cycles + layer violations
import ast,pathlib,collections
pkg=collections.defaultdict(set)
for p in pathlib.Path('src/text2motion').rglob('*.py'):
    if '__pycache__' in str(p): continue
    parts=p.relative_to('src/text2motion').parts
    src=parts[0] if len(parts)>1 else 'root'
    for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
        if isinstance(n,ast.ImportFrom) and (n.module or '').startswith('text2motion'):
            seg=(n.module or '').split('.'); dst=seg[1] if len(seg)>1 else 'root'
            if dst!=src: pkg[src].add(dst)
edges={(a,b) for a,v in pkg.items() for b in v}
print('cycles:',[(a,b) for a,b in edges if (b,a) in edges and a<b] or 'NONE')
order=['motion','tokenization','generation','evaluation','studio','app']
rank={n:i for i,n in enumerate(order)}
print('violations:',[(a,b) for a,b in edges if rank.get(b,-1)>rank.get(a,99)] or 'NONE')
PY
```
