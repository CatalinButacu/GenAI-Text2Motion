# ADR 0004 — Local Clean Architecture inside stage packages

**Status:** Accepted (2026-08-24).

## Context

The repository already has the correct coarse dependency direction:

`motion -> tokenization -> generation -> evaluation -> app`; `studio -> motion`.

The remaining problem is local: several modules mix declarative objects, algorithms, I/O, workflow
orchestration, object construction, and entrypoint parsing. Moving every class into one global
`classes/` directory would separate files by syntax rather than by concept. Microsoft’s Clean
Architecture guidance warns that generic Models/Services folders scatter business logic as a system
grows. *Architecture Patterns with Python* instead separates domain, service layer, adapters, and
entrypoints, and Mark Seemann places object graph construction in one composition root near the
application entrypoint.

Sources:

- Microsoft Learn, “Common web application architectures”: dependencies point inward; infrastructure
  implements core abstractions; startup is the composition root.
- *Architecture Patterns with Python*, chapters 4 and 13: entrypoints parse requests, application
  services orchestrate use cases, domain code owns domain behavior, and a bootstrap module performs
  manual dependency injection.
- Python Packaging User Guide, “src layout vs flat layout”: keep importable application code under
  `src/` so tests exercise the installed package rather than accidental repository-root imports.
- Mark Seemann, “Composition Root”: compose the object graph in one preferably unique location as
  close as possible to the application entrypoint.

## Decision

Use **local Clean Architecture per pipeline stage**, not a repository-wide folder by technical type.
Each stage may contain only the roles it actually needs:

- `contracts.py`: immutable dataclasses, enums, protocols, requests, results, and boundary schemas.
  No filesystem, network, model loading, training, or process startup.
- `model.py`: behavior-rich domain/model classes and their invariants. Neural-network modules remain
  here; moving their methods away from their classes would break encapsulation.
- `operations.py` or a specific operation module (`training.py`, `preparation.py`, `evaluation.py`):
  stateless algorithms and use-case functions. Dependencies are explicit parameters.
- `adapters/` only when a stage has multiple external implementations. Do not create an adapter
  directory for one trivial wrapper.
- `app/training.py` and `app/serving.py`: application use cases that coordinate stages.
  They receive a composed `ApplicationContext`; the app has no command subpackage.
- `app/bootstrap.py`: the composition root. It loads configuration, selects device/precision, creates
  repositories/models/evaluators/services, and returns ready objects.
- `app/cli.py`: an entrypoint adapter. It parses arguments, converts them to requests, dispatches one
  use case, and contains no training, evaluation, loading, or rendering algorithm.

Dependency rule:

`contracts <- model <- operations <- app commands <- cli`

External adapters depend inward on contracts. Only `app/bootstrap.py` may know concrete
implementations from multiple stages and assemble the full object graph.

## SOLID interpretation

- **SRP:** one owner per fact or workflow; a file changes for one kind of reason.
- **OCP:** variability is introduced only at proven seams such as tokenizer/backbone selection.
- **LSP:** implementations honor existing tensor and streaming contracts; no permissive fallback.
- **ISP:** small `Protocol` contracts at real boundaries, not one interface per class.
- **DIP:** use cases accept repositories, models, clients, or callables; bootstrap supplies concrete
  implementations.

## Naming rules

- Never create global `classes.py`, `utils.py`, `helpers.py`, `managers.py`, or `services.py` dumping
  grounds.
- Prefer concept names: `normalization.py`, `annotations.py`, `generation.py`, `checkpoint.py`.
- A module named `contracts.py` must be declarative. A module named `operations.py` must not define
  mutable application state.
- One class per file is not a goal. Closely related tiny value objects stay together.
- Extract a package only after a module has at least two independently changing responsibilities.

## Target structure

```text
src/text2motion/
  motion/
    contracts.py
    model.py
    storage.py
    annotations.py
    normalization.py
    datasets.py
    dataset.py
    amass.py
    preparation.py
  tokenization/
    contracts.py
    model.py
    corpus.py
    evaluation.py
    trainer.py
  generation/
    contracts.py
    model.py
    pipeline.py
    decoder.py
    losses.py
    trainer.py
  evaluation/
    contracts.py
    evaluator.py
    matcher.py
    benchmark.py
  studio/
    contracts.py
    scene.py
    generation.py
    avatar.py
    viewer.py
  app/
    __init__.py
    bootstrap.py
    checkpoint.py
    cli.py
    cli_requests.py
    config.py
    run_log.py
    service.py
    training.py
    serving.py
```

The exact file names may stay more specific than these generic roles. Existing good concept modules
must not be renamed merely for symmetry.

## Migration sequence

1. Stabilize declarative stage contracts and remove duplicate definitions.
2. Move workflows out of entrypoints into `app/training.py` and `app/serving.py`.
3. Turn `app/container.py` into `app/bootstrap.py`; pass the composed `Application` into commands.
4. Keep `cli.py` parser-only, then split parser registration only if its size still blocks navigation.
5. Separate viewer generation workers from ImGui rendering.
6. Add import-boundary tests and run focused tests after every stage.

## Consequences

- Concepts remain together while declarations, operations, adapters, and composition are visibly
  distinct.
- Tests can call use cases with fakes without launching the CLI, service, GPU, or viewer.
- The architecture remains a modular monolith: no microservices, DI framework, message bus, or one-file-
  per-class ceremony is introduced.
- Some classes remain in operational modules when the class itself is the operation (`Trainer`,
  `Decoder`, PyTorch `Module`). This is intentional encapsulation, not a violation.
