# 08 — Type-checking policy

Status: **active**. Last reviewed 2026-05-21.

## TL;DR

- Pyright runs in the IDE only, in `basic` mode. **It is not in CI.**
- `beartype` enforces shape/dtype contracts at runtime where they matter.
- Type annotations are progressive: required on every new public API, optional
  inside research-y inner loops.

## Why no CI type gate

Research code mutates fast. Strict type-checking in CI forces a constant
`# type: ignore` arms race instead of doing science, because the things that
go wrong in ML aren't expressible in the type system:

- **Tensor shapes** (`(B, T, D)` vs `(B, D, T)`) — Python's type system can't
  prove them without heavy `jaxtyping`/`shaped` annotations everywhere.
- **Dtypes** (fp16 vs fp32 under autocast) — same problem.
- **Module mode** (train vs eval, frozen vs unfrozen) — runtime state, not types.

A type-gate produces noise without catching the bugs that actually cost us
money. The two real-money incidents we've had (`project_cloud_lessons_2026_05_13`
SBERT key mismatch, cache-hash drift) were both runtime contract violations
that NO type system would have caught. They're now covered by unit tests.

## What we actually use

| Layer | Tool | Where | Failure mode |
|---|---|---|---|
| IDE / dev loop | `pyright` (basic mode) | `pyrightconfig.json` | red squiggles only — never fails the build |
| Runtime tensor contracts | `beartype` | function signatures | `BeartypeCallHintParamViolation` at the call site |
| Hot-path correctness | unit tests (`tests/`) | `pytest` | fails CI |
| Style + import order | `ruff` (strict in CI) | `pyproject.toml` | fails CI |

## When to add a type annotation

**Required:** every new function signature in `src/*` that gets called from
outside its own module. Return types on anything more complex than `None`,
`int`, or a single dataclass.

**Required:** every function decorated with `@beartype` (or relying on
`@beartype.beartype` auto-application) — that's how the runtime check knows
what to check.

**Optional:** inside private helpers, list comprehensions, tight numpy loops.
Don't type-annotate `for i in range(n)`.

**Forbidden:** `Any` in public signatures. Use `object` or a `Protocol` or
narrow the type — `Any` is a lie that defeats both pyright and beartype.

## Why pyright not mypy

Both work. We picked pyright because:

- Faster in IDE (instant feedback on save vs 1-3s for mypy).
- Better default settings for project-relative imports.
- Microsoft's `ty` and Astral's `pyrefly` (both gaining traction in 2026) are
  pyright-compatible; mypy isn't.

If pyright is removed or gated on, this doc must be updated.

## Future revisit triggers

Bump the policy and consider adding a CI type-gate when any of these happen:

- We start shipping `src/` as an importable library (currently it's not).
- We add a public Python API consumed by another project.
- We introduce more than ~5 `# type: ignore` comments to silence pyright.

Until then: keep the IDE warnings visible, fix what's easy, don't fight the
type system over inner-loop tensor code.
