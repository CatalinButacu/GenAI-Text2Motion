# Streaming Text-to-Motion (SMPL-X) -- clean rebuild

A fresh, disciplined rebuild of a master's thesis system: **real-time, incremental
text -> whole-body SMPL-X motion**, with our own trained generator, viewed in a studio
viewer (aitviewer). This repo replaces a prior attempt that was structurally fine but
whose model **plateaued** (top-1 ~12% vs MoMask ~25%, FID never computed). We keep the
prior project's *assets and lessons*, not its model. Read this file first, every session.

## Hard requirements (locked; architecture is NOT -- see below)
1. **Text -> human motion**, with our own trained model. Two tracks (ADR 0001):
   PRIMARY = standard **HumanML3D-263** (body-only) for citable FID; DEFERRED = **SMPL-X 168**
   whole-body (body+hands; face deferred) for the studio demo. See `motion-representation`.
2. **Real-time / incremental generation** in chunks with a bounded-memory streaming step.
   This is the thesis novelty -- any architecture we pick MUST support causal streaming or
   be rejected. (Disqualifies full-sequence diffusion as the *runtime* generator.)
3. **Studio visualization via aitviewer**, fed live by the streaming decoder. No website.

## Thesis -- two contributions + reuse strategy (ACCEPTED, see ADR 0001)
Train from scratch what we *claim* novelty on; reuse everything else so numbers stay comparable.
**Two contributions** (research-grounded -- see `.claude/docs/references.md`): (A) a **Residual-FSQ
motion tokenizer** on 263 (FSQ is motion-proven by ScaMo / arXiv:2508.08991; residual by MoMask -- the
*combination* is novel), benchmarked vs a strong RVQ+EMA/reset and MoMask's RVQ; (B) the **first
token-autoregressive S6/Mamba motion generator** (an unoccupied literature cell), vs a causal-AR
transformer twin (T2M-GPT mold) -- claim = bounded recurrent state vs growing KV-cache at long
horizons, at matched FID. Reference ceiling: MoMask 0.045 / Mogo 0.079.
- **Eval matcher**: **reuse** Guo et al. (`donor data\t2m\text_mot_match\model\finest.tar`),
  **unmodified**, standard 263 input. Never retrain -- keeps FID comparable to the field.
- **Baseline**: MoMask / T2M-GPT -- **reuse published numbers** as the reference ceiling; reuse
  MoMask's released RVQ as a tokenizer baseline to beat. Do not retrain a baseline generator.
- **Tokenizer (Contribution A)**: build a **better** RVQ on 263 -- do not just inherit one.
- **Generator (Contribution B)**: train from scratch; the controlled twin is transformer vs SSM
  under identical tokenizer/data/budget. Claim is on the **streaming/efficiency axis**.
- **Staging:** primary HumanML3D-263 (citable FID) now; SMPL-X 168 whole-body demo deferred.

## Architecture is an OPEN decision -- do not pre-commit
Candidates (autoregressive token / residual-masked / SSM-Mamba-S6 / physics-constrained SSM /
diffusion-as-baseline) are debated by `motion-model-architect` via the `arch-decision` skill:
a candidate x constraint matrix, adjudicated with a real FID on a small run, recorded as an ADR
in `.claude/decisions/`. Until ADR 0002 is accepted, no architecture-specific code is "blessed."

## Donor project -- READ-ONLY resource at `D:\Facultate\dissertation`
Port selectively (`port-from-donor` skill). Never copy its model/training core wholesale -- that
core is what plateaued.
- **Data (~290 GB, prior run used <10%):** `data\amass` (151G), `data\humanml3d` (33G),
  `data\inter-x` (45G), `data\arctic`, `data\pahoi`, `data\stats`, `data\vocabulary`,
  `data\t2m` (eval matcher), `data\models_smplx_v1_1.zip` (SMPL-X bodies), `data\skel_models`.
- **Reference code to re-implement clean:** `src\data\*`, `src\modules\render\*`,
  `src\modules\runtime\*`, `scripts\evaluation\*`.
- **Lessons:** `.claude\docs\TRAINING_DIAGNOSIS_AND_PHYSICS.md`, `...\research\SSM_INTEGRATION.md`.

## Lessons from the plateau -- bake into every modeling decision
Full data mix + mirror aug * EMA 0.999 * train 100-200+ epochs * losses beyond token-CE
(soft-decode recon + velocity + foot/root) * don't fully freeze the text encoder * non-greedy
sampling * scale the generator (50-150M, not 5M) * **compute FID early and often**.

## Tooling map
- `.claude/agents/` -- 6 specialists, one per workstream. Delegate.
- `.claude/skills/` -- recipes; invoke the matching skill before coding that stage.
- `.claude/decisions/` -- ADRs. ADR 0002 (architecture) gates all model code.
- `.claude/hooks/` -- `post_edit.py` auto-formats/lints Python after every edit.

## Package layout -- one package per pipeline stage, dependencies point inward
`motion` -> `tokenization` -> `generation` -> `evaluation` / `streaming` / `studio` -> `app`.
Nothing imports a stage above itself (enforceable: zero upward imports today). `app` is the only
layer allowed to construct anything.
- `motion/` -- `model.py` (domain objects), `representation.py` (263 layout + recovery),
  `kinematics.py`, `dataset.py` (`MotionRepository`, `Split`), `preparation.py` (AMASS -> 263).
- `tokenization/` -- `model.py` (FSQ/RVQ + `MotionTokenizer` facade), `trainer.py`, `corpus.py`,
  `evaluation.py`.
- `generation/` -- `model.py` (backbones + `GeneratorArchitecture`), `text.py`, `losses.py`,
  `trainer.py`, `pretrain.py`, `pipeline.py` (`MotionGenerator` facade).
- `evaluation/` -- `metrics.py`, `matcher.py` (frozen Guo evaluator), `evaluator.py`, `benchmark.py`.
- `streaming/` -- `decoder.py`, `service.py`, `protocol.py`. `studio/` -- `avatar.py`, `scene.py`, `viewer.py`.
- `app/` -- `config.py`, `checkpoint.py` (external schemas), `runtime.py`, `run_log.py`,
  `container.py` (`ApplicationFactory`), `cli.py` (every stage is a subcommand).
Stage boundaries exchange named objects (`MotionClip`, `MotionTokens`, `MotionBatch`,
`GeneratedMotion`, `MotionChunk`), never bare tuples. Backbone/quantizer selection lives in one
registry per subsystem, not in scattered `if backbone == ...` branches.

## Golden rules
- Before any stage, open its skill. Before modeling, ADR 0002 must be accepted.
- **Overfit one batch before any real run** (`sanity-overfit`) -- catches plateaus on day 1.
- Assert tensor shapes at module boundaries; canonical shapes live in `motion-representation`.
- Data/SMPL-X files are license-gated -- never commit; load from configured paths.
- **snake_case** (PEP8) everywhere -- the donor's camelCase is NOT inherited. Python 3.12, ruff (line 100).
- **Config-driven via typed dataclasses** (`app/config.py`): instantiate one `Config`, pass a
  function only the sub-config it needs. No module-level global constants, no magic numbers in code.
- **Guard every local run** (`scripts/training/local_guard.ps1`): launch long local jobs WITH the watchdog --
  stall-kill on a stale `metrics.jsonl` heartbeat + a max-hours budget (the laptop twin of the
  cloud cost guards; a frozen tokenizer run once burned 10.6 h unnoticed). Inspect
  `outputs/GUARD_KILL.txt` before any relaunch. Long python jobs always run with `-u`.
- **Log every run** (`app/run_log.py`): `start_run()` at every train/eval entrypoint (config +
  git commit + seed + versions manifest), `log_metrics()` per epoch/eval. Any number quoted in an
  ADR/STATUS/the dissertation must trace to a run dir. Model selection on **val** only; `test` is
  touched once per final table (the 20-rep `text2motion evaluate` protocol).

## Status
- [ ] Repo scaffolded * [ ] Donor assets mapped * [ ] Motion representation round-trips
- [ ] Unified dataset loads at scale * [ ] **ADR 0002 (architecture) accepted** <- gates model code
- [ ] Eval harness computes FID * [ ] aitviewer studio * [ ] Streaming decode loop
