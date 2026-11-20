# A26 (proposed): project structure around runs

Status: proposed, waiting for owner approval. No files moved yet.

## Principle
Every computed thing belongs to a run. A run is one command with one config, one seed and one data version. Its inputs, log, metrics and outputs live in one folder, so nothing needs to be searched for. Shared folders only hold inputs (data) and a single index of runs.

## 1. Run folders
`artifacts/runs/<kind>/<run_id>/`, where `run_id = YYYYMMDD-HHMM-<short-name>`.

| kind | what it is | contents |
|---|---|---|
| `data` | a preparation or conversion (HumanML3D 60 fps, a source converter) | `manifest.json`, counts per step, `train.log` |
| `tokenizer` | RVQ / FSQ training (and its reconstruction test) | `config.yaml`, `train.log`, `metrics.csv`, `checkpoints/`, `recon/` |
| `overfit` | the mandatory single-batch gate before a real run | same as tokenizer |
| `generator` | motion generator training (Mamba or transformer twin) | same as tokenizer, plus `frozen_from.json` pointing to the tokenizer run it depends on |
| `part` | hand or face blocks, trained on a frozen body | same as generator, plus `frozen_from.json` for the body generator |
| `eval` | FID, R-precision, diversity on a checkpoint | `scores.json`, `per_clip.csv`, `config.yaml` |
| `bench` | streaming latency and memory (Mamba vs transformer) | `timings.csv`, `env.json` (GPU, driver, torch version) |
| `render` | mesh renders of generated motion | `frames/`, `videos/`, `motion_input.npz` |

Each run folder contains a `run.json`: command, git commit, config hash, data version, seed, GPU and driver, start and end time, status, and the parent run ids. A run is complete only when `run.json` has `status: ok`.

## 2. Run index
`artifacts/runs/INDEX.csv` is appended by the tool at the start and end of every run: run_id, kind, status, key metric, parent. It is the only file that spans kinds.

## 3. Frozen blocks
A frozen block is a run plus a pointer. Example: `artifacts/runs/generator/<id>/frozen_from.json` names the tokenizer run and the SHA-256 of its checkpoint. A part run refuses to start unless its parent exists with `status: ok`. Changing a frozen block creates a new run, so old results keep pointing to the version they used.

## 4. Data layout
- `data/raw/<source>/`: downloads as delivered, read-only after extraction. The manifest records checksums and licence.
- `data/processed/<data_version>/`: the 263 features at 60 fps, splits, Mean/Std, the clip index. Each data version has a hash. Runs record the version.
- `data/models/`: SMPL-X v1.1, the Guo evaluator, GloVe, the MANO and SMPL-X correspondences.
- `data/renders/<source>/`: preview videos from the source, for watching the motion.
- Removed: `_restore_backup`, `_hml3d_convert`, `grab_raw` (empty), and the duplicate copies.

## 5. Logs
- Training, tokenizer, eval and bench logs live inside their run folder, and nowhere else.
- The LaTeX build log stays in `doc/paper/build/`, because it belongs to the paper.
- The paper-watch and replay logs move to `artifacts/runs/notes/` as ordinary runs, or are deleted if no longer used.
- No Azure or AWS logs (those services are gone).

## 6. Legacy results
Current `artifacts/outputs` (0.5 GB, old figures and benchmarks) moves to `artifacts/legacy/2026-11/` with a README saying which run produced each file, if known. Nothing is deleted, because these numbers are cited in the paper draft.

## 7. Code (later, after the data and runs work)
The packages already group code by job. The only move is the top-level clutter: `.archify`, `.vscode`, `imgui.ini`, `typings`, `third_party`, `licenses`, `infra`, `dist`, `graphify-out`. Each move is checked by the test suite, because `tests/test_package_layers.py` and `tests/app/test_import_boundaries.py` enforce the layout, configs resolve paths from the repo root, and `.claude/settings.json` hook paths are absolute.

## 8. Steps and checks
1. Write `run.py` (a small wrapper): creates the run folder, writes `run.json`, appends `INDEX.csv`. Test: a dry run creates the right files.
2. Move `artifacts/outputs` to `legacy`, and point the config's `outputs_dir` and `logs_dir` at `artifacts/runs`. Test: `make check` passes.
3. Reorganise `data/` into raw, processed, models and renders. Test: loaders find every file, checked by counts and checksums.
4. Code moves at the top level, one at a time, each followed by `make check`.

## Decisions for the owner
1. Approve the run folder layout and the kinds.
2. Approve the legacy move (no deletion).
3. Approve deleting the duplicate `_restore_backup` and `_hml3d_convert` (I checked that the converted features are already in `HumanML3D_official`).
