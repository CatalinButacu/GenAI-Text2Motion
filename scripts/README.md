# scripts/ — what each series is for

One folder per series. If you are looking for "the thing that produced number X", start here.

Everything that needs a GPU should be launched behind `training/local_guard.ps1` (see
`CLAUDE.md`) — a frozen tokenizer run once burned 10.6 h unnoticed.

## `training/` — runs that produce checkpoints

| file | what it is |
|---|---|
| `local_guard.ps1` | the watchdog. Stall-kills on a stale `metrics.jsonl` heartbeat plus a max-hours budget, and writes the reason to `artifacts/outputs/GUARD_KILL.txt`. Wrap every long local job in it. |
| `run_3day_local.ps1` | the local generator chain: overfit gate -> pretrain -> finetune, for both twins. Written for the 4 GB laptop, so its `--epochs`/`--batch_size` are laptop compromises, not the intended recipe. |

### `training/tokenizer_sweep/` — Contribution A, and it is FINISHED

The grid that settled the tokenizer question. **Do not re-run these.** The winner is frozen at
`artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt` (recon-FID **0.0170**), every generator is trained on its
tokens, and retraining the tokenizer would invalidate every generator checkpoint we own.

| file | what it is |
|---|---|
| `tokenizer_sweep.ps1` | drives the whole grid |
| `fsq_sweep.ps1` / `rvq_sweep.ps1` | one arm each: grouped-FSQ vs strong-RVQ baseline |
| `matrix_complete.ps1` / `matrix_remaining.ps1` | resume helpers for a partially-finished grid |
| `run_fsq_g*.ps1` / `run_rvq_l*.ps1` | individual grid cells, one config each |

Results live in `artifacts/outputs/runs/tokenizer/*/metrics.jsonl` (18 runs, full test split, 2189 clips).
Headline: FSQ beats RVQ at every matched depth, and by 26% at matched depth *and* matched vocab.

## `evaluation/` — thesis metrics

| file | what it is |
|---|---|
| `twin_eval.py` | the citable 20-rep protocol. Thin shim over `text2motion evaluate`. **Touches `test`** — run once, for the final table. Was `_twin_eval.py`. |
| `twin_test_eval.ps1` | driver that runs the above for both twins |
| `l2_ground_truth.py` | GT-reproduction driver that validated the frozen Guo evaluator against published FID/Diversity. Was `_l2.py`. |
| `eval_generalization.py` | held-out generalization analysis |
| `pretrain_generalization.py` | does AMASS pretraining actually transfer? |
| `streaming_decode_fidelity.py` | proves streaming decode == batch decode |

## `benchmarks/` — speed, memory, and size probes

These answer "is it fast enough", never "is it good". No FID here.

| file | what it is |
|---|---|
| `perf_probe.py` | the main harness: `--probe {determinism,step,sync,rollout,demo,kernel}`. `kernel` is what the cloud canary runs. Always compare interleaved in ONE process — thermal drift on the laptop has produced false readings twice. |
| `mem_probe.py` | memory-footprint probe |
| `scaling_budget.py` | compute-budget scaling analysis |
| `param_count_100m.py` | twin param-count matcher — the twins must match on parameters, not just hyperparameters. Was `_sanity_100m.py`. |

## `figures/` — paper figures

`architecture_diagram.py`, `make_figures.py`, `plot_streaming_bench.py`.

## `render/` — SMPL-X visualization

`render_clip.py`, `render_mesh_demo.py`.

## `demos/` — interactive

`stream_viewer.py`.

## Cloud entrypoints are NOT here

They live in `infra/`: `infra/kernel_canary.sh` (the fused-kernel canary),
`infra/make_bundle.sh` (builds the upload payload), `infra/azure/run_twins.sh` (the six training
runs), plus the Terraform in `infra/azure/`.

## `qa/` — test the studio the way a person uses it

| file | what it is |
|---|---|
| `studio_human_test.py` | drives the **real** studio viewer offscreen (headless window, real PyQt6 key codes, 60 Hz paced like vsync) through human scenarios: type a prompt and press Enter, double Enter, run forever then Stop then a new prompt, keyboard shortcuts (H skin, S shadows, R reload), live typing, switching models. Checks state invariants every frame (no duplicate nodes, playhead in range, no error status, finite joints) and reads rendered pixels to verify the mesh / skeleton are actually drawn. Starts its own service on port 8797. Screenshots + `report.json` in `artifacts/outputs/qa/studio/`. |
| `studio_controls_test.py` | clicks **every control** in the studio for real: records each ImGui widget's on-screen rect as the panels draw it, then drives the viewer's own mouse/keyboard handlers (move, press, drag, release, scroll, type, Enter) and asserts each control's effect -- all Environment, Sampling, Avatar, Model, Playback and Prompt-bar widgets (57 checks). Every button click must actually fire, or the test reports the rect and clip it missed. Exit code 1 on any failure; `report.json` in `artifacts/outputs/qa/studio_controls/`. |
| `motion_naturalness_probe.py` | per-stage motion statistics (acceleration, jerk, frames over the physical limit, root teleports, foot skate) for ground truth, tokenizer round-trip, generated skeleton and the studio mesh -- localizes where unnatural motion enters. |
| `motion_glitch_probe.py` | generates long motions through the exact service path (streaming decoder + chunked skeleton recovery), plain vs re-anchored, and counts over-limit joint accelerations, root teleports, foot skating, NaNs, and streamed-vs-whole recovery error, with spike positions checked against decoder seams and re-anchor rebuilds. |

Both studio QA scripts take `--visible` (drive the real PyQt6 window on screen, with the injected mouse cursor drawn) and `--pace N` (slow typing and add pauses so a person can follow). Selenium does not apply: the studio is a native OpenGL/ImGui window with no DOM or accessibility tree; these scripts are the equivalent -- find a control by label, act on it, assert the effect.
