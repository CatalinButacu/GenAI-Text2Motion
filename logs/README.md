# logs/ — every run leaves evidence here

One rule: **anything that records what a run did goes under `logs/`. Anything a run produced
for a human to look at goes under `outputs/`.**

| path | what | written by |
|---|---|---|
| `logs/train/<stamp>_<name>/` | `manifest.json` (config + git commit + seed + versions) and `metrics.jsonl` (per epoch / eval) | `app/run_log.py` via `start_run()` |
| `logs/cli/` | console logs of long-lived processes: motion service, studio | `streaming/protocol.connect_or_start_motion_server`, shell redirects |
| `logs/perf/inference.jsonl` | one row per inference: prompt, backbone, checkpoint, timing breakdown, peak GPU, host RSS | `streaming/service.record_inference` |
| `logs/*/\_archive/` | console logs from earlier sessions, kept as evidence, not written to any more | moved by hand |

## Reading the perf log

```powershell
Get-Content logs/perf/inference.jsonl | ConvertFrom-Json |
  Format-Table utc, backbone, frames, realtime_factor, ms_per_step, peak_gpu_gb
```

Discard the first row of a fresh service — CLIP and the CUDA context warm up on the first
generation, so it reads slower than steady state (measured: 45.9 ms/step cold vs 24.5 warm).

`peak_gpu_gb` covers the **service process only**, so it isolates the generator and excludes the
studio's renderer and SMPL-X fitting. That is what makes it comparable with the offline probes in
`scripts/benchmarks/`. Do not quote it as whole-application memory.

## outputs/ — what a run produced

| path | what |
|---|---|
| `outputs/runs/` | **historical** run dirs from before the `logs/` split. ADRs, STATUS and the dissertation cite these paths, so they stay put. New runs go to `logs/train/`. |
| `outputs/demo/`, `outputs/demo_gallery/` | rendered clips, stills, joint dumps |
| `outputs/figures/` | paper figures and the JSON behind them |
| `outputs/*.txt` | sweep sentinels and `GUARD_KILL.txt`, read by `scripts/training/*.ps1` |

## Where a new file goes

- a number you might quote later → `logs/`, never a bare `print`
- something you want to look at → `outputs/`
- a model → `checkpoints/<stage>/`, matching `generator/` and `tokenizer/`
- a throwaway → the scratch dir, not the repo
