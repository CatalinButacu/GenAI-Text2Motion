# Streaming Text-to-Motion

Generate whole-body human motion from a text caption **one chunk at a time**, committing to each frame
before the next one exists — with a recurrent state that stays constant no matter how long you keep
generating.

Master's thesis, built from scratch: motion tokenizer, generator, training pipeline, evaluation
harness, and a live viewer fed by the streaming decoder.

<!-- DEMO: drop the aitviewer screen capture here (type a caption -> avatar moves live). -->

## Why prediction

Most text-to-motion systems generate offline. A diffusion model sees the entire horizon and refines it
until it looks right; that is a legitimate way to get a beautiful clip, and it is also a way of never
answering the harder question.

This project takes the opposite constraint: **commit to frame $t$ before you are allowed to see frame
$t+1$.** No rewinding, no global refinement pass. Once emitted, a frame is spent.

That constraint is what makes prediction interesting to me. It forces the model to carry everything it
knows about the past in a *bounded* summary rather than an ever-growing transcript, and it forces you
to be honest about what you lose. It is also the constraint that a robot controller, a telepresence
codec, and a live avatar all actually live under — the offline setting is the special case, not the
general one.

## What is new here

Text-to-motion is a crowded field. Two cells in it are not occupied, and this thesis sits in both.

**1. A grouped/residual FSQ motion tokenizer.** Finite Scalar Quantization is proven on motion
(ScaMo); residual quantization is proven on motion (MoMask); the *combination* is not published.
It is benchmarked against a strong RVQ+EMA baseline and MoMask's released RVQ.
Reconstruction FID **0.017** — the ceiling the generator is measured against.

**2. The first token-autoregressive S6/Mamba motion generator**, evaluated against a causal
transformer twin that differs *only* in the sequence mixer — identical tokenizer, data, split, seed,
and compute budget, with layer counts set to match total parameters.

The claim is deliberately on the **efficiency axis**, not the quality axis:

> a bounded recurrent state matches a growing KV-cache in generation quality, at matched budget.

## Results

**The mechanism — streaming state vs. horizon** (both 100M twins, single stream, CUDA):

| horizon $L$ | Transformer KV-cache | Mamba recurrent state | ratio |
|---:|---:|---:|---:|
| 1024 | 76.7 MB | 2.68 MB | 29× |
| 4096 | 303.2 MB | 2.68 MB | 113× |
| 8192 | 605.2 MB | **2.68 MB** | **225×** |

The transformer's cache grows without bound; the SSM state does not change size. This is guaranteed by
the recurrence — it holds independent of kernel, precision, or hardware. Per-step latency crosses over
near $L \approx 6500$ *even without* the `mamba-ssm` CUDA kernel, which is unavailable on the Windows
test machine and costs the SSM a roughly constant $2\times$.

**The controlled twin** (full test split, 2189 clips, 20-rep protocol, one sampling setting applied
identically):

| Model | params | FID ↓ | R@1 ↑ | MM-Dist ↓ | stream state |
|---|---|---|---|---|---|
| Tokenizer ceiling (recon) | — | 0.017 | — | — | — |
| Transformer-34M (ours) | 34M | **3.30** | 0.246 | 5.26 | $O(L)$ KV |
| Mamba-34M (ours) | 34M | 4.06 | **0.255** | **5.00** | **$O(1)$ state** |
| T2M-GPT (cited) | ~230M | 0.116 | 0.417 | — | $O(L)$ |
| MoMask (cited, non-streaming) | — | 0.045 | 0.521 | — | — |

Neither mixer dominates. The SSM tracks the caption slightly better (ahead on all three R-precision
ranks and on MM-Dist); the transformer better matches the ground-truth motion distribution (FID 3.30
vs 4.06). That is the result the thesis argues for — parity at matched budget, alongside an $O(1)$
streaming state.

**These are 34M pilots, not SOTA.** The cited baselines run at 100M+ scale and MoMask runs
bidirectionally, so they are a reference ceiling, not a target this pilot claims to beat. A separate
100M cell reaches val FID 1.63, but it moved capacity, conditioning, and sampling simultaneously, so
it is reported as an un-decomposed configuration change rather than a scale claim.

**AMASS generator-pretraining** (34M pilot ablation): pretraining a text-free motion prior on
unlabeled AMASS before fine-tuning on HumanML3D cuts pilot FID **6.58 → 2.12** with *fewer* fine-tune
epochs.

Every number above traces to a run directory under `outputs/runs/` with a manifest recording config,
git commit, seed, and library versions.

## What this project demonstrates

- **Training under a hard compute ceiling** — everything here was trained on a 4 GB GPU, with
  gradient accumulation, watchdog-guarded runs, and full resume state so a stalled job costs one epoch
  rather than the run.
- **Discrete-latent modeling** — quantizer design, codebook utilization, reconstruction-vs-downstream
  trade-offs. The same machinery behind any tokenized generative system.
- **Evaluation discipline** — a frozen third-party evaluator (Guo et al.) reused unmodified so FID
  stays comparable to the published field; model selection on validation only; the test split scored
  once per final table.
- **Ablation hygiene** — confounded comparisons are labeled as confounded and decomposed with
  parameter-identical control cells, rather than reported as the flattering interpretation.
- **Streaming inference** — a bounded-memory decode loop with a *measured* decoder receptive field
  ($\pm 4$ latent tokens $= \pm 0.8$ s), a `stream == batch` numerical parity test proving the
  evaluated model is the streamed model, and a live viewer consuming it.

## Layout

The package is organised by pipeline stage, and dependencies point strictly inward
(`app -> studio/streaming/evaluation -> generation -> tokenization -> motion`):

```
src/text2motion/
  motion/         canonical motion domain: 263 layout, kinematics, dataset, AMASS preparation
  tokenization/   FSQ/RVQ tokenizer, its trainer, corpus encoding, reconstruction eval
  generation/     transformer + Mamba generators, CLIP text encoder, trainers, loss recipe
  evaluation/     FID, R-precision, MM-Dist, Diversity, MultiModality, streaming bench
  streaming/      bounded-memory incremental decode loop, service, wire protocol
  studio/         SMPL-X fitting + aitviewer studio
  app/            config, checkpoint schemas, composition root, unified CLI
configs/          YAML hyperparameters (single source of truth)
scripts/          sweep launchers, watchdog, one-off analyses, figures
tests/            shape / round-trip / parity / sanity contracts
paper/            dissertation draft
```

Each stage exchanges named domain objects (`MotionClip`, `MotionTokens`, `MotionBatch`,
`GeneratedMotion`, `MotionChunk`) rather than bare tuples, and exposes one facade —
`MotionRepository`, `MotionTokenizer`, `MotionGenerator`, `MotionEvaluator`, `StreamingService`.
Object construction happens in exactly one place, `app.container.ApplicationFactory`.

## Setup

```bash
uv sync                  # core
uv sync --extra viewer   # aitviewer studio
uv sync --extra dev      # ruff + pytest
pytest                   # shape and parity contracts
```

## Running

Every stage is a subcommand of one entry point (`python -m text2motion.app.cli <command>`, or
`text2motion <command>` once installed):

```bash
text2motion prepare          --config configs/default.yaml --stage all
text2motion train-tokenizer  --config configs/tokenizer/fsq_g8_v1024.yaml --tokenizer fsq
text2motion tokenize         --config configs/default.yaml --out data/amass_tokens.npz
text2motion sanity-overfit   --config configs/default.yaml --backbone mamba \
                             --tokenizer_ckpt checkpoints/tokenizer/fsq_g8_v1024.pt --out gate.json
text2motion pretrain         --config configs/generator/final100m.yaml --backbone mamba
text2motion train-generator  --config configs/generator/final100m.yaml --backbone mamba \
                             --overfit_gate gate.json
text2motion evaluate         --config configs/default.yaml --backbone both --split test
text2motion benchmark        --config configs/generator/final100m.yaml
text2motion serve            --config configs/generator/final100m_fsq8x1024.yaml
text2motion studio           --config configs/generator/final100m_fsq8x1024.yaml
```

Datasets and SMPL-X body models are license-gated and are not distributed here; paths are configured
in `configs/`.

## Limitations

- Pilot scale (34M) — absolute FID is well above published 100M-class systems.
- Most ablation cells are single-seed; no variance bars outside the main twin table.
- Mamba runs in eager mode on the test machine, so reported latency is a lower bound on its advantage.
- Whole-body SMPL-X (hands, face) is staged behind the body-only 263 track used for citable FID.
