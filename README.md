# Streaming Text-to-Motion

A text-to-motion generator that makes human motion one chunk at a time, with a memory that does
not grow. Master's thesis: tokenizer, generator, training pipeline, evaluation harness and a live
viewer.

## Problem

The most accurate text-to-motion models read the full sequence in the two directions. They change
a frame after they see the frames that follow it. Thus they cannot generate motion while it plays.

A streaming generator has two conditions:

1. **Causal.** The frame at time $t$ uses only the text and the frames before $t$.
2. **Bounded state.** The memory between two steps does not grow with the length of the motion.

A causal transformer satisfies condition 1 only. It keeps the keys and values of each past
position, so its memory grows with each step. A state-space model (Mamba) keeps a state of fixed
size.

This thesis asks one question: **does the fixed-size state cost quality?** It compares a Mamba
generator with a transformer that has the same parameter count, tokenizer, data, seed and budget.
The two models are different only in the sequence mixer.

## Demo

<!-- DEMO: put the aitviewer screen capture here (type a caption, the avatar moves live). -->

The studio shows the motion while the model generates it. Type a caption in the prompt bar.

```bash
text2motion studio --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone mamba \
                   --ckpt artifacts/checkpoints/generator/generator_mamba_bs8_e60.pt \
                   --tokenizer_ckpt artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt
```

The studio starts the generation service in a second process. The viewer paces the stream, so the
service generates 20 frames for each second of playback. The display interpolates to 60 frames
per second.

## Features

### A Grouped-FSQ motion tokenizer

Finite scalar quantization with a grouped-residual structure. It has no learned codebook. The
baseline is a residual VQ with the same encoder, decoder, data, seed and budget.

| Codes per step | RVQ-512 | RVQ-1024 | FSQ-512 | FSQ-1024 |
|---:|---:|---:|---:|---:|
| 4 | 0.0626 | 0.0558 | 0.0612 | 0.0527 |
| 6 | 0.0342 | 0.0310 | 0.0305 | 0.0283 |
| 8 | 0.0277 | 0.0221 | 0.0196 | **0.0170** |

Reconstruction FID on 2189 test clips, lower is better. FSQ is better in all six matched cells.

### A token-autoregressive Mamba generator with a matched transformer twin

| Model | Parameters | FID ↓ [95% interval] | R@1 ↑ | R@3 ↑ | MM-Dist ↓ | Stream state |
|---|---:|---|---:|---:|---:|---|
| Transformer | 33.91M | 0.132 [0.127, 0.230] | 0.395 | 0.669 | 3.796 | grows with length |
| Mamba | 34.10M | **0.121** [0.114, 0.212] | **0.418** | **0.693** | **3.677** | fixed |

Full test split, 2189 clips, 20 repetitions, one sampling setting for the two models, 60 epochs,
one seed. The Mamba is better in each point estimate. The R@1 lead is approximately three standard
deviations. The FID intervals overlap, so the two models are equal on FID.

For context only: T2M-GPT reports FID 0.116 and MoMask reports FID 0.045. MoMask is bidirectional
and cannot stream. This project does not claim state-of-the-art quality.

### Bounded-memory streaming

| Stream length (tokens) | Transformer cache | Mamba state | Ratio |
|---:|---:|---:|---:|
| 1024 | 76.7 MB | 2.68 MB | 29× |
| 4096 | 303.2 MB | 2.68 MB | 113× |
| 8192 | 605.2 MB | **2.68 MB** | **225×** |

The memory result follows from the two recurrences. It does not depend on hardware. For each step
the transformer is faster below approximately 6500 tokens; the Mamba runs without its compiled
kernel.

The tokenizer decoder reads 4 latent tokens on each side of a frame. A lookahead of 4 tokens
(0.8 s) makes the streamed motion identical to a decode of the full sequence.

### Engineering

- **One GPU with 4 GB.** All models were trained locally, with guarded runs and full resume state.
- **A frozen evaluator.** The evaluator of Guo et al. is used without change, so the numbers are
  comparable with published work. Model selection uses validation only. The test split is scored
  one time for each selected checkpoint.
- **Traceable numbers.** Each result has a run folder with the config, the code revision, the
  seed and the library versions.
- **One entry point.** Each stage is a subcommand of `text2motion`. Each flag has a help text.

## Setup

### Install

```bash
uv sync --inexact                    # core, including the aitviewer studio
uv sync --inexact --extra prep       # pyarrow, for the official-release converter
uv sync --inexact --extra tracking   # MLflow, only when mlflow_tracking is on
uv sync --inexact --extra dev        # ruff, pytest, pyright
pytest                               # shape and parity contracts
```

Use `uv sync --inexact` (as `make install` does). A plain `uv sync` removes every package that
`uv.lock` does not name, and the optional Mamba CUDA kernel (`mamba-ssm`, used when a config sets
`use_kernel: true`) is not in the lock: it is a 46-minute MSVC build of the patched source in
`third_party/mamba_ssm-2.3.2.post1-msvc` (see `doc/notes/implementation-difficulties.md`).

```bash
make kernel-restore   # copy back the saved working build from dist/mamba_kernel (torch 2.12+cu132)
make kernel-build     # rebuild from source; the MSVC/NVCC flags are a reconstruction, UNTESTED
```

`make help` lists a short command for each step of the project (GNU make is necessary).

### Run

`text2motion <command> --help` shows each flag and its effect.

```bash
text2motion prepare          --config configs/default.yaml --stage all
text2motion train-tokenizer  --config configs/tokenizer/fsq_g8_v1024.yaml --tokenizer fsq
text2motion tokenize         --config configs/generator/gen_pilot_fsq8x1024.yaml \
                             --token_pack data/amass_tokens_fsq8x1024.npz
text2motion sanity-overfit   --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone mamba \
                             --tokenizer_ckpt artifacts/checkpoints/tokenizer/fsq_g8_v1024.pt \
                             --overfit_gate artifacts/outputs/gates/mamba.json
text2motion pretrain         --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone mamba
text2motion train-generator  --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone mamba \
                             --overfit_gate artifacts/outputs/gates/mamba.json
text2motion evaluate         --config configs/generator/gen_pilot_fsq8x1024.yaml --backbone both \
                             --split test --cfg_scale 6.0
text2motion benchmark        --config configs/generator/final100m_fsq8x1024.yaml
```

### Layout

Dependencies point inward: `app -> studio / streaming / evaluation -> generation -> tokenization -> motion`.

```
src/text2motion/
  motion/         263 layout, kinematics, data set, AMASS preparation
  tokenization/   FSQ and RVQ tokenizer, trainer, corpus encoding
  generation/     transformer and Mamba generators, text encoder, trainers, losses
  evaluation/     FID, R-precision, MM-Dist, Diversity, MultiModality, streaming benchmark
  streaming/      incremental decode loop, service, wire protocol
  studio/         SMPL-X fit and aitviewer studio
  app/            config, checkpoint schemas, composition root, command line
configs/          YAML settings
scripts/          training drivers, evaluation, benchmarks, figures, rendering
tests/            one folder for each source stage
doc/              thesis (LaTeX) and notes
```

### Containers

The repository has a multi-stage image and Compose profiles for tests, inference, evaluation and
training. See [DOCKER.md](DOCKER.md).

### Licences

This project uses the [PolyForm Noncommercial License 1.0.0](LICENSE). You can use it for study,
research and teaching. You cannot use it to make money. [NOTICE](NOTICE) gives the copyright line
and the MIT notices of the ported code.

The data sets and the SMPL-X body model permit non-commercial research only, and this repository
does not contain them. [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) lists each third-party
licence with its source.

### Limitations

- One seed for each model.
- 34M parameters. No matched pair exists at a larger scale.
- The Mamba runs without its compiled kernel, so its latency is an upper limit.
- The models were trained on clips of a maximum of 196 frames (approximately 10 seconds). After
  that length the motion loses its relation to the text. The memory result is measured at long
  lengths; the motion quality at long lengths is not.
- The body has 22 joints. Hands and face are not generated.
- Generated motion has approximately 50% more foot sliding than real motion.
