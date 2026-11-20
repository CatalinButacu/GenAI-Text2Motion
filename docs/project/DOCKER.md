# Docker

The container setup has three services:

| Service | Purpose |
|---|---|
| `test` | Run the test suite against the mounted source tree |
| `service` | Run the long-lived GPU motion service |
| `job` | Run any one-shot command such as evaluation, benchmark, or training |

The image contains code and dependencies. Datasets, checkpoints, outputs, logs, and caches are mounted
from the host. The runtime is non-root and dependencies are installed from `uv.lock`.

## Setup

Copy `.env.example` to `.env` and adjust paths, model selection, and the memory limit. The default
limit is 8 GB with no additional swap. Licensed data and checkpoints are never copied into images.

## Test

```bash
docker compose --profile test build test
docker compose --profile test run --rm test
```

## Service

```bash
docker compose --profile service up --build -d service
docker compose ps
docker compose logs -f service
docker compose --profile service down
```

The health check uses the native socket protocol. The service exposes port `8765` by default.

## Jobs

The `job` service accepts any `text2motion` command:

```bash
docker compose --profile job run --rm job evaluate \
  --config configs/default.yaml \
  --backbone transformer \
  --ckpt /app/checkpoints/generator/generator_transformer_bs8_enc_last.pt \
  --tokenizer_ckpt /app/checkpoints/tokenizer/fsq_g8_v1024.pt
```

Use the same service for `benchmark`, `sanity-overfit`, `pretrain`, and `train-generator`. Compose does
not automatically start or restart one-shot jobs.

## Models

`configs/demo_models.yaml` is a deployment registry, not a second model implementation. Each entry has
a stable `name` and date-based string `version`. Shared values live once in `defaults`. Mamba 34M and
Mamba 100M use the same code; they are different trained capacities and should be registered as
different releases. Exact parameter counts belong in evaluation reports, not deployment selection.
Only atomic `*_last.pt` bundles containing both generator and text-encoder state may be served.

The current 100M exports are not registered because they are bare generator exports. Add a new version
only after an atomic bundle exists.
