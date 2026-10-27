# tokenization

This package turns motion features (HumanML3D-263) into discrete token indices and back. It
depends only on the `motion` package (see `tests/app/test_import_boundaries.py`).

Internal chain, from the base up: `contracts` → `nn_quantizers` → `nn_model` → `corpus`,
`metrics`, `trainer`. `ema` has no local imports.

## Interface used by the rest of `src`

| Name | File | What it is | Used by |
|---|---|---|---|
| `TokenizerConfig` | `contracts` | Config of the finite scalar tokenizer (YAML `tokenizer:`) | `app/config` |
| `ResidualVectorQuantizationConfig` | `contracts` | Config of the residual vector baseline (YAML `rvq_baseline:`) | `app/config` |
| `TokenizerKind` | `contracts` | `FINITE_SCALAR` (`fsq`) or `RESIDUAL_VECTOR` (`rvq`) | `app/cli`, `app/config`, `app/commands/generator`, `app/commands/tokenizer` |
| `TrainTokenizerRequest` | `contracts` | Arguments of the `train-tokenizer` command | `app/cli`, `app/commands/tokenizer` |
| `TokenizeRequest` | `contracts` | Arguments of the `tokenize` command | `app/cli`, `app/commands/data` |
| `build_tokenizer_network` | `nn_model` | Builds the network that the config selects | `app/bootstrap` |
| `TokenizerNetwork` | `nn_model` | Base network: `encode`, `decode`, `forward`, `quantizer_stack`, `codebook_size` | `generation/losses`, `generation/trainer`, `streaming/decoder` |
| `FrozenMotionTokenizer` | `nn_model` | Evaluation wrapper: `MotionClip` ↔ `MotionTokens` | `app/bootstrap`, `generation/pipeline` |
| `TokenizerTrainer` | `trainer` | One optimizer step, EMA weights, code statistics | `app/commands/tokenizer` |
| `train_tokenizer` | `trainer` | The training loop with resume and best checkpoint | `app/commands/tokenizer` |
| `evaluate_tokenizer_reconstruction` | `metrics` | Reconstruction FID and MPJPE on a split | `app/commands/tokenizer` |
| `build_token_pack` | `corpus` | Encodes the corpus into a token pack file | `app/commands/data` |
| `read_pack_identity` | `corpus` | Reads the identity of a token pack | `app/commands/generator`, `generation/pretrain` |
| `pack_clip_keys` | `corpus` | Lists the clip keys in a token pack | `generation/pretrain` |
| `ExponentialMovingAverage` | `ema` | EMA copy of model weights | `generation/trainer` |

All users import from the file, for example `text2motion.tokenization.nn_model`. No code imports
from `text2motion.tokenization` itself.

## Used only by tests and scripts

- `contracts`: `DROPPED_CODE`, `FiniteScalarStackLayout`
- `corpus`: `PACK_FORMAT`, `PACK_IDENTITY_KEY`, `PACK_VERSION`, `window_starts`
- `nn_model`: `FiniteScalarQuantizationTokenizer`, `ResidualVectorQuantizationTokenizer`,
  `downsample_stage_count`, `feature_velocity_reconstruction_loss`
- `nn_quantizers`: `FiniteScalarQuantizer`, `QuantizerStack`, `VectorQuantizer`, `typed_modules`

Everything else in the package is internal.
