# Chapter 5 — Training Methodology and Data Pipeline

> **Target: 10-12 pages.** Three-stage training (RVQ, MotionSSM, planner LM) plus the data pipeline that feeds them and the cloud infrastructure that runs the headline experiment. Every recipe here corresponds to a script in `scripts/training/` and a config knob in `configs/motion_ssm.yaml`.

---

## 5.1 Datasets

We train on three distinct corpora, each serving a different stage:

| Stage | Dataset | Source | Use |
|---|---|---|---|
| RVQ tokenizer | AMASS (SMPL-X) | [Mahmood et al., 2019] | Continuous-motion reconstruction |
| MotionSSM | HumanML3D (annotations) + AMASS (motion backing) | [Guo et al., 2022 CVPR] | Text-conditioned token prediction |
| Planner LM | Synthesised (instruction, action_list) pairs | This work, `scripts/data/synthesize_planner_data.py` | Action-decomposition fine-tune |

**Motion feature representation.** Every motion in every dataset is packed into a 168-dimensional per-frame SMPL-X feature vector via `src/data/amass/smplx_pack.py`. The fields, by channel:

| Channels | Content |
|---:|---|
| [0:3]   | Root orientation (axis-angle) |
| [3:6]   | Root translation (world-frame xyz) |
| [6:69]  | Body pose (21 joints × 3 axis-angle) |
| [69:159]| Hand pose (15 left + 15 right × 3 axis-angle) |
| [159:162]| Jaw pose (axis-angle) |
| [162:168]| Eye pose (2 eyes × 3 axis-angle) |

Z-score normalisation per channel using statistics computed offline from the AMASS training split, with a separate translation-channel statistics file for the HumanML3D backing path (the two distributions differ enough that shared normalisation hurts reconstruction).

**HumanML3D split.** We use the standard Guo splits:

| Split | Clips | Hours |
|---|---:|---:|
| train | 11650 | ~24 |
| val   | 1400  | ~3  |
| test  | 1400  | ~3  |

Each clip carries ~3 human-written caption variants; we sample one uniformly per epoch (the `text_per_epoch_resample` flag in the dataset loader).

---

## 5.2 Data Pipeline

```
AMASS .npz files          HumanML3D texts/      synthesized planner.jsonl
       │                          │                      │
       ▼                          ▼                      ▼
src/data/amass/        src/data/humanml3d_loader   scripts/data/
  smplx_pack.py             .py                     synthesize_planner_data
       │                          │                      │
       └──────┬───────────────────┘                      │
              ▼                                          │
       UnifiedDataset (data/.cache/unified_buf_*.joblib)│
              │                                          │
              ▼                                          ▼
       PyTorch DataLoader                       PyTorch DataLoader
       (RVQ + MotionSSM training)               (planner LM training)
```

**Caching.** AMASS contains ~14k .npz files totaling ~150 GB. Loading them per-batch is infeasible. We pre-compute a single ~60 MB joblib that contains the packed 168-d motion tensors + caption indices + per-source statistics, keyed by a content hash so a cache regeneration is detectable. `scripts/data/prebuild_unified_cache.py` does this offline; the cache file uploads to S3 once and the cloud training jobs `aws s3 cp` it onto a fresh instance in ~10 seconds.

**Augmentation.** Three augmentations applied during training:

- **Mirror flip** (probability 0.5): flips left/right body joints. Effectively doubles the dataset.
- **Random temporal crop**: clip length is sampled in [40, max_motion_length=200] frames. Forces the SSM to learn length-conditional structure rather than memorising one length.
- **Caption resampling**: pick a random caption from the per-clip list. Implicit data augmentation on the text side.

Augmentation is implemented in `src/data/augment.py` as a small composable pipeline so we can ablate any single transformation by turning it off in the config.

**Frame masking.** Sequences shorter than `max_motion_length` are right-padded with zeros; a `motion_mask: (B, T)` tensor records which frames are real. The RVQ tokenizer is trained on the entire padded sequence (the network never sees the mask, but the loss is masked); the MotionSSM is trained on the latent mask `build_latent_mask(motion_mask, down_t)` so the token-prediction loss only counts valid latent positions.

---

## 5.3 Training Stage 1: RVQ Tokenizer

**Objective.** Reconstruction + commitment + auxiliary smoothness:

```
L_rvq = L_recon + λ_vel · L_vel + λ_commit · L_commit
```

| Term | Definition | Default weight |
|---|---|---:|
| `L_recon`  | MSE between original motion and `decoder(quantize(encoder(motion)))` | 1.0 |
| `L_vel`    | MSE on per-frame finite-difference velocity (foot-skating proxy) | 0.5 |
| `L_commit` | VQ-VAE commitment loss: `||sg(quantizer_input) - quantizer_output||²` | 0.25 |

The velocity term is the standard fix for the "static pose" failure mode where the decoder learns to output a frozen good pose because the per-frame MSE is locally minimised by motion damping.

**Codebook dead-code revival.** Vector-quantizers are notorious for codebook collapse — codes that get used early get reinforced; codes that don't get used die. Every 5 epochs (`--reset-dead-every 5`), codes whose usage count is below `--reset-dead-threshold 1.0` are re-initialised from the current quantizer-input distribution. `src/modules/motion/rvq_tokenizer.py:reset_dead_codes_pipeline` runs this rebalance.

**Causal decoder variant.** A separate training pass with `--causal-decoder` produces the streaming-compatible decoder (Ch 4.2). The encoder + codebooks are frozen for this pass; only the decoder retrains. ~30 min on a T4, no extra data.

**Training schedule.**

| Hyperparameter | Value |
|---|---:|
| Epochs | 200 (early-stop on val FID surrogate) |
| Batch size | 32 |
| Optimizer | AdamW, lr=2e-4, weight_decay=0.01 |
| LR schedule | OneCycleLR, 10% warmup, cosine |
| `latent_dim` | 128 |
| Codebooks | 6 × 512 entries each |
| `down_t` | 4 (200 frames → 50 latents) |

Best run "all3rd" achieves `val_recon = 0.0946` on the AMASS held-out split. The checkpoint lives in S3 at `s3://dissertation-motion-cache-91340264/checkpoints/rvq_tokenizer/best_model.pt`.

---

## 5.4 Training Stage 2: MotionSSM

**Objective.** Token cross-entropy + length regression:

```
L_ssm = sum_k CE(logits_k, target_tokens_k, latent_mask) + λ_len · MSE(length_pred, true_length)
```

where the CE is summed across K=6 codebooks and λ_len = 0.1.

**Classifier-free guidance dropout.** During training, with probability `cfg_dropout_prob = 0.1` we replace the text input with an empty string, forcing the model to learn an unconditional distribution alongside the conditional one. At inference time this enables CFG sampling: `logits_final = logits_uncond + s · (logits_cond - logits_uncond)`. We swept `s ∈ {1, 2, 4, 7}` on a pre-trained checkpoint (Ch 6.4) and selected `s = 4` for the headline run, matching the MoMask sweet spot.

**Pose-prefix curriculum.** With probability `pose_prefix_prob = 0.5` per batch, the trainer:

1. Picks a random split point P in latent steps (uniform over 1..⌈T/4⌉)
2. Encodes the prefix frames through the frozen tokenizer's RVQ-decode path → `(B, P, rvq_latent_dim)`
3. Projects to `(B, P, d_model)` via `model.seed_from_latent`
4. Calls `forward(text, motion_length=(T-P)·down_t, seed_latent=seed)`
5. Supervises CE loss against `target_tokens[:, P:]` only

This teaches the model "continue from this pose" — necessary for the inference-time action-transition path where the second action begins with the avatar mid-stride (Ch 4.7).

**Schedule.**

| Hyperparameter | Value |
|---|---:|
| Epochs | 30 (early-stop patience 30) |
| Batch size | 64 (cloud), 8-16 (local smoke) |
| Optimizer | AdamW, lr=1e-4, weight_decay=0.01 |
| LR schedule | OneCycleLR, 10% warmup |
| Gradient clip | 1.0 |
| AMP | enabled (CUDA fp16 autocast + GradScaler) |
| `pose_prefix_prob` | 0.5 |
| `cfg_dropout_prob` | 0.1 |
| Architecture flags | `--bidirectional --use-film --gradient-checkpointing` |
| Decoder head | `independent` (K-codebook parallel; `residual_k` is opt-in for ablation) |

**Multi-GPU.** When `torch.cuda.device_count() > 1` and `--single-gpu` is not set, the trainer auto-wraps the model in `nn.DataParallel`. The unwrapped `self.model` is preserved for checkpoint state-dict serialisation (the wrapped variant adds `module.` prefix that would break single-GPU resume).

---

## 5.5 Training Stage 3: Planner LM (NEW)

**Objective.** Supervised fine-tuning of GPT-2-small (124M parameters, MIT license) on synthesised (instruction, action_list) pairs.

**Prompt format.**

```
Instruction: walk forward then sit down
Actions: [{"action": "walks forward", "until": "duration(80)"}, {"action": "sits down", "until": "completed"}]<|endoftext|>
```

**Loss masking.** The cross-entropy loss is masked on the prompt portion (`Instruction: ...\nActions: `) so the model only learns to complete the JSON. Labels for prompt tokens are set to `-100`, which is the ignore index for `nn.CrossEntropyLoss`. Without masking, the model wastes capacity learning to predict the instruction surface form.

**Dataset asymmetry.** The synthesis script deliberately uses paraphrased verbs and directions in the instruction string but keeps canonical forms in the JSON action_text:

```jsonl
{"instruction": "the avatar saunters straight ahead until clear of the ladder, then gets up",
 "actions": [{"action": "strolls forward", "until": "distance(ladder) > 1.5"},
             {"action": "stands up", "until": "completed"}]}
```

The LM learns the mapping `saunters → strolls`, `straight ahead → forward`, `clear of → distance > 1.5`, `gets up → stands up`. This is what gives the system robustness to wording the user has never typed in the literal training-data form.

The paraphrase pools live in `scripts/data/synthesize_planner_data.py`:
- `VERB_PARAPHRASES`: 28 canonical verbs, 3-6 paraphrases each
- `DIRECTION_PARAPHRASES`: 6 directions, 3-4 paraphrases each
- `DIST_LT_PHRASES + DIST_GT_PHRASES + ROTATED_PHRASES`: 4-6 phrasings of each `until` variant

**Schedule.**

| Hyperparameter | Value |
|---|---:|
| Base model | `gpt2` (124M params) |
| Epochs | 3 |
| Batch size | 8 (fits in 4 GB VRAM at max_length 128) |
| Max sequence length | 128 tokens |
| Optimizer | AdamW, lr=5e-5 |
| LR schedule | OneCycleLR, 10% warmup, cosine |
| Gradient clip | 1.0 |
| Train examples | 6000 (synth) |
| Val examples | 600 (synth) |
| Wall-clock | ~60-90 min on local GPU |

**Smoke validation.** A 200-train / 50-val / 1-epoch fast pass takes ~90 seconds and drops loss from 2.55 to 0.40 (`scripts/training/train_planner_lm.py --max-train 200 --max-val 50 --epochs 1`). Inference on the smoke checkpoint already produces valid JSON with valid grammar, confirming the pipeline is sound before the full run.

**Generalization eval.** A held-out hand-authored set of 25 paraphrased instructions (`tests/fixtures/planner_held_out.jsonl`) tests the LM on wording NOT in the synthesis templates. Four metrics (Ch 8.4): valid-JSON %, valid-grammar %, canonical-hit %, mean action count. The thesis passes if canonical-hit ≥ 80% — anything lower means we memorised templates rather than generalising.

---

## 5.6 Cloud Training Infrastructure

**Strategy revision.** Originally we planned 3-4 cloud experiments (Ch 5 revision 2026-05-22): CFG sweep, CLIP-encoder retrain, AR-K-head retrain, headline run. The current strategy is **single-shot**: all architectural ablations land locally at tiny scale; the cloud is fired exactly once for the headline run with the locked config. The S3 unified-buffer cache and the existing 20260511 baseline checkpoint make this viable.

**Provisioning.** Terraform-managed `g5.xlarge` (1× A10G, 24 GB VRAM) in `eu-west-1`. The `scripts/cloud/aws/main.tf` + `startup.sh` + `terraform.tfvars` triple defines:

- the AMI (Deep Learning AMI Ubuntu 22.04 with PyTorch + CUDA pre-installed)
- the IAM role granting S3 read/write + self-terminate-by-tag
- the security group (SSH ingress from a single CIDR; all egress)
- the spot-vs-on-demand toggle (`use_spot = false` for the headline run — the $3 surcharge dwarfs the cost of a spot preempt the night before defense)

**Startup script** (`scripts/cloud/aws/startup.sh`):

1. Activate the DLAMI's pre-installed PyTorch venv
2. `uv pip install` extra deps (sentence-transformers, wandb, spacy)
3. `git clone` the repo at the configured commit hash
4. `aws s3 sync` the unified buffer cache + the RVQ tokenizer checkpoint
5. Launch `python scripts/training/train_motion_ssm.py` with the locked flags
6. On exit (success, error, signal): `aws s3 sync` checkpoints out, `terminate-instances` self

**Wallclock failsafe.** A background `( sleep 10800 && shutdown -h now ) &` fires 3 hours after boot regardless of what the foreground process is doing. This is the safety net for a hung Python process or a deadlocked NCCL collective — even if `cleanupOnExit` never fires, the kernel halts and AWS billing stops.

**Wandb.** Online mode when `WANDB_API_KEY` is set in `terraform.tfvars` (marked `sensitive = true` so it never appears in `terraform.tfstate`); the startup script forwards the env var via `sudo --preserve-env`. Otherwise offline mode + post-run sync.

**Cost.** A 30-epoch run at `batch_size=64, d_model=384` averages ~$10-15 on `g5.xlarge` over a 6-8 hour wall.

---

## 5.7 Reproducibility

Three guarantees:

1. **Seeded everything.** `src/utils/seed.py:lockSeed` sets `random`, `numpy`, `torch.manual_seed`, and `torch.cuda.manual_seed_all`. DataLoader workers get `worker_init_fn=seed_worker`. CUDA's algorithmic non-determinism is acknowledged but documented (and the streaming-equivalence test tolerates 5e-4 numerical drift).

2. **Config drift test.** `tests/test_config_drift.py` asserts `src/shared/constants.py` matches `configs/motion_ssm.yaml`. Any silent edit to one without the other fails CI. The YAML is the source of truth.

3. **Snapshot at training start.** `src/shared/run_ctx.py:snapshot_config` writes the full resolved config + the git commit hash + the dataset hash into `<checkpoint_dir>/run.json` at every training launch. Any checkpoint can be traced to the exact code state that produced it.

---

## 5.8 Compute Budget Summary

| Stage | Where | Cost | Wall |
|---|---|---:|---:|
| RVQ tokenizer (symmetric) | Cloud (already done, May 2026) | $1.20 | 2 h |
| RVQ tokenizer (causal variant) | Local | $0 | 30 min |
| MotionSSM headline (single cloud shot) | `g5.xlarge` on-demand | $10-15 | 6-8 h |
| Planner LM | Local | $0 (electricity) | 60-90 min |
| Generalization eval | Local | $0 | <1 min |
| **Total** | | **$11-16** | ~1.5 days |

The dissertation's total cloud cost is under $20 — a deliberate constraint chosen 2026-05-22 after the prior session's $6.95 abort-at-epoch-4 incident. Local iteration is cheap; cloud is reserved for the single run whose numbers go in the headline table.

---

## Figure inventory

- **F5.1** Data pipeline block diagram (the ASCII version in §5.2, redrawn).
- **F5.2** RVQ loss-curve plots (`L_recon`, `L_vel`, `L_commit` over epochs).
- **F5.3** MotionSSM loss curves (token CE per codebook, length-MSE).
- **F5.4** Planner LM train + val loss curves.
- **F5.5** Codebook utilisation histogram (proves no collapse).

## Table inventory

- **T5.1** Dataset statistics (the table in §5.1).
- **T5.2** RVQ hyperparameters.
- **T5.3** MotionSSM hyperparameters.
- **T5.4** Planner LM hyperparameters.
- **T5.5** Compute budget summary (the table in §5.8).

---

## Cross-references

- §5.3's causal decoder produces the streaming-compatible tokenizer used in **Ch 4.2** and **Ch 7**.
- §5.4's pose-prefix curriculum lands the architecture introduced in **Ch 4.7**.
- §5.5's dataset asymmetry is the mechanism behind the generalization claim in **Ch 8.4**.
- §5.6's headline-run-only strategy is justified by the cost analysis in **Ch 9.4**.
- §5.7's reproducibility guarantees are the empirical claim behind the released checkpoint pack.
