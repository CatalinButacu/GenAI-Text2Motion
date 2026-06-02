# Reproducibility Runbook — From Empty Repo to Working Demo

> **Total budget**: ~$15 cloud + ~2-4 hours of local GPU training + ~30 minutes of setup.
>
> **End state**: a working end-to-end demo where you type a natural-language instruction and an avatar streams motion to your screen.
>
> **All commands are run from the repo root.** Time estimates assume a single-GPU local machine and a `g5.xlarge` cloud instance.

---

## Stage 0 — One-time environment setup (~15 min)

```bash
# Clone
git clone https://github.com/CatalinButacu/GenAI-Text2Motion.git
cd GenAI-Text2Motion

# Python 3.12 + deps (use uv if you have it, pip works too)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows PowerShell
pip install -r requirements.txt

# spaCy model used by M1 (scene parser); ~13 MB
python -m spacy download en_core_web_sm

# (Optional, only for cloud) AWS CLI + credentials
aws configure
```

**Verify Stage 0**:
```bash
python -m pytest tests/ -q --ignore=tests/benchmarks
# Expected: 373+ passed, 1 known failure (missing checkpoint), 3 skipped
python -m ruff check .
# Expected: All checks passed!
```

---

## Stage 1 — RVQ Tokenizer (~30 min local OR free via S3)

The RVQ tokenizer is the bridge between continuous motion and discrete tokens. **Two variants** matter:

### 1a) Reuse the existing symmetric-decoder checkpoint (FREE, recommended for first run)

```bash
mkdir -p checkpoints/rvq_tokenizer
aws s3 cp s3://dissertation-motion-cache-91340264/checkpoints/rvq_tokenizer/best_model.pt \
          checkpoints/rvq_tokenizer/best_model.pt
```

That single ~11 MB file is enough for offline MotionSSM training.

### 1b) Train the causal-decoder variant (~30 min on local GPU; required for streaming demo)

```bash
python scripts/training/train_rvq_tokenizer.py \
    --data amass \
    --causal-decoder \
    --epochs 50 \
    --batch-size 32 \
    --checkpoint-dir checkpoints/rvq_tokenizer_causal
```

Outputs `checkpoints/rvq_tokenizer_causal/best_model.pt`. Reuses the encoder + codebooks weights from 1a internally; only the decoder retrains.

**Verify Stage 1**:
```bash
python -c "
from src.modules.motion.rvq_tokenizer import MotionRVQTokenizer
import torch
ck = torch.load('checkpoints/rvq_tokenizer/best_model.pt', weights_only=False, map_location='cpu')
print('codebooks:', len([k for k in ck['model_state_dict'] if 'codebook' in k]))
print('OK' if ck['model_state_dict'] else 'FAIL')
"
```

---

## Stage 2 — Planner LM (Action-Decomposition Agent) — ~90 min local, free

**Note**: Stage 2 is **independent of Stage 1**; you can run it in parallel if you have CPU headroom.

### 2a) Generate the synthetic training dataset (~2 seconds)

```bash
python scripts/data/synthesize_planner_data.py --num-train 6000 --num-val 600
```

Outputs `data/planner/{train,val}.jsonl`. Reproducible — same seed → same dataset.

### 2b) Fine-tune GPT-2-small (~60-90 min on local GPU; ~5 hours on CPU)

```bash
python scripts/training/train_planner_lm.py \
    --epochs 3 \
    --batch-size 8 \
    --max-length 128 \
    --log-every 25 \
    --device cuda
```

Outputs `checkpoints/planner_lm/{pytorch_model.bin, tokenizer.json, ...}`. **Expected val_loss after epoch 3: ~0.05-0.10.**

### 2c) Validate generalization on the held-out paraphrase set (~30 sec)

```bash
python scripts/maintenance/eval_planner_generalization.py \
    --checkpoint checkpoints/planner_lm
```

**Dissertation passing thresholds**: valid_json ≥ 95%, valid_grammar ≥ 99%, canonical_hit ≥ 80%.

---

## Stage 3 — MotionSSM (Text → Motion-Token Model) — single cloud shot

This is the only stage that needs cloud compute. **Total cost: ~$10-15**, single 6-8 hour run.

### 3a) Pre-flight (do this locally before clicking `terraform apply`)

```bash
# Verify config drift catcher passes (constants.py == YAML)
python -m pytest tests/test_config_drift.py -q

# Verify streaming-equivalence still holds with whatever architecture changes
python -m pytest tests/test_streaming_equivalence.py -q
```

### 3b) Edit `scripts/cloud/aws/terraform.tfvars` (one-time)

```hcl
key_pair_name = "your-aws-key-pair"
s3_bucket     = "dissertation-motion-cache-91340264"
s3_cache_uri  = "s3://dissertation-motion-cache-91340264/cache/"
repo_url      = "https://github.com/CatalinButacu/GenAI-Text2Motion.git"
instance_type = "g5.xlarge"
use_spot      = false                 # protect against pre-empt before defense
data_source   = "humanml3d"
epochs_ssm    = 30
batch_size    = 64
text_encoder  = "clip-b"              # CLIP-ViT-B/32, beats SBERT on motion verbs
ar_k_head     = false                 # leave off for the headline; opt-in for ablation
compile_model = false                 # torch.compile is auto-skipped under DP anyway
single_gpu    = false                 # auto-wrap DP if multi-GPU instance picked
wandb_api_key = "..."                 # online wandb (optional)
```

### 3c) Launch + monitor

```bash
cd scripts/cloud/aws
terraform init    # one-time
terraform apply
# (creates 1× g5.xlarge with the AWS Deep Learning AMI + your IAM role)

# Watch progress
ssh -i ~/.ssh/<key>.pem ubuntu@$(terraform output -raw public_ip) \
    "tail -f /var/log/user-data.log"
```

Self-terminating contract: when training completes (success OR crash OR signal OR wall-clock 3 hours), the EC2 instance:
1. Syncs `checkpoints/motion_ssm/` to S3
2. Calls `aws ec2 terminate-instances --instance-ids self`

**No babysitting required.** Walk away. Budget alarm will email you at 50% / 90% / 100% of the $20 cap.

### 3d) Pull the result back local (~30 sec)

```bash
LATEST=$(aws s3 ls s3://dissertation-motion-cache-91340264/checkpoints/motion_ssm/ \
    | grep PRE | awk '{print $2}' | sort | tail -1 | tr -d '/')
mkdir -p checkpoints/motion_ssm/headline
aws s3 sync s3://dissertation-motion-cache-91340264/checkpoints/motion_ssm/$LATEST \
            checkpoints/motion_ssm/headline
```

---

## Stage 4 — End-to-end demo (~10 seconds per instruction)

You now have all three checkpoints (RVQ + MotionSSM + planner LM). Wire them up:

```bash
python scripts/demo/run_streaming_demo.py \
    --planner-ckpt checkpoints/planner_lm \
    --motion-ckpt  checkpoints/motion_ssm/headline/best_model.pt \
    --rvq-ckpt     checkpoints/rvq_tokenizer/best_model.pt \
    --instruction  "walk forward until you reach the tree, then turn left, then wave" \
    --scene-object "tree=5,0,0"
```

Outputs `runs/streaming_demo/<timestamp>/`:
- `instruction.txt` — what you typed
- `plan.json` — planner decomposition
- `frames.npy` — raw 168-d SMPL-X frames
- `events.jsonl` — per-frame timing + position
- `summary.json` — TTFF, total wall, generation fps

**This is the defense demo.** Run it a few times, pick the best one, screen-record for safety.

---

## Stage 5 — Empirical validation (the dissertation's load-bearing figure)

### 5a) Constant-memory benchmark (~5 min on GPU)

```bash
python scripts/maintenance/benchmark_streaming_memory.py \
    --device cuda --ts 50 250 1000 2000 4000
```

Outputs `runs/streaming_benchmark/results.json` + console table:

```
     T    parallel_mem   parallel_s    stream_mem    stream_s   mem_ratio
    50         97.3 MB       0.39s       68.9 MB      6.70s       1.41x
   ...
  4000       2503.8 MB       7.67s       68.9 MB    564.04s      36.32x
```

**Pass criterion**: `streaming_mem` is flat (≤ 5% spread across T). This is Chapter 7's headline figure.

### 5b) Planner generalization eval (~30 sec)

```bash
python scripts/maintenance/eval_planner_generalization.py
```

**Pass criteria**: valid_json ≥ 95%, canonical_hit ≥ 80% on the held-out 25-instruction set.

### 5c) CFG inference sweep (~3 min on GPU, optional but free)

```bash
python scripts/maintenance/validate_cfg_local.py \
    --checkpoint checkpoints/motion_ssm/headline/best_model.pt \
    --rvq-checkpoint checkpoints/rvq_tokenizer/best_model.pt
```

Sweeps cfg_scale ∈ {1, 2, 4, 7}. Pick the scale that minimises FID; bake into the demo.

---

## Dependency graph

```
Stage 0 (env)
   │
   ├──> Stage 1a (RVQ S3 pull)        ────┐
   │                                       │
   ├──> Stage 1b (causal RVQ retrain) ────┤
   │                                       │
   ├──> Stage 2a (planner dataset)         │
   │       │                                │
   │       └──> Stage 2b (planner train) ──┤
   │              │                          │
   │              └──> Stage 2c (eval) ─────┤
   │                                          │
   └──> Stage 3a (preflight) ────────────────┤
          │                                    │
          └──> Stage 3b (tfvars)               │
                 │                              │
                 └──> Stage 3c (cloud launch) ──┤
                        │                        │
                        └──> Stage 3d (pull) ────┤
                                                 │
                              Stage 4 (demo) <──┘
                                  │
                                  └──> Stage 5 (validate) ──> defense
```

Stages 1 and 2 are independent — run in parallel if your machine has headroom.

Stage 3 (cloud) is the only stage that costs money; everything else is electricity.

---

## Cost + time summary

| Stage | Wall | $ | Notes |
|---|---:|---:|---|
| 0 — env setup | 15 min | 0 | one-time |
| 1a — RVQ S3 pull | 1 min | 0 | use the existing one |
| 1b — causal RVQ retrain | 30 min | 0 | local GPU |
| 2a — planner dataset | 2 sec | 0 | combinatorial generation |
| 2b — planner fine-tune | 60-90 min | 0 | local GPU |
| 2c — generalization eval | 30 sec | 0 | local CPU OK |
| 3a — preflight | 1 min | 0 | local tests |
| 3b — tfvars edit | 5 min | 0 | one-time |
| 3c — cloud train | 6-8 hr | $10-15 | g5.xlarge on-demand |
| 3d — checkpoint pull | 30 sec | 0 | |
| 4 — demo run | 10 sec / instr | 0 | local GPU or CPU |
| 5a — memory bench | 5 min | 0 | local GPU |
| 5b — planner eval | 30 sec | 0 | |
| 5c — CFG sweep | 3 min | 0 | local GPU |
| **Total** | **~9 hr active** | **~$15** | |

---

## Reproducibility guarantees

1. **All datasets are deterministic given a seed.** `--seed 42` everywhere.
2. **`tests/test_config_drift.py`** asserts `configs/motion_ssm.yaml == constants.py`. CI fails if you edit one without the other.
3. **`tests/test_streaming_equivalence.py`** asserts `stream_step` over T calls matches `forward()` over T tokens within 5e-4 float tolerance. The streaming claim cannot silently regress.
4. **Every cloud run snapshots its full resolved config** + git commit hash + dataset hash into `<checkpoint_dir>/run.json`. Any released checkpoint traces back to the exact code that produced it.
5. **The benchmark, eval, and demo scripts all write to dated `runs/<script>/<timestamp>/` directories** so you can re-run without overwriting old measurements.

---

## Failure recovery

| Symptom | What to check |
|---|---|
| `pytest test_config_drift` fails | Someone edited the YAML or constants.py without the other; reconcile |
| Cloud run uses more $ than expected | Check `terraform output cost_estimate`; budget alarm at 50% / 90% |
| Planner emits invalid JSON | Run `eval_planner_generalization.py`; if canonical_hit < 80%, retrain with more epochs |
| Demo CLI errors "Missing checkpoints" | Stage 1 / 2 / 3 didn't finish; see preflight output |
| Streaming memory grows with T | Bidirectional Mamba was used; check `bidirectional=False` in the headline config |

---

## What's NOT in this runbook (deferred to journal version)

- Multi-actor extension (Inter-X paired loader) — Ch 9.1
- WASM browser demo — Ch 8.7
- n=60 human A/B study — Ch 8.8

Defense version is built end-to-end by following Stages 0-5 above.
