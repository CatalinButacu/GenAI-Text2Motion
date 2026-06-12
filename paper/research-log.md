# Research Log — day by day, every number traceable

Companion to `proposal.md`. Source of truth: `.claude/docs/STATUS.md` + `outputs/runs/` manifests.
Convention: **bold** = a result that survives into the dissertation tables.

## 2026-06-03 — Foundations, faithful port, first validations

- Clean repo scaffolded (typed dataclass configs, snake_case, ruff, fail-loud rules) after the
  prior attempt plateaued (top-1 ~12%, FID never computed). Donor project demoted to a read-only
  data/asset source.
- HumanML3D-263 feature pipeline re-implemented byte-faithfully (quaternion/skeleton/feature math
  verified constant-exact); round-trip recovery L1 = 0.0000 on real SMPL-X joints.
- Discovery: donor AMASS is the **SMPL-X release** (165-dim poses) → no gated SMPL+H download
  needed; `raw_pose` rewritten for SMPL-X; Z-up→Y-up validated on real clips.
- Full regeneration: 22,552 valid 263-dim clips (~40 h motion). Eval matcher (Guo `finest.tar`)
  loaded with **3 reproduction bugs found and fixed**: silently-dropped trained GRU initial state,
  mean-pooling instead of final-hidden-state, spurious L2-normalize. GT R-prec 0.239 → 0.512 (top-3).
- Streaming queue deadlock fixed (blocking sentinel put); CLIP text encoder (partial unfreeze) and
  strong-RVQ baseline built. Tests green.

## 2026-06-04 — Official data + evaluator locked; Contribution A first results

- Switched citable track to the **official HumanML3D-263** (TeoGchx mirror; 29,228 clips, canonical
  splits 23,384/1,460/4,384). Matcher architecture bug found by diffing Guo's code: LeakyReLU(0.2)
  + conv padding=1 → **R-precision 0.239→0.413, Diversity 9.34 ≈ published 9.50 (exact)**.
  Official evaluator checkpoint verified byte-identical to the donor's. **Eval harness LOCKED.**
- Tokenizer stack trained (500 epochs, T2M-GPT budget):
  - Strong-RVQ baseline: **recon-FID 0.0382** (full test, 2,189 clips; MPJPE 125 mm, perplexity 328/512).
  - Naive Residual-FSQ: **collapsed at 0.22** (4-D summed lattice cannot span 263-D) — kept as the
    motivating negative finding.
  - **Grouped-FSQ: recon-FID 0.0266, MPJPE 119 mm, perplexity 572/1000 — beats strong-RVQ → H1
    supported.** Both beat T2M-GPT's released VQ (0.071).

## 2026-06-08 — Four-agent validation; twins made trainable; runs launched

- Independent validation of data/algorithm/architecture/eval. **Root cause of the last GT gap: the
  eval driver loaded the wrong word vocabulary (`hhi_vab`)** → after the one-line fix, GT
  reproduces published exactly: **R@1 0.514 / 0.511 pub, MM-Dist 2.977 / 2.974 pub** (20 reps).
- Novelty re-verified vs June-2026 literature; framing tightened to "first next-token, discrete,
  causal, fixed-recurrent-state t2m generator" (cite & distinguish MoSa, MotionStreamer, PRISM).
- Mamba training forward rebuilt: vectorized projections + Hillis-Steele parallel scan + per-block
  gradient checkpointing (bs16 fits in 2.3 GB on the 4 GB GPU). Twins **param-matched at 31.6M vs
  31.8M (+0.6%)**. LR warmup+cosine, seeded twin fairness, resume-safe checkpoints added.
- Local reduced twin runs launched (40 ep, bs16); transformer finished (best in-train FID 2.59
  @ep10 on the noisy 200-clip probe), mamba OOM-crashed locally → moved to cloud.

## 2026-06-09 — Cloud twin run (g4dn.xlarge, eu-west-1)

- Terraform infra: on-demand g4dn.xlarge + S3 + SSM + cost guards (36 h hard lifetime, idle-GPU
  watchdog). Both twins, 150 epochs, bs 64, sequential.
- **Transformer: 150 epochs in 5h57m (~2.4 min/ep). Best in-train FID 3.24 @ epoch 20 (R@1 0.193);
  monotonic train-CE descent to 1.08 while eval degraded → overfits from ~ep 20.**
- Mamba started 19:27 UTC at ~35–43 min/epoch (eager scan, see 06-10 analysis).

## 2026-06-10 — Diagnosis day: code review, sampling lock, Track A, harvest

- **Live-read + full code review.** Six findings: (1) model selection leaked onto the test split;
  (2) CFG trained but never applied at inference; (3) weight_decay=0 + no input corruption;
  (4) cosine schedule decayed past the empirical peak; (5) caption-segment crop could mislabel
  pairs; (6) in-train 200-clip/1-rep FID is trend-only. All fixed same day (val-split selection,
  wd 0.01, pkeep 0.8, 60-ep schedule, segment-aware caption choice).
- **Checkpoint-loss bug averted:** run.sh synced only logs; started an on-box `ckpt-sync` service
  via SSM 15 min before it would have mattered.
- **Sampling locked (E4), existing ep-20 transformer ckpt, val 300 clips:**
  cfg-scale sweep 1→8 + temperature sweep at 5/6 → **cfg 5.0, T 1.1, top-p 0.9: FID 6.136→3.463
  (−44%), R@1 0.105→0.213; R@1 keeps rising to 0.249 at scale 8 while FID turns** (over-guidance
  onset). Zero retraining.
- **Track A shipped** (28 tests green): explicit-opt-in fused mamba-ssm kernel (fail-loud import),
  16-token CLIP prefix (P=1 parity-tested), END token (soft-decode-excluded; fixed-length mode for
  protocol), **100M twins param-matched: 94.70M (transformer 12L/d768) vs 96.37M (mamba 23L,
  +1.77%)**; bf16. **Sanity-overfit gate PASS both: CE 7.0 → 0.003/0.004.**
- **Why mamba trains ~16× slower (38 min vs 2.4 min/ep):** eager scan materializes
  (B,L,d_inner,d_state) intermediates → memory-bandwidth-bound on the T4 + checkpointing recompute
  + 15 vs 8 layers. Implementation artifact, not architecture: resolved by the fused kernel (E5);
  inference remains O(1)/step (H3).
- **Mamba pilot result (200-clip trend, no CFG): ep20 FID 4.80 / R@1 0.193 == transformer's ep20
  R@1; ep30 FID 4.17 / R@1 0.208 — above ANY transformer epoch; ep40 R@1 0.094 = the same
  overfitting turn, one decade later → H4 registered.** Best (ep-30 EMA) checkpoint harvested to
  `checkpoints/generator_mamba_cloud.pt`; **box terminated 7 h early** (no evals remained), ~$4 saved.
- Iso-vocab FSQ ablation (E2) training locally: 500 ep; at ep 100 recon-FID 0.054 and falling,
  perplexity 331/512 (healthy, no dead codes).
- Eval driver promoted to `text2motion.eval.evaluate` + **MultiModality** implemented (100×30,
  10 pairs); streaming benchmark harness (`eval/streaming_bench.py`) built — CPU smoke already
  shows the KV-grows-vs-state-fixed mechanics. Run manifests (`shared/run_log.py`) wired into all
  entrypoints; publish chain automated (post-commit hook → donor/thesis-v2 → GitHub).

## 2026-06-11 — Twin table v0: the ADR-0002 gate PASSES

- **E3 complete — first citable generation numbers** (full test 2,189 clips, 20-rep, CFG 5.0/T1.1):
  **transformer 31.6M: FID 3.310, R@1 0.217±0.008, MModality 3.747 · mamba 31.8M: FID 3.928,
  R@1 0.168±0.006, MModality 4.059.** FID ratio **1.19× ≤ 1.5×** → **gate PASSED**, E5 unblocked.
  Caveats recorded in ADR 0002 (epoch-budget asymmetry 150 vs ~42; resolved by E5).
- Incidents (both caught by guards, both now structural): the pilot transformer ckpt failed loud on
  the new max_seq_len default (fixed via `configs/eval_pilot31m.yaml`); the iso-vocab run froze
  mid-epoch and was killed after burning 10.6 h CPU → **`scripts/local_guard.ps1`** (heartbeat-stall
  + max-hours watchdog) is now mandatory for local runs (CLAUDE.md rule), relaunch resumed E2.
- Design note: SE(2) placement augmentation rejected (RIC features are invariant by construction);
  E7b (AMASS generator pretraining) registered as the real data-side lever.

## 2026-06-12 — E2 complete: H1a CONFIRMED (iso-vocab FSQ still beats RVQ)

- **Iso-vocab Grouped-FSQ (6 groups × 512 = RVQ's vocab, matched encoder/budget, 500 ep):
  best recon-FID 0.0307 / MPJPE 119.5 mm / perplexity 340/512 — beats strong-RVQ 0.0382 by 20%
  at EQUAL vocabulary → the FSQ win is not a codebook-size artifact.** Ladder: FSQ-1000 0.0266 <
  FSQ-512 0.0307 < RVQ-512 0.0382 (vocab helps, but the quantizer is the difference).
  Run was interrupted twice (frozen-run kill; planned machine handback) — the relaunch with
  `--resume` support + local guard completed cleanly overnight.
- g5 canary (E5 gate): prebuilt mamba-ssm wheels are ABI-incompatible with the DLAMI torch (cu130)
  and the box ships only the 12.8 toolkit → canary now installs cuda-toolkit-13-0 and force-builds
  from source (sm_86). Cost lesson: the original gate-1 failure left a never-busy GPU idling ~15 h
  (~$18) — watchdog gained a never-busy-within-90-min terminate path.
- **Streaming finding (E6):** the AR-transformer twin **hard-crashes at the horizon where its
  learned absolute-position table ends** (CUDA index assert past position 96) — streaming beyond
  the trained horizon requires positional surgery (clamping/RoPE/extension), while the SSM has no
  positional bookkeeping at all and streams indefinitely. Recorded as a qualitative limitation in
  the H3 chapter; benchmark sizes the table to the horizon for the latency/memory measurement.
- All-AMASS pretraining corpus built (`pretrain_corpus.py`): pose_data already covered the FULL
  donor AMASS (16,407 sequences) → CPU-only featurization; 2,188 val/test-underlying sources
  excluded (leakage guard); official normalization preserved. `tokenize_corpus.py` collapses the
  corpus to a tens-of-MB int16 token pack for the cloud (E7b).

## Pending (auto-queued)

- [ ] E2 verdict: iso-vocab final recon-FID vs RVQ 0.0382 (ETA 2026-06-11 ~07:00 UTC).
- [ ] **Twin table v0** (E3): `evaluate --backbone both --split test --cfg_scale 5.0
  --temperature 1.1`, 20-rep, full test, both cloud ckpts → ADR-0002 gate verdict.
- [ ] E7 AMASS regen + tokenizer pretraining (GPU queue after E2).
- [ ] E7b (registered 2026-06-11) AMASS **generator** pretraining: local regen+tokenize → token pack
  to S3 → unconditional CE-only pretrain on cloud → conditional HumanML3D fine-tune. Design note:
  SE(2) placement augmentation was considered and rejected — the 263 RIC representation already
  factors out ground-plane rotation+translation (bit-identical features), so "more motion" (this
  experiment) is the real variance lever, not more views of the same motion.
- [ ] E5 g5.xlarge canary (kernel parity at scale) → final 100M twin run.
- [ ] E6 streaming benchmark on final checkpoints → the H3 figure.
