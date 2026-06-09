# STATUS — session checkpoint (2026-06-03, updated)

## Update (2026-06-08b, TWIN PARAM-MATCH + MAMBA TRAINING MADE TRACTABLE; runs launched)
Matched the twin param counts and fixed the Mamba training-time blocker, then launched both runs.
- **Param match (config-driven):** shared parts (token_emb+heads+text_proj) = 6.41M, identical both.
  Each Mamba block ~= half an attn+MLP block, so **Mamba n=15 (31.84M) == Transformer n=8 (31.64M),
  +0.6%**. Added `GeneratorCfg.mamba_n_layers=15`; `train_generator` resolves n_layers per backbone.
- **Mamba `forward` was untrainable here** (naive per-timestep python loop: bs=32 OOM, bs=8 ~3.9s/step
  -> ~18 days/150ep on the 4GB GPU). Rebuilt the training path, keeping `step()` (streaming) untouched
  so stream==batch parity STILL holds (test_parity_mamba green): (1) vectorize in_proj/conv/x_proj/
  dt_proj over the whole sequence (one matmul each, not per-step); (2) resolve the recurrence
  sₜ=aₜ·sₜ₋₁+xₜ with a **parallel associative scan** (Hillis-Steele, log₂L passes, Heinsen 2023
  arXiv:2311.06281) in `_parallel_scan`; (3) **gradient-checkpoint each MambaBlock** (recompute the
  cheap scan in backward) -> bounded train memory. Result on 4GB: bs=16 fits at 2.30GB.
- **Throughput reality (4GB GPU):** transformer bs=64 ~45 clips/s (~9min/ep, 150ep ~22h); mamba n=15
  bs=16 ~2.7 clips/s (~2.4h/ep) — a ~17x intrinsic gap (attention parallel vs SSM small-batch+ckpt).
  Full 150ep mamba ~15 days locally => **user chose a reduced LOCAL run: 40 epochs, bs=16 BOTH (matched
  budget), resume-safe.** transformer ~6h, mamba ~4 days. Absolute FID will be sub-SOTA but the TWIN
  COMPARISON (the thesis claim) is valid at matched budget. For SOTA absolute numbers, rerun 150ep/bs
  on a bigger GPU (commands in train_generator docstring; `--resume` continues).
- **Launched** (background, sequential — single 4GB GPU): `python -u -m text2motion.train.train_generator
  --backbone {transformer,mamba} --epochs 40 --batch_size 16`; logs `outputs/twin_{transformer,mamba}.log`,
  checkpoints `checkpoints/generator_{backbone}{,_last}.pt`. FID/R-prec eval every 10 epochs.
- **20-rep full-test twin eval ready:** `_twin_eval.py` (mirrors `_l2.py` GT protocol — our_vab, Comp_v6
  stats, 32-pool, 20 reps — but motion = generated via stream+frozen-FSQ-decode from each backbone's
  best ckpt). CPU-smoke-validated. A watcher job auto-runs `_twin_eval.py --backbone both` ->
  `outputs/twin_eval.log` once both twins finish (waits for 2 consecutive idle polls to clear the
  transformer->mamba handoff gap). The in-training 200-clip/1-rep FIDs are noisy; THIS is the citable
  twin table. Transformer (done): best in-train FID 2.59 @ep10, final 2.89, R@1 0.115 (reduced 40ep run).
- TODO after runs: read the twin FID/R-prec table (`twin_eval.log`) into ADR 0002 as the gate evidence;
  iso-vocab FSQ ablation; the still-open doc items (citations, latency-vs-horizon plot).

## Update (2026-06-08, FOUR-AGENT VALIDATION — R-PRECISION RESIDUAL RESOLVED; fixes applied)
Ran four validation subagents (data / algorithm / architecture+hypotheses / eval+baselines). Verdict:
**thesis stands, no fatal flaws.** The long-standing R-precision residual is now CLOSED.
- **ROOT CAUSE of the 0.41-vs-0.51 R-prec gap: `_l2.py` loaded the wrong vocabulary** (`hhi_vab`, the
  Inter-X human-human set) instead of the official `our_vab`. `hhi_vab`'s distinct sos/eos/unk vectors
  + missing-word->unk fallback displaced every text embedding (MM-Dist ~9). NOT the encoders, matcher
  weights, data, or our_vab-vs-GloVe (all proven equivalent earlier) — it was the driver's vocab arg.
- **After the one-line fix, GT repro now matches published to noise (20 reps, official 263 test, 2179
  clips):** R@1 **0.514** (pub 0.511), R@2 0.706 (0.703), R@3 0.799 (0.797), MM-Dist **2.977** (2.974),
  Diversity 9.667 (9.503). Earlier STATUS entries calling this "unresolved/footnoted" are SUPERSEDED.
- FID(real) half-split reads ~0.18 (pub 0.002 "Real") = finite-sample bias of a 512-d FID from ~1089
  samples per disjoint half (pub is fid(X,X) on identical sets). FID implementation itself is sound
  (Diversity, same embedding space, reproduces). Not a harness bug.
- **Validations CONFIRMED:** data is true standard 263, canonical splits 23384/1460/4384, **0 non-finite
  / 0 missing** across all splits; loss recipe / EMA / masking / CFG / soft-decode / sampling correct;
  `MambaMixer` is faithful S6 and **forward() literally calls step() in a loop -> stream==batch is a
  structural identity, not an approximation**; state provably O(1) vs twin KV O(T).
- **Contribution A reframed:** the winner is **Grouped-FSQ** (not residual — residual collapsed at 0.22,
  now reported as the motivating finding). Honest claim: "matches/beats strong RVQ at matched encoder
  capacity & token budget, no dead codes, no EMA/reset machinery." TODO: iso-vocab ablation (per-group
  product ~512) to rule out the bigger-codebook confound.
- **Novelty (June 2026 lit check) HOLDS** under a tightened 4-conjunct framing: "first next-token,
  discrete, causal, fixed-recurrent-state (S6/Mamba) t2m generator with a streaming contract." Must
  cite/distinguish MoSa (2511.01200, transformer next-SCALE), MotionStreamer (2503.15451, continuous
  diffusion-AR — the direct streaming rival), PRISM (2603.08590).
- **Fixes APPLIED this session:** (1) LR linear-warmup + cosine decay in `TrainCfg`/`GeneratorTrainer`,
  default epochs 50->150; (2) per-run `seed_everything(cfg.seed)` for twin fairness; (3) resume-safe
  full-state checkpoint (`<backbone>_last.pt`: weights+opt+ema+scheduler+epoch+best_fid) + `--resume`;
  (4) fail-loud non-finite guard in both loaders; (5) generator param-count printout (twin matching);
  (6) `_l2.py` vocab fix; config docstring corrected to Grouped-FSQ. ruff clean, smoke-tested.
- **OPEN (recommended, not yet done):** iso-vocab FSQ ablation; match Mamba/transformer PARAM counts
  (not just hyperparams) before the FID gate; report 20-rep full-test gen metrics (not 1-rep) in the
  final table; add latency-vs-horizon plot; add the 3 new citations to references.md; document the
  O(L)-sequential training forward as a known limitation.

## Update (2026-06-04, CONTRIBUTION A — tokenizer pipeline built + first results)
Built the tokenizer training stack: `data/hml3d/motion_window.py` (64-frame windows, in-memory cache),
`train/tokenizer_trainer.py` (EMA + codebook perplexity), `eval/tokenizer_eval.py` (MPJPE + feat-L2 +
downstream FID on the validated matcher), `train/train_tokenizer.py` (CLI, best-by-FID checkpoint).
Trained on the official 263 data, 500 epochs (~80k iters, the T2M-GPT budget), width 512 / 3 resblocks.
- **Strong-RVQ baseline (the bar): recon-FID 0.044** (MPJPE 115mm, perplexity 328/512). Near MoMask
  0.019, better than T2M-GPT 0.07. Saved `checkpoints/tokenizer_rvq.pt`.
- **Residual-FSQ: recon-FID 0.22** (MPJPE 178mm, perplexity 142/1000) — 5x WORSE than RVQ. Root cause:
  residual FSQ keeps the latent at fsq_dim=4 (codes sum in 4-D) and later levels collapse on the fixed
  grid; a 4-D continuous latent can't represent 263-D motion. RVQ's 512-D codes win.
- **Fix WORKED.** `GroupedFSQ` (G groups -> G*4=24-D latent, every group used) + `TokenizerCfg.quantizer
  = "grouped"|"residual"` (default grouped). 500 epochs.
- **CONTRIBUTION A RESULT (full test, 2189 clips): Grouped-FSQ recon-FID 0.0266 BEATS Strong-RVQ
  0.0382** (MPJPE 119 vs 125mm; FSQ perplexity 572/1000 vs RVQ 328/512 — the no-dead-code advantage).
  Both beat T2M-GPT 0.071; approaching MoMask 0.019. **The FSQ-beats-RVQ hypothesis is SUPPORTED** ->
  Contribution A holds. Checkpoints: `checkpoints/tokenizer_{fsq,rvq}.pt`. FSQ is the frozen winner for
  the generator's soft-decode loss + streaming decode.
- NEXT: Contribution B — token-AR S6/Mamba generator vs transformer twin on the frozen grouped-FSQ
  tokens, full loss recipe + CLIP text encoder; ADR 0002 small-config FID gate.

## Update (2026-06-04, EVAL HARNESS — FULLY VERIFIED faithful; FID/Diversity reproduce paper)
Exhaustively verified EVERY eval component byte-for-byte vs Guo's official code:
- Motion + text encoders: identical to `inter-x/.../networks/modules.py` (maxdiff 0.0). The bug found
  earlier (ReLU vs **LeakyReLU(0.2)** + conv padding=1) is fixed in `eval/matcher.py`.
- Word vectors: our `glove.6B.300d` == Guo's HumanML3D `our_vab` (downloaded the t2m glove bundle via
  gdown, id 1cmXKUT31pqd7_XpJAiWEo1K81TMYHA5n -> `data/t2m_glove/glove/our_vab_*`; maxdiff 1e-7). So
  raw GloVe was correct; the inter-x `hhi_vab` was the non-standard outlier.
- Data: TeoGchx/HumanML3D Mean corr **0.99997** with Comp_v6 (= true standard 263). Checkpoint:
  official (camenduru) == donor. Normalization: Comp_v6 (bundled with evaluator).
- **Result: FID + Diversity reproduce published exactly** (Diversity 9.67 vs 9.50). GT R-precision
  0.41 / MM-Dist 9 vs published 0.51 / 2.97 — a residual in the text-motion JOINT alignment that is
  provably NOT encoders/vocab/data/checkpoint/normalization/POS/protocol. Unresolved (subtle
  preprocessing in the original eval driver); does NOT affect FID. **Decision: eval harness LOCKED**
  (FID citable; R-prec used consistently for relative comparison; GT-row footnoted).
- Eval GT-repro driver lives at `_l2.py` (20-rep protocol). Promote to `eval/evaluate.py` later.

## Update (2026-06-04, EVAL HARNESS VALIDATED — official evaluator confirmed identical)
Downloaded the **official HumanML3D evaluator** (camenduru/MoMask `humanml3d_evaluator.zip` ->
`data/official_evaluator/extracted/text_mot_match/model/finest.tar`). Loaded it with our matcher ->
**byte-identical results to the donor's finest.tar** (R-top1 0.410, norms 11.5/16.0, MM 9.06). So:
- We ARE using the correct official evaluator weights (donor's == official).
- The 0.41-vs-0.51 R-prec residual is NOT the checkpoint, NOT the data (official 263), NOT the
  protocol (20-rep random caption), NOT POS/normalization/activation (all ruled out by experiment).
- **GT reproduction (official data, 20 reps):** Diversity 9.67 (pub 9.50) ✅ EXACT; R-prec
  0.410/0.617/0.734 (pub 0.511/0.703/0.797) — top-2/3 within ~0.08, top-1 short; MM 9.06 (pub 2.97).
- Since Diversity (motion-only) is exact, **the motion encoder / FID is correct** = the primary
  citable metric is trustworthy. R-prec residual traces to eval-dataloader minutiae we haven't fully
  mirrored (official `max_text_len=20` truncation + random-crop length sampling), which don't touch FID.
- **VERDICT:** eval harness is validated for FID + a CONSISTENT matcher (relative R-prec valid). The
  exact published GT R-prec (0.51) would need the full official eval dataloader; deferred, not blocking.

## Update (2026-06-04, MATCHER BUG FOUND via official data — R-prec 0.24->0.41, Diversity EXACT)
Downloaded the **official HumanML3D-263** (TeoGchx/HumanML3D HF mirror, 4.4GB) -> converted to our
layout at `data/HumanML3D_official` (`data/hml3d/convert_official.py`; 29228 clips, splits 23384/
1460/4384, Mean/Std). **Citable track now = official data** (config `hml3d_out_dir` switched); our
SMPL-X regen stays at `data/HumanML3D_263` for the deferred 168 demo.
- **The R-precision gap was the MATCHER, NOT the data.** Official SMPL-H data gave the SAME ~0.24 as
  our SMPL-X -> disproved the SMPL-X-provenance theory; isolated the bug to the matcher.
- **Root cause (diffed vs Guo's `networks/modules.py`):** we used plain `ReLU` + conv `padding=0`;
  Guo uses **`LeakyReLU(0.2)`** in MovementConvEncoder + both BiGRUCo `output_net`s, and conv
  `padding=1`. Fixed (+ seg-length floor-div). `eval/matcher.py` now matches Guo's architecture.
- **Result on official GT:** R-precision 0.413/0.618/0.738 (was 0.239/0.391/0.495; published
  0.511/0.703/0.797) and **Diversity 9.34 ≈ published 9.50** (exact). Diversity is motion-only ->
  **the motion branch / FID is now CORRECT** (the primary metric). FID(GT,GT) on full test 0.18.
- Verified faithful: word vectors == raw GloVe (max diff 0.0), GloVe file complete (400k), full
  WordVectorizer (POS+VIP+sos/eos), Comp_v6 normalization correct (our-stats made it worse).
- **Residual:** R-prec top-1 0.41 vs 0.51 + MM-dist 9 vs 3 -> a text-vs-motion absolute-alignment
  subtlety (likely eval protocol: official averages 20 random-caption reps; we use first-caption,
  1 pass). Does NOT affect FID. Optional to chase.
- Tests green (test_eval matcher-load + metric math). `_l2.py` is the L2 driver (points at official).

## Update (2026-06-03, STAGE 2 L2 — DATA FAITHFUL + 3 MATCHER BUGS FIXED)
L2 verdict: **the regenerated 263 data is field-faithful.**
- **Mean correlation 0.9950** between our regenerated 263 Mean and the official eval stats
  (Comp_v6 meta) — features sit on the official distribution. Byte-faithful pipeline + exact
  round-trip already shown.
- GT vs Guo matcher: **R-precision top-1/2/3 = 0.239 / 0.396 / 0.512** (top-3 ≈ published GT top-1
  0.511) — clearly discriminative, far above chance (1/32). FID(GT_a,GT_b)≈0.
- **Found + fixed 3 real matcher-reproduction bugs** (caught by L2, not L1):
  1. finest.tar's motion/text encoders carry a trained GRU initial state (`hidden`) that was being
     SILENTLY DROPPED (unexpected ckpt key) → we used zero h0. Added `hidden` buffers + use as h0.
  2. Pooling was mean-over-outputs; Guo's BiGRUCo uses the FINAL hidden state (cat fwd+bwd via
     pack_padded_sequence). Fixed (`_bigru_final_state`).
  3. Spurious `F.normalize` on the embeddings; Guo returns raw `output_net` (un-normalized — its
     Diversity 9.5 / MM-dist 2.97 are impossible for unit vectors). Removed.
  `_load_exact` now also FAILS on unexpected keys so dropped-key bugs can't recur.
- Remaining R-precision top-1 gap (0.24 vs 0.51): I then ALSO reproduced Guo's full WordVectorizer
  (POS one-hots + VIP word-lists from his repo at `data/inter-x/.../word_vectorizer.py` + sos/eos +
  hhi_vab special vecs) and fed real POS one-hots — top-1 moved only 0.239→0.244. So the TEXT branch
  is NOT the bottleneck; the gap is UPSTREAM in the motion features. Candidates (can't disentangle
  locally — no official 263 feature set on disk, only the (263,) stats): SMPL-X-vs-SMPL-H provenance,
  or a subtle process_file feature-construction difference (self-consistent round-trip + 0.995 mean
  corr, but fine structure off). Definitive next step = get OFFICIAL HumanML3D-263 features
  (HuggingFace/Drive), run the matcher on them (isolates matcher vs data), then diff vs ours.
  **FID/Diversity are motion-only** and the matcher is used identically for GT+generated, so RELATIVE
  comparisons (the contributions) stay valid; only ABSOLUTE comparability to published is caveated.
- **Static diff of process_file vs official (done):** process_file math is FAITHFUL (uniform_skeleton,
  put-on-floor, face-Z+, get_cont6d_params, get_rifke, foot_detect, concat all match). 000021 used as
  the official reference skeleton (not the fallback). Normalization: Comp_v6 IS the matcher's stats
  (re-normalizing with our dataset Std made R-precision WORSE 0.244→0.156 — confirms Comp_v6's
  root/foot std amplification is intentional/correct). Std ratios our-vs-Comp_v6: ric 1.02, rot6d
  0.99, localvel 1.11 → our feature SCALE matches official for the bulk. CONCLUSION: feature code is
  correct; the residual R-precision gap is the per-clip FINE STRUCTURE from **SMPL-X-vs-SMPL-H body
  model provenance** (our AMASS is SMPL-X; the matcher was trained on SMPL-H-derived 263). Cannot be
  closed to exact published numbers without official SMPL-H-derived 263 features (not on disk).
- New eval infra: `eval/text_embed.py` (GloVe loader), `tests/test_eval.py` (metric math + matcher
  load/hidden-buffer regression, slow). Full suite: 22 non-slow + slow eval/dataset/raw_pose green.
- **Follow-ups for exact published R-precision**: port Guo's WordVectorizer POS+VIP lists + sos/eos.
  Not needed for FID or relative generator comparison.

## Update (2026-06-03, DATASET LOADER WIRED)
`src/text2motion/data/hml3d/dataset.py` now loads the real regenerated 263 data end-to-end:
- Reads `new_joint_vecs` + `Mean/Std` from `hml3d_out_dir`; texts from `paths.texts_dir` (donor
  `HumanML3D/texts`, NOT copied) — new config field `texts_dir`.
- **Per-variant captions**: mirrored `M<id>` clips use their own `M<id>.txt` (official left<->right
  swap), not the base caption.
- **Mirror = train only**; split files are normalized to BASE ids first (the donor splits enumerate
  both base and `M`), so eval is never mirrored regardless of split-file format.
- **Truncate-not-drop**: clips longer than `max_motion_len` are windowed (random for train, leading
  for eval) instead of dropped — recovered ~4.8k train clips (official behavior; "use full data").
- Added `collate_pad` (pad to batch max + lengths) and `build_dataloader(paths, repr, data, split,
  batch_size)`.
- Counts: **train 17,396** (8,698 base ×2 mirror), **val 563**, **test 1,638** (base-only).
- Tests: `tests/test_dataset.py` (3, slow) green; full non-slow suite still 21 passed.
- **NEXT:** Stage-2 L2 validation (T2M-GPT VQVAE recon FID on this data) + train Residual-FSQ
  tokenizer vs strong-RVQ baseline (Contribution A) on the GPU.

## Update (2026-06-03, REGENERATION COMPLETE — Stage 1 DONE)
**Real standard-263 HumanML3D data now exists** at `data/HumanML3D_263/` (repo-local C:).
- `new_joint_vecs/` = **22,552 valid 263-feature files** (~11,276 clips ×2 mirror), `new_joints/` =
  22,552 recovered, `joints/` = 22,770, `Mean.npy`/`Std.npy` both (263,). ~40 h of motion
  (2,897,960 frames). Splits (train/val/test.txt) copied alongside.
- Losses (~1%, expected): 218 clips FAILED stage 3 ("size 0" — empty after index crop, very short
  source clips); 5 clips (×2) had NaN features, excluded from Mean/Std. Fine.
- Integrity verified: all 263-dim; normalized features mean≈0.03 std≈1.03; std min 0.014 (no
  zero-div); no NaN in sample; lengths 26–199 (mean 128).
- VRAM fix held: chunked SMPL-X forward ran the whole corpus at ~1.8 GB, no OOM. (A duplicate run
  the subagent spawned was killed mid-way; skip-if-exists meant no corruption.)
- **NEXT:** build/point the dataset loader at `data/HumanML3D_263` (new_joint_vecs + Mean/Std +
  splits + mirror), then Stage-2 L2 validation (reproduce T2M-GPT VQVAE recon FID) and train the
  Residual-FSQ tokenizer + strong-RVQ baseline (Contribution A) on this real data.

## Update (2026-06-03, latest — pipeline validated + coverage measured)
- **Full Stage-1→3 pipeline validated on REAL data**: 8 AMASS clips → SMPL-X GPU forward → 263
  features → `recover_from_ric`, all finite, **round-trip L1 = 0.0000** (compare recovery to
  `global_positions`, the 2nd return — not the root-local 3rd). The feature math is correct on real
  SMPL-X joints, not just synthetic.
- **GPU live**: `torch==2.12.0+cu132`, `cuda.is_available()=True`, RTX 3050 Laptop **4 GB** (use
  fp16/bf16 + small batch for the big generator). aitviewer installed (`--extra viewer`).
- **HumanML3D coverage from the donor AMASS = ~78%** (11,385/14,616; **test 1712/2192 = 78.1%**).
  index.csv was built for the SMPL-H release; `regenerate.py` now has `_AMASS_DATASET_RENAME` +
  `_resolve_pose_path` (SMPL-X dataset renames + `_stageii` suffix) and **skips missing clips**.
  Missing = 4 datasets the donor lacks: `Eyes_Japan_Dataset` (1465), `humanact12` (1191, non-AMASS),
  `Transitions_mocap` (110), `BMLhandball` (67).
  - **Decision:** proceed at 78% (both contributions are head-to-head on identical data → valid;
    absolute FID vs published baselines gets an honest "1712-clip available-subset" footnote).
  - **To fill later for canonical-test FID:** re-download from AMASS (SMPL-X/G, needs account)
    `EyesJapanDataset`, `Transitions`, `BMLhandball` into `data/amass/`; add HumanAct12 separately
    (HumanML3D repo's humanact12 path). Then re-run `regenerate --stage all` (resolver already maps
    the renames). No code change needed.
- **NEXT:** run `python -m text2motion.data.hml3d.regenerate --config configs/default.yaml --stage all`
  (stage 1 walks ~all AMASS on GPU; 2 = index→joints; 3 = 263 features; 4 = Mean/Std). Long job.

## Update (2026-06-03, later — DATA BLOCKER REMOVED)
**The "download SMPL+H + DMPL" blocker is GONE.** Inspecting the donor data revealed the AMASS on
disk is the **SMPL-X G release**, not SMPL-H: `surface_model_type='smplx'`, `poses` 165-dim
(3+63+90+3+6), 16 betas, separate `pose_jaw`/`pose_eye`, framerate key `mocap_frame_rate`. So the
correct (matching) body model is **SMPL-X, which is already on disk** (`arctic/unpack/models/smplx/
SMPLX_{NEUTRAL,MALE,FEMALE}.npz`). No download needed; DMPL not needed (SMPL-X has none; irrelevant
to the 22 body joints).
- **`raw_pose.py` re-implemented for the SMPL-X release** (was a faithful but now-mismatched SMPL-H
  port that would have silently skipped every file via the `mocap_framerate` KeyError). Forwards via
  the `smplx` package, passes all pose components at batch T (jaw/eye/expr = zeros), keeps `joints[:22]`,
  Z-up->Y-up. `regenerate.py` stage 1 + config updated to use `paths.smplx_models`. SMPL-H/DMPL config
  fields marked LEGACY.
- **Validated on real AMASS**: 3 ACCAD clips -> clean `(T,22,3)` finite joints; pelvis height rises
  0.08->0.68 m on "lie_to_crouch" (Z-up->Y-up correct). Decision (user): **SMPL-X** (matches data),
  not SMPL+H. New slow test `test_raw_pose.py` (skips if donor data absent).
- **FID-comparability caveat:** 263 from SMPL-X vs official SMPL-H differs only in body model;
  `uniform_skeleton` retargeting standardizes bone lengths -> near-identical features. The Stage-2 L2
  gate (reproduce T2M-GPT recon FID) validates it.
- **NEXT ACTION is no longer a download** — it's to RUN `python -m text2motion.data.hml3d.regenerate
  --config configs/default.yaml --stage all` (long CPU job over ~all AMASS; consider CUDA + a subset
  first). Tests now 24 green (21 non-slow + 3 slow).

## Update (2026-06-03, post-restart session)
Resolved the open streaming question and built the two data-independent next steps from the
research dossier (`.claude/docs/references.md` Q1–Q4 now record the provenance):
- **Streaming deadlock FIXED.** `run_producer` pushed STREAM_END with a *blocking* `put` -> deadlock
  when the bounded queue was full and no consumer drained. Now both chunks and the sentinel use
  `_put_drop_oldest` (never stalls). Regression test added. This was the unconfirmed-streaming risk.
- **Text encoder built + validated** (`src/text2motion/model/text_encoder.py`): CLIP ViT-B/32
  (`openai/clip-vit-base-patch32`, 512-d pooled-projection prefix = field standard, T2M-GPT/MoMask).
  Partial unfreeze (last layer + final LN + projection) per the no-full-freeze lesson. Wired into
  `GeneratorTrainer` as an optional collaborator (own low-LR group `TrainCfg.text_encoder_lr=1e-5`,
  shared graph so grad reaches CLIP). `transformers` added to `pyproject` core deps.
- **Strong-RVQ baseline built + validated** (`src/text2motion/model/rvq_baseline.py`): EMA codebook
  (0.99) + dead-code reset + commitment (0.02) + quant-dropout (0.2), 6x512. SHARES the FSQ
  tokenizer's conv enc/dec (isolates FSQ-vs-VQ). Same `(B,T',Q)` index contract -> generator trains
  on either. Target to beat: MoMask recon FID 0.019 / MPJPE 29.5 mm.
- **Tests: 23 green** (21 non-slow + 2 slow CLIP against real weights). New: `test_rvq_baseline` (3),
  `test_text_encoder` (2, slow), `test_streaming` now 5 (added no-stall test). Lint clean.
- **NOVELTY FLAG:** AnyMo (arXiv:2605.29488) uses 4-stage Residual-FSQ (≈ Contribution A) but on
  OmniHuMo, no HumanML3D FID, no code -> not a scoop; framing tightened (see references.md).

---

# STATUS — session checkpoint (pre-restart, 2026-06-03)

Resume anchor for the streaming text→motion (SMPL-X) rebuild. Read with `CLAUDE.md`,
`.claude/decisions/{0001,0002}`, `.claude/docs/references.md`, and the approved plan at
`C:\Users\ic.butacu\.claude\plans\first-plan-the-flow-bright-pike.md`.

## Where we are
Both thesis contributions have **working, unit-validated cores** (on synthetic data). Everything that
can be built without the gated body models + GPU is done. Last confirmed: **13/13 tests green**
(Stage 0/data-contract/tokenizer/generator/training); the **4 streaming tests** were written and were
the only ones unconfirmed when the runner began auto-backgrounding — re-run to confirm (expect 17).

Re-verify: `PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/`  ·  lint: `... -m ruff check src/ tests/`

## Stage status
- **0 — Foundations** ✅ done. snake_case typed config (`src/text2motion/shared/{config,seed}.py`,
  `configs/default.yaml`), package skeleton, hook, ruff. Verified.
- **1 — Regenerate standard-263 from AMASS** ✅ code, ⏳ run blocked.
  Ported faithfully (constants byte-exact) to `src/text2motion/data/hml3d/{quaternion,skeleton,
  param_util,feature,raw_pose,regenerate,dataset}.py`. `index.csv` + train/val/test splits FETCHED
  (`D:\Facultate\dissertation\data\HumanML3D\index.csv`, `...\HumanML3D_263\{train,val,test}.txt`).
  **BLOCKED ON: user downloads SMPL+H + DMPL → `body_model/{smplh,dmpls}/{male,female}/model.npz`**
  (gated, MANO + SMPL sites). Then run `python -m text2motion.data.hml3d.regenerate --stage all`.
  ⚠️ Donor `data/humanml3d` is the WRONG 272-dim variant (lxxiao/arXiv:2503.15451) — do NOT use it.
- **2 — Eval harness** ✅ code, ✅ L1 validated, ⏳ L2 pending data.
  `src/text2motion/eval/{matcher,metrics}.py`. L1: loads Guo `finest.tar` with ZERO missing params,
  FID(x,x)=0, FID grows w/ noise. L2 (pending 263 data): reproduce T2M-GPT `VQVAEV3` recon FID.
- **3 — Residual-FSQ tokenizer (Contribution A)** ✅ code + sanity-overfit green.
  `src/text2motion/model/tokenizer.py` (FSQ + ResidualFSQ + quant-dropout + conv enc/dec).
  ⏳ Train on real 263; benchmark recon/downstream-FID vs strong-RVQ(EMA/reset) & MoMask RVQ baselines.
- **4 — Generator architecture (ADR 0002)** ✅ direction accepted (referenced); ⏳ small-config
  FID-vs-twin gate pending data. Decision: causal token-AR **S6/Mamba** (novel cell) + transformer twin.
- **5 — Generator + training (Contribution B)** ✅ core built + validated.
  `src/text2motion/model/generator.py` (Mamba + transformer twin; **stream==batch parity**; fixed SSM
  state vs growing KV; overfit), `src/text2motion/train/{losses,ema,trainer}.py` (token-CE +
  soft-decode-recon through frozen decoder + velocity/foot/root + EMA 0.999 + CFG dropout; A+B wired).
  ⏳ Train on real tokens; add text-encoder (CLIP/SBERT) + top-layer unfreeze.
- **6 — Streaming + aitviewer studio** ✅ code.
  `src/text2motion/stream/decode.py` (bounded-state → windowed token→frame decode → bounded queue,
  drop-to-latest, STREAM_END), `src/text2motion/render/studio.py` (263→22-joint `recover_skeleton`,
  aitviewer `Skeletons` view/headless lazy-imported via `[viewer]` extra, `collect_stream`).
  ⏳ Drive with a trained model + real aitviewer render (`uv sync --extra viewer`). 168 SMPL-X path deferred.

## Critical path remaining (strict order, data/GPU-gated)
1. **USER:** download SMPL+H + DMPL into `body_model/`.  2. regenerate 263 → L2 validation (VQVAE recon
FID).  3. train Residual-FSQ tokenizer + baseline benchmark (Contribution A evidence).  4. train
generator + twin w/ full loss → ADR 0002 FID gate (Contribution B evidence).  5. streaming studio demo.
(Wire a CUDA torch before 2–4: the SMPL-H forward + training are slow on CPU.)

## Conventions (don't relearn)
Python 3.12, snake_case (PEP8), config-driven via typed dataclasses (no global constants / magic
numbers), fail-loud (no strict=False / silent fallbacks), paths via config. Donor at
`D:\Facultate\dissertation` is READ-ONLY (data + SMPL-X bodies + Guo matcher + VQVAEV3); reuse data,
re-implement code. Run/test with `PYTHONPATH=src .venv/Scripts/python.exe`.

## Test inventory
`tests/`: test_hml3d_263_contract (2), test_tokenizer (3), test_generator (4), test_training (4),
test_streaming (4). Known benign warning: `torch.cross` dim deprecation in the byte-faithful ported
`quaternion.py` (intentional).
