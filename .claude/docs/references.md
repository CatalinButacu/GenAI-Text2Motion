# References & provenance

Every design choice here traces to a proven result. "Motion-proven" = reported on HumanML3D;
"image/audio-proven" = strong elsewhere, transferable hypothesis only. Keep this current as we build.

## Generator (Contribution B — causal token-AR S6/Mamba)
- **T2M-GPT** — Zhang et al., arXiv:2301.06052 (CVPR 2023). Causal AR transformer over VQ tokens;
  HumanML3D FID 0.116; streamable. → our **non-streaming-cost twin baseline** (causal transformer
  with KV-cache) + the EMA+reset VQ recipe.
- **Mogo** — arXiv:2412.07797; v2 arXiv:2506.05952. Causal RVQ transformer, *explicitly streaming*,
  long sequences (~13 s), FID 0.079. → the **mold** for our causal-RVQ streaming generator.
- **AttT2M** — arXiv:2309.00796. Causal AR + body-part attention, FID 0.112. → alt reference.
- **MoMask** — Guo et al., arXiv:2312.00063 (CVPR 2024). Masked-parallel residual transformer,
  FID 0.045. **Not streamable** (bidirectional). → reference ceiling + RVQ recipe.
- **MMM** arXiv:2312.03596, **BAMM** arXiv:2403.19435 — masked / hybrid; reference only (BAMM's
  refinement pass is bidirectional → fails the streaming gate).
- **Motion Mamba** arXiv:2403.07487, **T2M Mamba** arXiv:2602.01352, **KMM** arXiv:2411.06481 —
  Mamba inside *diffusion* or *masked* pipelines, all bidirectional. → establish the GAP: none is a
  token-autoregressive Mamba/S6 motion generator. This unoccupied cell is our novelty.
- **Mamba (S6)** — Gu & Dao, arXiv:2312.00752; **S4** — arXiv:2111.00396. → the selective-scan core.
- **MotionGPT** arXiv:2306.14795 — AR over tokens but LLM-heavy, FID 0.232; related work.

### Text encoder (caption conditioning) — confirmed field standard
- **CLIP ViT-B/32 text encoder** (Radford et al., arXiv:2103.00020). T2M-GPT, MoMask and Mogo all use
  the *identical* recipe (verified in their repos): the **pooled, projected 512-d** sentence vector,
  linear-projected, prepended as a **single prefix token** (no cross-attention). This is exactly
  `MotionGenerator.text_prefix`'s interface → keeping CLIP keeps our FID comparable. HF id
  `openai/clip-vit-base-patch32` (`CLIPTextModelWithProjection`, projection dim 512). Implemented in
  `src/text2motion/model/text_encoder.py`.
  - The field *fully freezes* CLIP; we follow the prior-plateau lesson ("don't fully freeze") with a
    minimal compromise — unfreeze only the **last transformer layer + final LN + text projection**
    (~3M params) at a low LR (`TrainCfg.text_encoder_lr`). Stays comparable to frozen-CLIP baselines.
  - Do NOT adopt T5 (MotionGPT / **AnyMo** arXiv:2605.29488): it changes the text feature space and
    breaks FID comparability with our T2M-GPT/MoMask baselines.

## Tokenizer (Contribution A — Residual-FSQ on HumanML3D-263)
- **FSQ** — Mentzer et al., arXiv:2309.15505. Finite scalar quantization; ~100% codebook usage, no
  collapse, no commitment/EMA/reset machinery. → our quantizer.
- **ScaMo** — arXiv:2412.14559. FSQ-VAE on motion; FSQ > VQ on HumanML3D. → motion-proof for FSQ.
- **Spatial-Temporal Multi-Scale Quantization** — arXiv:2508.08991. FSQ multi-scale motion tokenizer:
  recon FID 0.037 / gen FID 0.063, beats MoMask. → motion-proof FSQ beats RVQ.
- **MoMask** — arXiv:2312.00063. Residual VQ recipe (EMA + code reset + **quantization dropout**);
  single-VQ recon FID 0.091 → RVQ **0.019 / MPJPE 29.5 mm** (the target to beat). → strong baseline +
  residual structure + quant-dropout. Released RVQ checkpoint: github.com/EricGuo5513/momask-codes
  (`download_models.sh`, Google Drive, MIT; NO HF model repo). Config: 6 levels · 512 codes/level ·
  512 code-dim · stride ~4. ⚠️ verify the checkpoint's `opt.txt` `down_t` (default 3 = stride 8 vs the
  paper's factor 4) and match our stride for a fair comparison.
- **T2M-GPT** — arXiv:2301.06052. VQ ablation: naive 0.492 → EMA+reset 0.070 (7×). → the **baseline
  must include EMA+reset**, never naive VQ. VQVAE recon FID **0.071**.
- **Confirmed strong-RVQ baseline recipe** (implemented in `src/text2motion/model/rvq_baseline.py`):
  EMA decay **0.99** · commitment β **0.02** · `ema_reset` dead-code reinit · 6 levels · 512 codes/level ·
  quant-dropout **0.2** · velocity aux loss **0.1**. Shares the conv enc/dec with the FSQ tokenizer so
  the comparison isolates FSQ-vs-VQ at matched capacity. Sources: T2M-GPT + MoMask + EnCodec defaults.
- ⚠️ **NOVELTY FLAG — AnyMo** (arXiv:2605.29488, 2026) uses a **4-stage Residual-FSQ** motion tokenizer
  (≈ our Contribution A design). It does NOT scoop us: evaluated on OmniHuMo (not HumanML3D), **no**
  HumanML3D recon FID reported, **no code released**. Our defensible framing tightens to: Residual-FSQ
  **vs strong-RVQ head-to-head on HumanML3D-263 with citable FID**, combined with the streaming S6
  generator. The S6 streaming generator (Contribution B) remains the genuinely unoccupied cell.
- Lineage: VQ-VAE (van den Oord, arXiv:1711.00937), Jukebox (arXiv:2005.00341, EMA+random-restart),
  SoundStream (arXiv:2107.03312, RVQ+reset), EnCodec (arXiv:2210.13438).
- Optional cheap levers (image/audio-proven only — transferable hypotheses, flag as such):
  ViT-VQGAN low-dim+L2 codes (arXiv:2110.04627), SimVQ (arXiv:2411.02038), Rotation trick
  (arXiv:2410.06424), LFQ/MAGVIT-v2 (arXiv:2310.05737 — image/video only, vocab too large for HML3D).

## Large Motion Models / scaling (related work — cite & distinguish; we are NOT a scale competitor)
Our niche is a CONTROLLED mechanism study (SSM vs param-matched transformer twin, matched
data/seed/budget, standard HumanML3D + citable FID), not a scale race. Cite these as the scaling
zeitgeist and distinguish on the 4-conjunct claim (next-token, discrete, causal, fixed-state SSM).
- **LMM** — Zhang et al., arXiv:2404.01284. "Large Motion Model", Diffusion-Transformer + MotionVerse
  + ArtAttention. The canonical "LMM". → diffusion, not streaming-causal-AR → distinguish.
- **Being-M0 / MotionLib** — arXiv:2410.03311 (ICML 2025). First MILLION-clip dataset (~15x); shows
  scaling data+model matters; 2D **lookup-free (LFQ)** tokenizer "Motionbook". → JUSTIFIES our E7b
  AMASS pretraining lever; LFQ is FSQ's cousin (Lesson 4 landscape).
- **MotionMillion ("Go to Zero")** — arXiv:2507.07095. Million-scale data, zero-shot generation.
- **MotionGPT** — arXiv:2306.10900. Finetuned LLM on discrete motion codes (motion-as-language).
- **LLaMo** — arXiv:2602.12370 (Feb 2026). THE closest rival: scales pretrained LMs, Mixture-of-
  Transformers, **real-time streaming** motion — BUT **continuous** autoregressive latent (like
  MotionStreamer), NOT discrete next-token causal SSM. Must cite + distinguish on discrete+SSM.
- **MotionGPT3** — arXiv:2506.24086. Bimodal motion-language (motion VAE + diffusion head).
- **UMO** (2603.15975), **GENMO** (2505.01425): motion foundation / generalist models.
- Defensible framing: "scale is a separate axis; this thesis isolates the sequence-mixer mechanism
  (bounded-state SSM vs growing-KV transformer) for streaming t2m at matched budget — a result the
  foundation-model papers do not address."

## Data & evaluation (reuse, unmodified)
- **HumanML3D** + the **Guo et al. evaluator** (`text_mot_match`) — Guo et al., "Generating Diverse
  and Natural 3D Human Motions from Text", CVPR 2022; repo github.com/EricGuo5513/HumanML3D. → the
  standard **263** representation, the fixed FID/R-precision matcher, recovery code. Canonical 263
  eval stats present at donor `data/t2m_download/.../Comp_v6_KLD005/meta`; T2M-GPT's 263 VQ-VAE
  (`VQVAEV3...`) present → use to validate our harness by reproducing its reconstruction FID.
- **Ported pipeline (regeneration)** — `src/text2motion/data/hml3d/` faithfully ports, constants
  verbatim, from `github.com/EricGuo5513/HumanML3D` (main): `common/quaternion.py`,
  `common/skeleton.py`, `paramUtil.py`, `motion_representation.ipynb` (process_file /
  recover_from_ric), `raw_pose_processing.ipynb` (amass_to_pose, trans_matrix, swap_left_right),
  `cal_mean_variance.ipynb`. index.csv + train/val/test.txt fetchable from the same repo.
- **DONOR AMASS IS SMPL-X (not SMPL-H).** The donor `data/amass` is the AMASS **SMPL-X G** release
  (`surface_model_type='smplx'`, `poses` 165-dim, 16 betas, `pose_jaw`/`pose_eye`, key
  `mocap_frame_rate`). The original HumanML3D used the SMPL-H release. So our 263 regeneration
  (`raw_pose.py`) forwards through the **SMPL-X** body model (on disk: `arctic/unpack/models/smplx`)
  and keeps `joints[:22]` -- no SMPL+H/DMPL download needed. Body-model difference vs official is
  washed out by HumanML3D's `uniform_skeleton` retargeting; validated by the Stage-2 L2 recon-FID gate.
- **DONOR DATA CAVEAT — 272 ≠ 263.** The donor's `data/humanml3d` is the **272-dim** variant
  `lxxiao/272-dim-HumanML3D` (HF; arXiv:2503.15451), a different feature spec. Stat-aligning its
  `Mean[:263]`/`Mean[9:]` to the standard 263 gives corr ≈ −0.24 / −0.05 → **not convertible by
  slice/reorder; rejected.** The citable-263 track must source true standard-263 features.

## Lessons (donor diagnosis, not external)
- Prior plateau root-causes (token-CE-only loss, no EMA, 24 epochs, frozen encoder, greedy sampling,
  ~5M params, 14k clips) — `D:\Facultate\dissertation\.claude\docs\TRAINING_DIAGNOSIS_AND_PHYSICS.md`.
