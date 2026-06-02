# 11 — References for future implementation work

Curated reading list for the techniques in use (and the techniques we'd
*consider* next). Grouped by topic; arXiv IDs given so the URL survives even
if the abs page format changes.

Verified-as-of: 2026-05-22. Links rot; if anything 404s, search the title.

## How to use this list

- The "must read first" entries under each section are the ones that make the
  *rest* easier to read.
- Code-first links (state-spaces/mamba, alxndrTL/mamba.py, etc.) often
  illuminate a paper faster than the paper itself, especially for SSM math.
- Visual guides go at the bottom — read them BEFORE the papers if the math
  in a paper isn't clicking.

---

## 1. Mamba / State-Space Models

The architecture under `src/modules/motion/ssm.py` and `nn_models.py`.

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**
  Gu & Dao, NeurIPS 2023.
  https://arxiv.org/abs/2312.00752
  *The S6 paper. What our `MambaLayer` implements (the Python scan; not the CUDA kernel).*

- **Mamba-2 / Structured State Space Duality (SSD)**
  Dao & Gu, ICML 2024.
  https://arxiv.org/abs/2405.21060
  *Reformulates the recurrence as a structured-matrix multiply -> single matmul on tensor cores. ~3-5x faster than the scan implementation; recommended for the next training run after current research stabilises.*

- **state-spaces/mamba** — official CUDA implementation.
  https://github.com/state-spaces/mamba
  *Contains `mamba_ssm.ops.selective_scan_fn`. Drop-in replacement for our Python scan, Linux/CUDA only.*

- **alxndrTL/mamba.py** — pure-PyTorch parallel scan.
  https://github.com/alxndrTL/mamba.py
  *Windows-friendly fallback when CUDA kernels can't compile. ~3-5x over naive sequential loops.*

- **S4: Efficiently Modeling Long Sequences with Structured State Spaces**
  Gu, Goel, Re. ICLR 2022.
  https://arxiv.org/abs/2111.00396
  *The "S4" papers Mamba grew out of. Skip if Mamba is enough; revisit if continuous-time formulations matter later.*

## 2. RVQ / VQ-VAE Tokenization

The codebook head under `src/modules/motion/rvq_tokenizer.py`.

- **Neural Discrete Representation Learning** (VQ-VAE original)
  van den Oord et al., NeurIPS 2017.
  https://arxiv.org/abs/1711.00937
  *Foundation. The straight-through estimator + commitment loss we use comes from this paper.*

- **SoundStream: An End-to-End Neural Audio Codec** (RVQ origin)
  Zeghidour et al., 2021.
  https://arxiv.org/abs/2107.03312
  *Introduces residual VQ stacks. The K-codebook pattern in our `ResidualVectorQuantizer` traces back here.*

- **High Fidelity Neural Audio Compression (EnCodec)**
  Defossez et al., 2022.
  https://arxiv.org/abs/2210.13438
  *EMA codebook update + dead-code revival -- both used in our `RVQCodebook`.*

- **Finite Scalar Quantization (FSQ)**
  Mentzer et al., 2023.
  https://arxiv.org/abs/2309.15505
  *Replaces VQ entirely with rounding to a fixed grid. Eliminates the commitment loss + codebook collapse. Worth an ablation in a second wave, but FSQ in residual cascades has a "magnitude decay" problem (see next).*

- **Residual Magnitude Decay in FSQ Cascades** (2025)
  https://arxiv.org/abs/2509.09550
  *Why FSQ-RVQ is not a free win for motion. Read before considering an FSQ migration.*

## 3. Text-to-Motion Generation

The papers your dissertation needs to compare against.

- **MoMask: Generative Masked Modeling of 3D Human Motions**
  Guo et al., CVPR 2024.
  https://arxiv.org/abs/2312.00063
  *Direct reference for our K=6 codebook config + Residual Transformer for AR head. Their reported FID 0.045 / top1 ~25% on HumanML3D is the bar we're chasing.*

- **MotionGPT: Human Motion as a Foreign Language**
  Jiang et al., NeurIPS 2023.
  https://arxiv.org/abs/2306.14795
  *LLM-as-motion-generator framing. Useful for the agent-layer direction (see `07_AGENT_REACT.md`).*

- **T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations**
  Zhang et al., CVPR 2023.
  https://arxiv.org/abs/2301.06052
  *VQ-VAE + GPT. Closest architecturally to our pipeline (discrete tokens + autoregressive). InfoNCE contrastive aux loss recipe comes from here.*

- **Motion-Agent**
  Wu, Zhao et al., 2024.
  https://arxiv.org/abs/2405.17013
  *Closest published prior art for the conversational layer; GPT-4 dispatcher over MotionLLM. Reference for `07_AGENT_REACT.md`.*

- **Iterative Motion Editing with Natural Language**
  Goel et al., SIGGRAPH 2024.
  https://purvigoel.github.io/iterative-motion-editing/
  *"Motion editing operators" compiled from natural language by an LLM. Reference for multi-turn edit chat.*

- **MDM (Human Motion Diffusion Model)**
  Tevet et al., ICLR 2023.
  https://arxiv.org/abs/2209.14916
  *Diffusion-on-pose baseline. We chose discrete RVQ over diffusion; cite this for the comparison.*

- **HumanML3D** (dataset)
  Guo et al., CVPR 2022.
  https://github.com/EricGuo5513/HumanML3D
  *Our primary text-paired training corpus.*

- **T2M Evaluator** (the standard FID/R-Precision protocol)
  Guo et al., CVPR 2022.
  https://github.com/EricGuo5513/text-to-motion
  *`finest.tar` checkpoint that our `scripts/evaluation/compute_fid.py` loads. T2M-protocol FID, R-Precision@K, diversity, multimodality all defined here.*

## 4. Text Encoders (SBERT, CLIP)

The text-conditioning layer under `src/modules/motion/nn_models.py::PretrainedTextEncoder`.

- **Sentence-BERT** — Reimers & Gurevych, EMNLP 2019.
  https://arxiv.org/abs/1908.10084
  *Theoretical basis for `all-MiniLM-L6-v2`. The siamese setup that makes pooled BERT useful as a sentence embedding.*

- **CLIP: Learning Transferable Visual Models From Natural Language Supervision**
  Radford et al., 2021.
  https://arxiv.org/abs/2103.00020
  *Why CLIP wins for motion: trained on captions describing actions and physical events ("man jumps", "person throws", ...). Direct alignment with HumanML3D's vocabulary.*

- **MotionCLIP: Exposing Human Motion Generation to CLIP Space**
  Tevet et al., ECCV 2022.
  https://arxiv.org/abs/2203.08063
  *First major use of CLIP for motion. Read alongside MoMask to understand the encoder-choice rationale.*

- **MoCLIP** — CLIP fine-tuned on motion captions (2025).
  https://arxiv.org/abs/2505.10810
  *Best-in-class text encoder for motion in 2025. Beyond the dissertation timeline but useful as future work.*

- **sentence-transformers documentation**
  https://www.sbert.net/
  *API we use to load both SBERT and CLIP. `SentenceTransformer(name).encode(texts)`.*

## 5. Training tricks (CFG, AR sampling, optimisation)

The mechanisms under `--cfg-scale`, `--ar-k-head`, `--compile`, `seed_all`.

- **Classifier-Free Diffusion Guidance**
  Ho & Salimans, 2022.
  https://arxiv.org/abs/2207.12598
  *The CFG formulation we use: `guided = unc + s * (cond - unc)`. Originally for diffusion, transfers cleanly to discrete-token models.*

- **Light-T2M** (CFG for text-to-motion, 2024)
  https://arxiv.org/abs/2412.11193
  *Specifically applies CFG to T2M; informative for the cfg_scale sweep range (typically 2-4).*

- **MoMask CFG ablation** — see Section 4 of the MoMask paper above. *Their CFG sweep: 0.05-0.1 FID improvement at scale=4.*

- **Mogo: RQ Hierarchical Causal Transformer**
  https://arxiv.org/abs/2412.07797
  https://arxiv.org/abs/2506.05952
  *EOS-token-based length prediction (vs our regressed length head). Worth porting in a second wave.*

- **Lion** — optimizer
  Chen et al., 2023.
  https://arxiv.org/abs/2302.06675
  *AdamW alternative; saves ~33% optimizer memory. Hyperparameter-sensitive: needs ~1/3 - 1/10 the LR of AdamW. Try after architecture changes stabilise.*

- **PyTorch reproducibility guide**
  https://pytorch.org/docs/stable/notes/randomness.html
  *Spec for what we wire in `src/shared/seed.py::seed_all`. `CUBLAS_WORKSPACE_CONFIG` notes here.*

- **W&B reproducibility report**
  https://wandb.ai/sayakpaul/reproducible-ml/reports/Increasing-Model-Reproducibility-With-Weights-Biases--Vmlldzo3ODMxNQ
  *Best-practice for logging seed config + checkpoint metadata; matches what `src/shared/seed.py::seed_dict` produces.*

## 6. Implementation references

- **PyTorch torch.compile** (the speed flag under `--compile`)
  https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
  *Tutorial. Use `mode="reduce-overhead"` with bucketed/padded batches (which we already do via `max_motion_length`).*

- **karpathy/nanoGPT** — the canonical "small GPT" reference repo.
  https://github.com/karpathy/nanoGPT
  *Useful as a sanity reference for AR sampling, top-p, and CFG-style logit blending. Most patterns transfer to discrete-token motion generation.*

- **aitviewer (headless rendering)**
  https://eth-ait.github.io/aitviewer/
  *Documentation for our M6 renderer. Y-up coordinate convention; `HeadlessRenderer` for offscreen video.*

- **uv** — the dep manager we adopted in 2026-05.
  https://docs.astral.sh/uv/
  *Project setup, lockfile semantics, CUDA-torch handling.*

## 7. Evaluation

- **FID for motion** (T2M protocol)
  See Guo et al. 2022 + the T2M Evaluator repo above. Our `compute_fid.py` mirrors this protocol.

- **Foot-skating metric** (SIGGRAPH MIG 2023)
  Onuki & Kanai, doi:10.1145/3623264.3624443.
  *FID is insensitive to foot-skate artifacts; this paper proposes a velocity-when-grounded metric. Worth adding as a second-wave evaluation.*

- **R-Precision @ K** — same Guo et al. 2022 paper.
  *32-sample pool (1 GT + 31 distractors). Our `compute_fid.py::compute_precision_r` follows this.*

- **HumanML3D-Eval / Evaluator FAQ**
  https://github.com/EricGuo5513/text-to-motion
  *Issues tab is the de-facto FAQ for evaluator gotchas. Read before reporting FID numbers.*

## 8. Visual guides & tutorials (read these FIRST)

When the papers in §1-§5 don't click, start here.

- **A Visual Guide to Mamba and State Space Models** — Maarten Grootendorst.
  https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state
  *The clearest accessible introduction to S4/S6/Mamba. Diagrams of the recurrence, the selective-scan idea, why "selective" matters. Read before §1.*

- **The Annotated S4** — Sasha Rush.
  https://srush.github.io/annotated-s4/
  *Walks through S4 line-by-line in JAX with mathematical commentary. Read alongside §1's Mamba paper to understand the discretisation step.*

- **The Annotated Transformer** — Sasha Rush.
  http://nlp.seas.harvard.edu/annotated-transformer/
  *Older but still the cleanest line-by-line transformer walk-through. Useful as a baseline reference when reading newer architectures.*

- **The Annotated Diffusion Model** — Niels Rogge & Kashif Rasul.
  https://huggingface.co/blog/annotated-diffusion
  *Diffusion 101 in PyTorch + HuggingFace. Useful background even though we chose discrete tokens over diffusion.*

- **LLM Powered Autonomous Agents** — Lilian Weng.
  https://lilianweng.github.io/posts/2023-06-23-agent/
  *Reference for `07_AGENT_REACT.md`. Survey of planning, memory, tool-use architectures.*

- **PyTorch internals: how `torch.compile` works**
  https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/
  *Edward Yang's 2025 update on graph capture + inductor backend. Useful when debugging compile failures.*

- **MoMask GitHub README** (with reproduction commands)
  https://github.com/EricGuo5513/momask-codes
  *The published codebase. Reference for hyperparameter values + training-recipe details our `train_motion_ssm.py` should match where possible.*

---

## Adding new references

Keep this file curated. Rules of thumb:

- Add a link only if you'd recommend it to your future self after 6 months.
- One line of "why this matters" beats a paragraph.
- arXiv IDs over arXiv abs URLs (the URL format is stable; the abs page layout isn't).
- Code links beat paper links when the goal is "make this work in our repo".

Bad: a link to a SOTA leaderboard, a Hacker News thread, or a Medium post that
just summarises a paper. Good: anything that's been useful at least once
already.
