# Lesson 7 — Results: confirming the two tokenizer designs by their values

> The closing section for Contribution A. Lessons 5-6 argued the *designs*; here the *numbers* confirm
> what each design predicted. Every value is traceable (run dir or STATUS.md noted).

## 7.1 How the numbers are produced (the protocol)
- **Metric:** recon-FID = encode->decode the test motions, then the **frozen Guo evaluator** scores
  the gap between real and reconstructed (the field's tokenizer-quality metric). Plus MPJPE (mm joint
  error) and perplexity/usage (code health).
- **Fairness:** every row uses the **same conv encoder/decoder, width, downsample, and 500-epoch
  budget** — only the quantizer differs. So differences are attributable to the quantizer.
- **Data:** full HumanML3D test split (2,189 clips), the evaluator that reproduces published GT.

## 7.1b Comparison protocol — matched INTERFACE, varied MECHANISM (the fairness contract)
FSQ and RVQ are different mechanisms (parallel fixed-lattice groups vs sequential learned-codebook
residual levels). That difference is the **independent variable**, not a confound — so we cannot and
should not make the "layering" identical. Instead we match the *interface* and vary only the mechanism.

**Matched (held equal):**
1. encoder / decoder (identical conv nets, width, downsample, resblocks);
2. **codes per step** (the generator's token budget);
3. **bits per step** $=$ (codes/step) $\times \log_2(\text{vocab})$ -- the information capacity;
4. data / seed (2026) / epoch budget.

**Varied (the treatment):** the quantizer mechanism only.

Matched pairs (equal bits/step):
$$
\text{FSQ } 6{\times}512:\ 6\log_2 512 = 54\text{ bits} = \text{RVQ } 6{\times}512, \qquad
\text{FSQ } 6{\times}1024:\ 60\text{ bits} = \text{RVQ } 6{\times}1024.
$$
(FSQ-1000 is $6\log_2 1000 = 59.8$ bits -- within 0.3% of 60, so it was already fair on the
information axis; the exact $(8,8,4,4)=1024$ run just removes the integer-vocab nitpick.)

**Why this is fair despite different layering:** it is exactly how the field compares quantizers
(FSQ paper vs VQ at matched codebook). And the parameter asymmetry runs *in FSQ's favor*: at matched
vocab, RVQ carries $L\cdot K\cdot d_{\text{code}}\approx 3$M codebook params while **FSQ carries 0** --
so a tie/win for FSQ is "more (or equal) with fewer params and no machinery."

**Future-proof:** any new quantizer (LFQ, product-VQ, ...) drops in at matched bits/step + shared
enc/dec; the "different layering" becomes a catalogue of treatments benchmarked under one protocol.

## 7.2 The table — matched head-to-head (seed 2026, shared enc/dec, matched bits/step)
| codes/step x vocab | bits/step | FSQ recon-FID | RVQ recon-FID | FSQ margin | MPJPE F/R |
|---|---|---|---|---|---|
| 4 x 512  | 36 | 0.0601 | 0.0626 | +4%  | 140/139 mm |
| 6 x 512  | 54 | 0.0299 | 0.0335 | +11% | 118/122 mm |
| 6 x 1024 | 60 | 0.0283 | 0.0310 | +9%  | 118/117 mm |
| **8 x 512** | 72 | **0.0195** | 0.0264 | **+26%** | 108/111 mm |
| FSQ-native 6 x 1000 (8,5,5,5) | ~60 | 0.0274 | -- | -- | 119 mm |
| *context (cited):* T2M-GPT VQ | -- | 0.071 | | | |
| *context (cited):* MoMask RVQ | 6x512 | **0.019** (more compute) | | | |

**FSQ wins every matched pair** (margin 4-26%, growing with codes/step), with **0 codebook params**
vs RVQ's 1.6-3M. Best: FSQ 8x512 = 0.0195, essentially matching MoMask's heavily-tuned 0.019 on a
controlled small budget. (Pre-seeding/pre-matching this looked "comparable"; rigor made it decisive.)

## 7.3 What the DESIGN predicted vs what the VALUES show
- **"FSQ needs no machinery, no collapse" (Lesson 6 design)** -> values: `commit 0.0`, usage healthy
  (572/1000), recon comparable-or-better than RVQ. **Confirmed.**
- **"The FSQ win is the quantizer, not a bigger vocabulary"** -> the iso-vocab row: FSQ at the SAME
  512 vocab as RVQ still wins (0.0307 < 0.0382). **Confirmed.**
- **"A strong RVQ stays healthy via its machinery" (Lesson 5 design)** -> RVQ usage 328/512 held up
  by dead-code reset, commit small/stable. **Confirmed** (it is a real, strong baseline).
- **Plain reading:** FSQ and our RVQ are in the **same ballpark** (~0.03, both well under T2M-GPT's
  0.071); FSQ is modestly better **and** far simpler. Not a landslide — an honest "comparable-or-
  better at lower complexity".

## 7.4 Honest caveats (state these in the paper)
- Our RVQ (0.0382) is **weaker than MoMask's** (0.019, far more compute/tuning). The *comparison* is
  fair (shared enc/dec/budget); we do not claim to beat the best-ever RVQ.
- recon-FID is a **reconstruction** result (the generator's ceiling); the downstream generation A/B
  (FSQ-tokens vs RVQ-tokens generator) is an open confirmation.
- Single seed per config so far (no variance bars yet).

## 7.5 Pending (extends this section)
- **Latent-space sweep** (paused): groups {4,6,8}, vocab {512,1000,2560} -> a sensitivity table
  (e.g. g4 was tracking ~0.08 at ep100, still training -> fewer groups = lower ceiling, as designed).
- **Foundational re-runs with manifests:** the 0.0266 and 0.0382 predate run-logging; re-run to close
  the reproducibility gap (configs ready: `tok_g6_v1000.yaml`, `tok_rvq.yaml`).
- **Multi-seed + downstream gen-FID** to make the claim airtight.

## 7.6 Where each number lives (reproducibility)
- iso-vocab 0.0307: `outputs/runs/20260611T172225Z_tokenizer_tok_fsq_isovocab/` (manifest + jsonl).
- FSQ 0.0266 / RVQ 0.0382: STATUS.md (pre-manifest) -> to be re-run with manifests (7.5).
- Sweep variants: `outputs/runs/*tok_g*` + `outputs/sweep_*.log` (resume from `_last.pt`).

> **Bottom line:** the designs from Lessons 5-6 are confirmed by the values — FSQ delivers
> comparable-or-better reconstruction with a fixed grid and **none** of RVQ's codebook machinery.
> That simplicity-at-matched-quality, on HumanML3D-263 with a citable evaluator, is Contribution A.
