# Dissertation completion plan (from 2026-07-06 to defense)

Synthesized from a planning pass + an evaluation pass (both subagents), grounded in the repo's
verified citations. Neutral, succinct. Every task names its deliverable and how it ties to a
contribution and to the official-paper comparison.

## Where we stand (progress, in context)

DONE and measured:
- Contribution A: Grouped-FSQ tokenizer, full 3x5 matrix, best recon-FID 0.0170 (beats our strong
  RVQ at all six matched cells; single seed per cell).
- Contribution B: 34M controlled twin on the full test set (2189 clips, 20-rep) -- transformer
  FID 3.30 / R@1 0.246 vs mamba FID 4.06 / R@1 0.255. Result = parity (neither mixer dominates).
- Streaming memory result: mamba O(1) state 2.68 MB vs transformer O(L) KV to 605 MB at 8192 (exact,
  kernel-free).
- AMASS-pretraining ablation (relative FID delta at pilot scale).
- Interactive aitviewer demo studio.
- Draft cores written (sec 1, 4, 5) + paper/FEEDBACK.md.

IN FLIGHT / SINGLE-SIDED:
- 100M mamba twin fine-tuning on AWS (eager; best-by-val syncing to S3). The 100M transformer exists
  at VAL FID 1.63 only -- it is single-sided and not on the test protocol.

Framing that governs the whole plan: the 34M twin already carries Contribution B, so the 100M pair
is a scale-confirmation, off the critical path.

## Comparison anchors (verified; reference ceiling, not a target)

All measured with the frozen Guo `text_mot_match` evaluator, HumanML3D-263 test, 20-rep -- the same
harness this repo reuses unmodified. That shared harness is what makes any comparison admissible.

- T2M-GPT (arXiv:2301.06052): gen FID 0.116, R@1 0.417; VQ recon-FID 0.071; ~230M-class.
- MoMask (2312.00063): FID 0.045, R@1 0.521; RVQ recon-FID 0.019. NON-streaming (bidirectional).
- Mogo (2412.07797): FID 0.079; streaming causal-RVQ.
- AttT2M (2309.00796): FID 0.112.
- GT reference R@1 ~0.511; we reproduce 0.514 (harness validity check).

Honesty rules attached to every comparison: (i) scale gap (34M/100M vs ~230M) stated; (ii) FID is
sample-size biased -- name test cardinality as its own caveat and reconcile the 1638 / 2189 / 4384
figures in one sentence; (iii) MoMask FID always cited adjacent to "non-streaming/bidirectional";
(iv) our rows never bold a win over these baselines.

## Phases

### Phase 0 -- Advisor decision gate [REQUIRED, do first]
Send paper/FEEDBACK.md; get written answers to: (a) is "parity + O(1) memory" the right headline;
(b) is the 34M twin sufficient or is the 100M mamba mandatory; (c) is disclosing the AMASS-prior
batch-size asymmetry enough. Pre-write the CONTINGENCY: if the advisor demands the 100M twin or a
batch-matched prior, that re-acquires a cloud dependency on infra that has repeatedly failed
(kernel wall, box self-terminations) -- decide now what ships if cloud fails again.
Ties to: decides whether Phase 1 is required or optional, and how the baseline rows are framed.

### Phase 1 -- Finish the 100M twin [off critical path]
1.1 Let the 100M mamba finish; best-by-val lands in S3 (eager; ~36 h cap risk; 10-min sync covers a
    cap-stop).
1.2 One-time TEST eval of BOTH 100M twins with the IDENTICAL cfg/temp/length as the 34M table
    (cfg 6 / temp 1.0 / fixed GT-length). The existing 100M transformer 1.63 is VAL and must be
    re-scored on test, or it is not on-protocol. Run on-box before teardown (cheaper) or locally.
1.3 terraform destroy; record final spend in STATUS.
Ties to: completes the matched 100M row; "gap closes with scale" becomes measured, not projected.
Constraint: do NOT ship a single-sided 100M transformer row in the MAIN twin table -- until the
mamba pair exists, it goes to a footnote/appendix only (an asymmetric row on a parity page misreads).

### Phase 2 -- Tables + figures (manifest-driven) [REQUIRED]
2.1 Table 1 -- tokenizer matrix, with cited rows T2M-GPT VQ 0.071 and MoMask RVQ 0.019; keep the
    controlled-budget caveat (our RVQ 0.0221 < MoMask's tuned 0.019 is not a best-ever claim).
2.2 Table 2 -- controlled twin table: GT row, tokenizer-ceiling row, the 34M pair, the 100M pair (if
    1.2 done), cited T2M-GPT 0.116 + MoMask 0.045 (non-stream) as reference ceiling. Keep the
    `stream state` column (O(L) KV vs O(1)) -- it is the differentiator FID alone hides.
2.3 Table 3 -- AMASS ablation as a RELATIVE delta (do not let the absolute 300-clip-val number read
    as a headline next to the 3.30/4.06 test figures).
2.4 Streaming figure: memory panel LEADS (exact, kernel-free); latency panel kept with its two
    caveats (eager ~2x, 4 GB memory-limit discontinuity). Regenerate on a kernel-capable GPU only if
    one becomes available.
2.5 FIGURES INVENTORY (was under-scoped): the PDF currently has ONE figure. Add: tokenizer + generator
    architecture diagrams; FSQ-vs-RVQ Pareto plot; training/CE curves backing sec 5.8; and QUALITATIVE
    rendered motion samples (still figures -- the demo video does not substitute). A text-to-motion
    thesis with no visual generated-motion figures is a predictable examiner complaint.
2.6 Generalization scatter (train-vs-test recon-FID) [NICE] -- pre-empts an overfit objection on A.
Ties to: A's core evidence + B's heart; cited rows make explicit this is matched-budget parity, not SOTA.

### Phase 3 -- Contribution A closure decision [REQUIRED to resolve, one of two]
Recon-FID is currently the ONLY evidence for A, but the claim language is unbounded. Pick one:
(a) Run the FSQ-tokens vs RVQ-tokens downstream GENERATION A/B (one 34M gen run + one test eval,
    ~$5-10 cloud or local) -- answers "does better reconstruction translate to better generation?",
    the single most predictable examiner question on A; OR
(b) Explicitly BOUND the Contribution A claim to RECONSTRUCTION in the abstract + contributions.
Do not leave it optional while the claim is unbounded.

### Phase 4 -- Prose expansion to 30-45 pp [REQUIRED, long pole]
Absorb paper/lessons/ into the numbered sections: sec 4.1-4.3 (FSQ math / grouped-residual / RVQ
baseline, lessons 05-07); sec 5.1-5.3 (generator arch / controlled twin / stream==batch parity,
lessons 08-13); sec 2 Related Work (lesson 16 -- distinguish MotionStreamer / MoSa / AnyMo / LLaMo:
the discrete + causal + fixed-state conjunction is the novelty defense); sec 3 Representation/Data/
Protocol (lessons 01-04, incl. the Guo-evaluator validity story + FID sample-size bias); sec 1 polish
+ sec 6-7. LIMITATIONS depth (own explicitly): single-seed, 34M-not-100M headline, test cardinality,
eager-mode latency, AMASS-prior batch asymmetry, and a couple of qualitative FAILURE cases. Add an
ETHICS + data-licensing subsection (AMASS/SMPL-X/HumanML3D are license-gated; text-to-motion is
dual-use) -- standard and currently missing.

### Phase 5 -- LaTeX/Overleaf + front matter [REQUIRED]
Build paper/references.bib (resolve the MotionStreamer id collision -- 2503.15451 is flagged in
references.md as the 272-dim variant; give it the correct id; complete CoverThomas/AMASS/VQVAE).
Port markdown to the institution thesis template (booktabs tables, TikZ redraws of the mermaid/DOT
diagrams -- they will not render as-is, \includegraphics figures, $$ math already compatible).
Front matter (currently absent from the plan): title page, signed originality/plagiarism
DECLARATION, acknowledgements, table of contents + list of figures/tables + abbreviations, abstract
formatting to the page/word spec, a similarity check (Turnitin or equivalent), and a code/data
availability + compute note. Full compile, page-count check, proofread.

### Phase 6 -- Reproducibility [REQUIRED]
Replace the hardcoded numbers in scripts/figures/make_figures.py with manifest-driven generation:
read outputs/runs/*/metrics.jsonl (+ manifest.json), pin ONE canonical seed-2026 run dir per cell,
emit each table + figure. Acceptance check: EVERY quoted table number traces to a run dir produced
AFTER the R-precision harness fix (the 0.24 -> 0.41 -> 0.514 correction) -- a stale pre-fix number in
a table is a defense-killer. Then a one-command regen entry point + a number-to-run-dir index.
Ties to: makes the thesis's "every number traces to a run directory" claim executable, not aspirational.

### Phase 7 -- Robustness [PROMOTED from optional]
Multi-seed (2-3) on the 34M twin: the parity headline currently rests on ONE seed (R@1 lead ~1.5
sigma, FID gap 23%) -- the weakest scientific link, and this pair is local/cheap. Promote to
required-if-feasible; at minimum report single-seed as the top acknowledged limitation. Iso-vocab
FSQ row (near-free harvest from an existing run) -- rules out the "bigger codebook" confound.

### Phase 8 -- Defense [REQUIRED]
Slide deck (~15-20): streaming gap -> thesis statement -> Contribution A (matrix + Pareto figure) ->
Contribution B (twin table + the neutral parity framing) -> streaming MEMORY figure (the money slide)
-> AMASS ablation -> honest limitations -> conclusion; explicit "we claim mechanism + streaming, not
SOTA" slide against MoMask/T2M-GPT. Demo video (1-2 min live streaming decode + headless MP4). Dry-run
+ anticipated-questions sheet (parity vs SOTA; why not more seeds; kernel/eager caveat; R-prec-vs-FID
split; novelty vs MotionStreamer/MoSa; recon-only vs downstream for A).

## Neutrality / phrasing fixes (fold into the writing everywhere)
- "First ..." always carries the 4-conjunct qualifier: first DISCRETE, CAUSAL, token-autoregressive
  STATE-SPACE (S6/Mamba) text-to-motion generator with a BOUNDED-STATE streaming contract.
- Drop "modestly": state the trade symmetrically -- transformer leads FID + diversity, SSM leads all
  R-precision ranks + MM-Dist, neither dominates.
- "Pareto-dominates" -> "consistently beats the strong RVQ at all six matched cells (single seed),
  with zero codebook parameters" (keep "Pareto" only alongside the explicit params axis).
- AMASS "3x FID cut" stays a relative ablation delta, not an absolute headline.

## Critical path and effort
Coupled long pole = Phase 3-closure + Phase 4 prose + Phase 5 LaTeX/front-matter + Phase 2.5 figures;
these consume most of the ~20-25 working-day required budget and cannot be parallelized against each
other. Phase 1 (100M) runs concurrently and off-path. Start prose expansion in parallel with the 100M
run now.

REQUIRED for a defensible defense: 0, 1.2 (or an explicit 34M-only decision), 2.1-2.5, 3 (A/B or
bound the claim), 4, 5, 6, 7 (at least single-seed disclosed), 8.
NICE / drop under pressure (disclose as limitations): 100M pair if cloud fails, 2.6, full multi-seed,
camera-ready kernel streaming figure, batch-matched AMASS prior.

## Risks the plan must hold in view
1. Prose + LaTeX + figures is the whole budget, not a week -- treat as one coupled pole; start now.
2. Advisor gate may re-impose a cloud dependency on infra that has burned repeatedly -- have the
   fallback pre-written.
3. Single-seed everywhere is a headline risk, not just a caveat (see Phase 7).
4. Number provenance across the harness saga -- verify every table number is post-fix (Phase 6 check).
5. The studio demo will surface failure cases -- include a couple in Limitations rather than have the
   viva surface them live.
