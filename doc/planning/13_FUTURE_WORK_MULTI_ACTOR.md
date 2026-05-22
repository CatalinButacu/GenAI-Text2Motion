# 13 — Future work: two-person and multi-actor text-to-motion

Status: **draft Ch 9 material**, authored 2026-05-22. The current thesis is single-actor; this document captures what we would do next.

## Literature scan (citations for the dissertation)

- **Inter-X** [Xu et al., 2024, CVPR, https://arxiv.org/abs/2312.16051]: 11,387 two-person interaction sequences, ~8.1M frames at 30 fps, captured in SMPL-X with finger articulation. 34,164 human-written annotations across 40 action categories. The canonical benchmark for two-person T2M.
- **InterGen** [Liang et al., 2024, IJCV, https://arxiv.org/abs/2304.05684]: shared-weight twin diffusion transformers with mutual cross-attention at every block, plus a relative-orientation loss between root joints. Trained on InterHuman (7,779 sequences). Most-cited two-person T2M baseline.
- **InterDiff** [Xu et al., 2023, ICCV, https://arxiv.org/abs/2308.16905]: human-object, not human-human, but introduces contact-conditioned diffusion that informs how to couple two streams.
- **ComMDM** [Shafir et al., 2024, ICLR, https://arxiv.org/abs/2303.01418]: freezes a pretrained single-person MDM and adds a small communication block between two parallel instances, fine-tuned on a handful of pairs. **Closest analogue to "reuse single-actor weights, add minimal inter-actor coupling"** — directly relevant to our proposed Mamba-ComMDM variant.
- **ReMoS** [Ghosh et al., 2024, ECCV, https://arxiv.org/abs/2311.17057]: reactive motion of one person conditioned on another's full sequence; asymmetric one-way cross-attention.
- **DuoLando** [Siyao et al., 2024, AAAI, https://arxiv.org/abs/2312.15660]: GPT-style autoregressive model with interleaved tokens from both dancers. Confirms AR token models (our family) can handle two-person interaction without diffusion.
- **in2IN** [Ruiz-Ponce et al., 2024, CVPRW, https://arxiv.org/abs/2404.09988]: per-individual and interaction-level text prompts ("A pushes B; B stumbles backward") fused at a cross-attention bottleneck. Relevant for our planner's per-actor action grammar.

Three families: shared-weight twin streams with mutual cross-attention (InterGen, Inter-X baseline, ComMDM, in2IN); conditional/asymmetric generation (ReMoS, InterDiff); interleaved-token AR (DuoLando). Pure independent generation with no coupling is nowhere in the literature — it is the trivial floor that all of these papers beat.

## Architectural hooks our codebase has for free

Confirmed via background-agent codebase scan (May 2026):

- `ScenePlanner.plan_parsed()` already emits `PlannedScene.entities` with `is_actor` flags and per-action `actor` / `target` fields. Multi-actor scenes are first-class in the planner output today (`src/modules/planner/planner.py`).
- `generate_action_clips()` in `src/modules/motion/clip_ops.py` iterates over sorted actions and **calls `motion_gen.generate(...)` once per action per actor**, seeding from the actor's previous clip via `last_pose_of`. Cross-actor state is never exchanged.
- `sequence_clips()` groups outputs by actor and returns `dict[actor_name, MotionClip]`. The downstream physics retarget consumes this dict actor-by-actor.
- `MotionGenerator` is a shared singleton — the same trained weights drive every actor.
- `src/data/amass/interx_loader.py` packs Inter-X into the existing 168-d SMPL-X format but **explicitly discards the pairing**: each `P1.npz` / `P2.npz` becomes an independent `SMPLXSample`. The relative-pose signal is lost at load time.

**What is free**: independent multi-actor generation with planner-chosen positions. **What is NOT free**: any cross-actor conditioning, relative-pose supervision, or paired-batch training.

## Three extension proposals, ranked by feasibility

### Proposal 1 — Independent generation with planner-coordinated positions
**Effort: ~1 person-week.** No new data, no architecture change. Add a multi-actor smoke test that drives two `PlannedEntity(is_actor=True)` through the existing pipeline and renders both into one scene. Document the floor — no interaction awareness, foot-skate and inter-penetration likely. **Outcome**: a demo video for the thesis defense, no quantitative claim. **This is what we can promise.**

### Proposal 2 — Shared-weight twin Mamba streams with state exchange ("Mamba-ComMDM")
**Effort: ~6-8 person-weeks.** Data: Inter-X paired loader (drop the line that splits P1/P2 into independent samples; emit `(motion_A, motion_B, text)` tuples). Architecture: instantiate two `TextToMotionSSM` instances sharing weights, add a single linear projection between their hidden states at every k-th Mamba block (the ComMDM recipe transplanted to SSM). Re-train only the projection on Inter-X pairs while freezing backbones. **Outcome**: measurable improvement in inter-person distance error vs Proposal 1; a credible "Mamba ComMDM" contribution. **This is the journal-version target.**

### Proposal 3 — Full Inter-X integration with paired RVQ retrain
**Effort: ~14-20 person-weeks.** Re-fit the 168-d normalisation on Inter-X (means/stds shift), retrain the RVQ tokenizer on the union of HumanML3D + Inter-X, retrain MotionSSM jointly on single-person and pair-token sequences, add a relative-pose decoder head. High risk: pairing changes the data distribution, tokenizer may collapse, and the planner's text grammar needs an "interaction-level" tier. **This is a follow-up thesis, not a thesis chapter.**

## What NOT to promise in the thesis

- Do **not** claim "we can generate two-person interactions" without retraining — Proposal 1 is geometric co-placement, not interaction synthesis.
- Reviewers will ask for **inter-penetration rate, contact precision/recall, and relative-orientation error on Inter-X**; we have no such numbers and computing them requires Proposal 2 minimum.
- Avoid claiming the model "generalises" to multi-actor — weight sharing across actors is parameter reuse, not generalisation, and there is no held-out paired evaluation.
- Do not state that SMPL-X hand articulation is "supported" beyond the loader: the RVQ tokenizer was trained predominantly on AMASS body channels and finger fidelity is untested.
- Do not promise "scaling to N>2 actors" — every cited baseline restricts itself to N=2 because cross-attention cost and dataset coverage both fail beyond two.

## Files of interest

- `src/modules/planner/planner.py` — multi-actor scene composition (free)
- `src/modules/motion/clip_ops.py` — per-actor independent generation (free)
- `src/modules/motion/generator.py` — `init_pose` seeding hook
- `src/data/amass/interx_loader.py` — Inter-X loader (currently discards pairing)
- `src/data/amass/smplx_pack.py` — 168-d feature packing
