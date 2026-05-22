# Dataset Expansion Roadmap — Which Verbs Each Dataset Unlocks

> **Question**: If we integrate dataset *X*, what new verbs become renderable that the current 28-verb set cannot do? **Ordered by ROI** (new verbs per week of integration effort), not alphabetically.

This document is an investigation, not a commitment. The thesis defense uses *only* what is already trained (HumanML3D + AMASS, 28 verbs). The roadmap exists so the journal version and any future thesis based on this code knows where to invest next.

---

## TL;DR (rank by ROI)

| Rank | Dataset | Verbs unlocked | Integration effort | Cost | Bottom-line ROI |
|---:|---|---:|---|---:|---|
| 1 | **BABEL** | ~150 | 2-3 weeks | Free | Highest verb-per-week; same SMPL-X format as our pipeline |
| 2 | **NTU RGB+D 120** | ~90 | 4-6 weeks | Free | Strong domestic + social actions; format conversion needed |
| 3 | **Inter-X** | ~60 (two-person verbs) | 6-8 weeks | Free | Already registered in codebase; thesis multi-actor extension |
| 4 | **Motion-X** | ~120 (body + face + hand) | 8-12 weeks | Free | Hands + face channels; biggest format jump |
| 5 | **Arctic** | ~30 (dexterous) | 10-14 weeks | Free | Hand-object detail; needs new RVQ for finger channels |
| 6 | **KIT-Motion-Language** | ~20 (locomotion variants) | 3 weeks | Free | Low ROI per week; gait-style nuance only |

The dissertation explicitly says **none of these are required for defense**. The thesis claims "streaming-capable single-actor text-to-motion." Multi-actor and richer vocabularies are Ch 9 future work.

---

## 1. BABEL — Bodies, Action and Behavior with English Labels [Punnakkal et al., CVPR 2021]

**What it adds.** Dense action labels overlaid on AMASS clips. Where HumanML3D has ~3 captions per clip, BABEL provides per-frame action segments tagged with categories from a 250-action ontology. New verb buckets we don't currently cover:

| Bucket | Example verbs new to our system |
|---|---|
| Domestic | cook, sweep, mop, hammer, polish, fold-laundry, iron, type |
| Sports | swim, climb, ski, surf, dribble, throw-ball |
| Exercise | squat, lunge, push-up, sit-up, jumping-jack, deadlift |
| Specific gestures | point, beckon, salute, shrug, nod, shake-head |
| Social (single-person but social-coded) | greet, agree, disagree, applaud |
| Medical | stumble, collapse, faint, limp |

**Why it's the highest ROI.**
- Same underlying motion data as our HumanML3D path (AMASS). No new feature extractor needed.
- Labels come as time-aligned segments — we can train the MotionSSM directly on (label, clip-segment) pairs.
- Crowd-sourced labels are ~95% reliable per the BABEL paper; we don't need re-annotation.

**Integration cost.**
- ~1 week to write the BABEL loader → emit `(label, segment_motion, segment_text)` triples.
- ~1 week to extend the planner LM's `VERB_PARAPHRASES` with the new ~150 verbs + retrain.
- ~3-5 days for the MotionSSM headline retrain on the union of HumanML3D + BABEL.
- ~3-5 days to refresh ablations, eval scripts, documentation.

**Why we didn't do this for the thesis.** Out of scope by Ch 9. BABEL is the obvious first follow-up.

---

## 2. NTU RGB+D 120 [Liu et al., TPAMI 2020]

**What it adds.** 120 distinct action classes recorded as RGB + depth + IR + 3D skeleton. New verbs vs our set:

| Bucket | Example verbs |
|---|---|
| Domestic | brush-teeth, eat, drink, wear-jacket, take-off-jacket, put-on-glasses |
| Medical | sneeze, cough, blow-nose, fall-down, vomit, headache |
| Social (single-person actions in social context) | salute, clap-hands, drink-toast, point-at-something |
| Object-manipulation | tear-paper, type-keyboard, write, take-selfie, count-money |
| Specific motion patterns | shake-fist, hand-up, hopping-on-one-leg, kick-something |

**Strengths.** Large action catalog with crisp class boundaries, well-curated, widely cited.

**Weaknesses.** Skeleton format differs from SMPL-X — needs joint-to-SMPL retargeting (open libraries exist, ~1 week of work). RGB/depth modalities are useless for our pipeline; we'd consume only the skeleton stream.

**Integration cost.**
- ~2 weeks for the skeleton retargeting pipeline (NTU joints → SMPL-X axis-angle).
- ~1 week for the loader + RVQ retrain.
- ~1 week for planner LM + MotionSSM retrain.
- ~1 week for evaluation refresh.

---

## 3. Inter-X [Xu et al., CVPR 2024] — TWO-PERSON INTERACTIONS

**What it adds.** 11k two-person interaction clips with SMPL-X output. **All verbs are inherently dual-actor.** New buckets:

| Bucket | Example verbs |
|---|---|
| Conflict | fight, punch-partner, kick-partner, shove |
| Affection | hug, kiss-on-cheek, hold-hands, lean-on |
| Cooperation | dance-with, support-partner, carry-together, lift-partner |
| Communication | hand-shake, high-five, fist-bump, salute-partner |
| Following | follow, lead, mirror-movements |

**Why it's strategically important.** This is the only credible path to *interactive scenes* — the user types `"two people meet and shake hands"` and the system can render both avatars correctly. Without Inter-X, the planner can position two actors via M2 (ScenePlanner) but each avatar moves independently with no awareness of the other.

**Integration is bigger than just new verbs** — it requires the full multi-actor extension described in `doc/planning/13_FUTURE_WORK_MULTI_ACTOR.md`. Three proposals are documented there, ranked by feasibility (independent generation with planner coordination → ComMDM-style mutual cross-attention → full paired RVQ retrain).

---

## 4. Motion-X [Lin et al., NeurIPS 2023]

**What it adds.** 81k whole-body (face + body + hand) motion-text pairs from diverse internet sources. New buckets:

| Bucket | Example verbs |
|---|---|
| Musical | play-piano, play-guitar, conduct-orchestra, beatbox |
| Detailed hand | snap-fingers, finger-counting, sign-language, point-with-index |
| Facial | laugh, frown, smile, surprised, angry |
| Dance variants | ballet, hip-hop, tango, breakdance |
| Sport-specific | golf-swing, tennis-serve, basketball-shot, soccer-kick |

**Strengths.** Largest motion-text corpus; whole-body coverage including face.

**Weaknesses.** Quality is variable (internet sources); face channels are noisy. Our MotionSSM head currently does not predict face/hand channels — adding them requires a new RVQ head architecture.

**Integration cost.** Largest format jump. Estimated 8-12 weeks for full integration; alternatively use only the body-channel subset (still ~30 new verbs) at ~4 weeks effort.

---

## 5. Arctic [Fan et al., CVPR 2023]

**What it adds.** Hand-object dexterous manipulation with sub-cm accuracy. New verbs:

| Bucket | Example verbs |
|---|---|
| Grasp variants | grasp-cylinder, grasp-handle, grasp-spherical-object |
| Manipulation | twist-cap, screw-driver, unscrew, open-jar, close-jar |
| Tool use | hammer, saw, drill, brush |
| Object-contact | rotate-cube, slide-object, push-object |

**Strengths.** Currently registered in the codebase (`src/data/amass/arctic_loader.py`) but never used. Highest hand-fidelity dataset available.

**Weaknesses.** Object-centric motions don't generalise to scene-level navigation; integration mostly buys *hand* fidelity, not new locomotion or gesture categories. Requires the SMPL-X hand channels (we currently zero them out).

---

## 6. KIT-Motion-Language [Plappert et al., 2016]

**What it adds.** ~3k clips, locomotion-focused, with detailed gait descriptions.

| Bucket | Example verbs |
|---|---|
| Gait nuance | limp, hobble, stagger, shuffle, prance, gallop |
| Style modifiers | walks-confidently, runs-fearfully, marches-stiffly |

**Why low ROI.** Mostly variants of locomotion we already cover. ~20 truly new verbs per ~3 weeks of integration effort.

---

## Verb-coverage matrix

Sum of unique new verbs per dataset, *deduplicated against current 28-verb set* and *between datasets* (BABEL + NTU overlap heavily on domestic/social actions; we count each verb only once):

| Cumulative dataset stack | Total verbs |
|---|---:|
| Current (HumanML3D-only) | 28 |
| + BABEL | ~178 |
| + BABEL + NTU | ~230 |
| + BABEL + NTU + Inter-X | ~290 (with two-person) |
| + BABEL + NTU + Inter-X + Motion-X | ~370 |
| + BABEL + NTU + Inter-X + Motion-X + Arctic | ~395 |

Diminishing returns set in after dataset #3. A pragmatic post-thesis roadmap is **BABEL first** (highest ROI, biggest vocabulary jump for least effort), then **Inter-X** (unlocks multi-actor demos, the obvious product extension).

---

## How verb-vocabulary expansion actually works in our system

Adding a verb requires *three* things, all of which are independently testable:

1. **Motion data**: the MotionSSM must be able to render the verb. Achieved by having (caption, motion-clip) pairs containing that verb in the training corpus.
2. **Planner vocabulary**: the canonical verb must appear in `VERB_PARAPHRASES` (`scripts/data/synthesize_planner_data.py`) so the LM emits it.
3. **Tokenizer**: the RVQ must be able to encode the motion's distribution. If the new dataset has *different-looking* motion (e.g. AMASS uses ground-Y while NTU uses ground-Z), normalisation alignment matters.

Step 3 is the silent killer — if the RVQ codebooks were trained only on AMASS-distributed motion, adding NTU's clips without re-fitting normalisation will produce dead codes for the NTU-flavored ranges. We do dataset-specific normalisation in `src/data/motion_dataset.py:trans_stats_by_source` for exactly this reason.

---

## Honest limitations

- Counting "new verbs" by dataset abstracts away the long tail. Many BABEL labels are minor variants ("walk on grass" vs "walk on tile"); de-duplication is conservative.
- Verb counts ≠ defensibility. A handful of *demonstrable* new categories (the two-person ones from Inter-X, the musical ones from Motion-X) make a stronger thesis-extension story than 150 new domestic verbs nobody asks for.
- Some verbs that look novel on paper would *just-work* with the current MotionSSM — CLIP's text embedding gives the model some chance on "saunters" even if HumanML3D never says exactly that, because the embedding is close to "walks". The hard cases are verbs requiring genuinely new motion patterns (climb, swim, lie-flat).

---

## Action items for the journal version

1. Pull BABEL labels (already structured as JSON per AMASS clip; no scraping needed).
2. Extend `VERB_PARAPHRASES` with the BABEL canonical labels + 2-3 paraphrases per (use the existing teacher-LLM-once approach if time-constrained).
3. Retrain MotionSSM on union(HumanML3D, BABEL); single cloud run, ~$20.
4. Refresh held-out generalization eval with BABEL-derived paraphrases.
5. Optional: Inter-X pairing per `doc/planning/13_FUTURE_WORK_MULTI_ACTOR.md`.
