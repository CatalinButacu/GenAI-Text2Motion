# Action Vocabulary and System Capability Surface

> This document describes the *restricted action field* the current system can handle, the *wide compositional field* unlocked by combining those actions, and the *known boundaries* where the system fails. Use this as the reference for what the demo accepts and what to avoid promising.

---

## 1. Canonical verbs (28) — what the MotionSSM was trained to render

The MotionSSM was trained on HumanML3D motion-text pairs; these 28 verbs cover the action space the demo can reliably produce. Each canonical verb maps to a *paraphrase pool* the planner LM learned to recognise.

| Category | Canonical verb | Paraphrases the LM accepts |
|---|---|---|
| **Locomotion** | walks | goes, moves, heads, proceeds, ambles |
| | runs | sprints, dashes, races |
| | jogs | jogs along, runs at a jog |
| | strolls | saunters, wanders, ambles |
| | steps | takes a step |
| | marches | strides |
| | tiptoes | sneaks, moves quietly |
| **Vertical motion** | jumps | leaps, hops up |
| | hops | skips |
| **Combat / striking** | kicks | swings a leg, punts |
| | punches | throws a punch, jabs |
| **Gestures (upper body)** | waves | waves a hand, raises a hand to wave |
| | claps | applauds, claps hands |
| | bows | takes a bow |
| | stretches | reaches out, extends arms |
| | raises arms | lifts arms, holds arms up |
| | crosses arms | folds arms |
| | swings arms | moves arms back and forth |
| **Body posture** | sits down | takes a seat, lowers themselves, is seated |
| | stands up | rises, gets up, stands |
| | lies down | lies flat, lays down |
| **Rotation** | turns | pivots, rotates |
| | spins | twirls, rotates |
| **Object interaction** | picks up an object | grabs an object, lifts an object |
| | drops an object | lets go of an object, puts down an object |
| **Other** | dances | moves to music, does a dance |
| | stops | halts, comes to a stop |
| | pauses | stays still, freezes |

---

## 2. Directional modifiers (6)

Locomotion + rotation verbs accept a direction. The planner LM learns the canonical→paraphrase mapping.

| Canonical | Paraphrases |
|---|---|
| forward | straight ahead, ahead, to the front |
| backward | back, in reverse |
| to the left | left, leftward |
| to the right | right, rightward |
| in a circle | around in a circle, around |
| diagonally | at an angle, off-axis |

---

## 3. Termination grammar (5)

Every action carries an `until` field. The grammar is small and fully enumerable; the parser at `src/modules/agent/conditions.py:parse_condition` rejects anything outside it.

| Variant | Meaning | Example |
|---|---|---|
| `duration(N)` | Stop after N raw frames (20 fps → N/20 seconds) | `duration(60)` = 3 sec |
| `distance(obj) < D` | Avatar within D meters of named object | `distance(tree) < 1.0` |
| `distance(obj) > D` | Avatar past D meters from named object | `distance(wall) > 2.5` |
| `rotated(deg)` | Absolute yaw change ≥ deg since action start | `rotated(90)` |
| `completed` | One-shot fire; for atomic gestures | `completed` |

---

## 4. Scene-object targets (10)

Used only for `distance(obj)` predicates. The demo CLI accepts `--scene-object NAME=x,y,z` overrides.

`tree`, `ball`, `chair`, `wall`, `door`, `rock`, `table`, `fence`, `ladder`, `box`

Adding objects is a runtime concern only — `WorldState.add_scene_object(name, position)` accepts any name; the planner LM's familiarity with new names is the practical limit.

---

## 5. Compositional reach (the "wide field")

The grammar is **sequential composition with per-action termination**. From the 28-verb × 6-direction × 10-target × 5-termination space we get:

| Slot dimension | Choices |
|---|---:|
| Canonical verbs | 28 |
| Directions (where applicable) | 6 |
| Scene-object targets | 10 |
| Termination variants | 5 |
| **Atomic-action combinations** | **~8 400** |

Composition is unbounded in principle; the training distribution covers up to 3-action chains, but `--max-action-latents 50` is the only practical cap.

### Concrete examples that work today

```
"walk forward then sit down"                                          → 2 actions
"march in a circle until tired, then take a bow"                      → 2 actions
"sprint toward the door until close, then turn around, then wave"     → 3 actions
"jump three times, then lie down"                                     → 2 actions
"saunter diagonally until past the fence, then drop into a chair"     → 2 actions
"first stand, then sit, then stand again"                             → 3 actions
```

---

## 6. The restricted boundary — what the system CANNOT do

| Hard limit | Why | Lift effort |
|---|---|---|
| **Two-person interactions** (hug, fight, dance-with-partner) | Inter-X dataset not integrated; current features are body-only single-person | Medium (6-8 weeks; see Ch 9.1) |
| **Hand-finger articulation** (typing, holding pen) | MotionSSM trained on AMASS body channels only | Medium (requires SMPL-X with hands + RVQ retrain on Arctic/Inter-X) |
| **Facial expression** | Not predicted by MotionSSM head | High (new head; new training data) |
| **Verbs outside the 28** (swim, climb, cook, sweep, kneel) | Not in MotionSSM's training distribution | Low-Medium (depends on dataset availability; see verb-investigation doc) |
| **Continuous adverbs** (walk *quickly*, wave *slowly*) | No speed/intensity field in PlannedAction | Low (add `speed: float`) |
| **Conditional branching** ("if X then Y else Z") | Grammar is purely sequential | Low (grammar extension) |
| **Persistent multi-turn memory** ("opposite of last action") | Each instruction processed independently | Medium (conversation buffer in planner prompt) |

---

## 7. Why the boundaries are set here

Three engineering choices fixed the boundary:

1. **HumanML3D + AMASS training corpus.** This is the *single-person* benchmark of the T2M field. It does not contain two-person clips or detailed object manipulation, so the MotionSSM has no signal for those.
2. **168-d SMPL-X feature vector** (`src/data/amass/smplx_pack.py`). The first 6 channels are root pose + translation, the next 63 are body pose. Hand pose channels (69-159) are present in the format but not exercised by the current model because AMASS sets them to default; Inter-X / Arctic populate them but feed a different RVQ codebook distribution.
3. **The planner's 28-verb canonical vocabulary** (`scripts/data/synthesize_planner_data.py:VERB_PARAPHRASES`). Adding a verb here is a 1-line change but the MotionSSM must already be able to render it, otherwise the LM will route the user's instruction to a verb that produces garbage motion.

The action vocabulary is *not* a hidden limit baked into Mamba or RVQ — it is the intersection of *what we trained* and *what the planner knows the trained model can do*. Both can be expanded independently.

---

## 8. Related reading

- The **dataset-expansion roadmap** documenting what new verbs each candidate dataset would unlock: `doc/DATASET_EXPANSION_ROADMAP.md` (this repo).
- The **variable-length training analysis** explaining why short clips waste compute and how bucketed sampling fixes it: `doc/VARIABLE_LENGTH_TRAINING.md` (this repo).
