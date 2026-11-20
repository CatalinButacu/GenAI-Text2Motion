# A25 (proposed): Data expansion and documented data process

Status: proposed, waiting for owner approval. Not yet in `epics.csv`.

## Question
Can we gather every usable motion, hand, face and scene dataset that fits the project contract, convert it with documented and tested transforms, and record each step with its licence, counts and checksums, so the data chapter of the thesis can be written from this record?

## Why
- Our checkpoints are lost, so the data pipeline is the only part we can rebuild exactly. Its record must be complete.
- The frozen-block plan (body, then hands and face, then environment) needs more motion than HumanML3D alone.
- The dissertation plan needs a data-licensing subsection and a Representation/Data section (`DISSERTATION_PLAN.md`, sec 3).

## Contract (fixed before collecting)
- Body: HumanML3D-263 features at 60 fps. Sources at a lower rate are cubic-spline resampled. Rendering at 120 fps.
- Hands and face: SMPL-X finger rotations and jaw/expression values, separate from the body (A19).
- Scenes: floor and contact geometry for the physics block (block 5).
- Stats (Mean/Std) come from the training split only. Splits are source-aware.

## Work packages
| WP | Output | Finish criterion |
|---|---|---|
| W0 Licence and access audit | `data_ledger.csv`: one row per source with licence, commercial use allowed?, access (open or login), format, size, citation, version | Every row has a licence quoted from the source's own page; 0 unverified rows |
| W1 Acquisition | Raw files under `data/raw/<source>/`, checksums in `SHA256SUMS` | Checksums logged; gated sources marked as waiting for the owner |
| W2 Conversion to contract | One converter per source in `scripts/dataset/`; a test per converter | Frame rate, units and joint order checked against the body contract; tests pass |
| W3 Cleaning | Newton check, T5g and duplicate filters applied per source, with counts before and after | Every removed clip has a recorded reason |
| W4 Splits and leakage | Source-aware splits; leakage check by source and by subject | 0 overlaps between train and val/test by subject |
| W5 Freeze and document | Data version `v2` hash; a process record for every step (inputs, parameters, counts, script name) | The record is complete enough to write the thesis data chapter without rerunning anything |

## Candidate sources (to be checked in W0, not yet verified)
- Body and text: HumanML3D (have), KIT-ML, BABEL (have), Motion-X and Motion-X++ (have, official files), AIST++ (dance), HAA500.
- Mocap: AMASS (gated, sub-folders empty), CMU and others inside AMASS, 3DPW.
- Hands and object interaction: GRAB (have partly), ARCTIC, OakInk, DexYCB, InterHand2.6M (check licence).
- Face: MEAD, VOCASET (FLAME, gated).
- Scenes and contact for the environment block: PROX, POSA.
- Physics: MuJoCo humanoid model (open source). No data needed; it is a simulator.

Several of these are non-commercial or gated. Every licence goes into the ledger, and the thesis states the terms.

## Two issues to settle before W2
1. **MoMask baseline frame rate.** MoMask's released RVQ was trained on HumanML3D at 20 fps with its own Mean/Std. Our contract is 60 fps. Comparing it with our FSQ needs either its own preprocessing or the same clip set evaluated under each model's contract. The comparison table must state which.
2. **Converted stats.** The converted Mean differs from the original by up to 0.02 (see the restore log). W5 must record which stats each model uses.

## Budget
- GPU: 0 hours. This epic is CPU and network only, so it runs while the GPU is free.
- Copilot calls: 0 by default. Use the orchestrator's own tools. Ask the owner before a council.
- Time box: 2 weeks of orchestrator work.

## Decisions for the owner
1. Approve the epic and the contract above.
2. Gated sources (SMPL-X, AMASS, FLAME/VOCASET, and any others found in W0): owner downloads with their own login; the orchestrator checks checksums and converts.
3. Accept that the MoMask comparison uses its own 20 fps preprocessing, documented in the table, or ask for a 60 fps re-evaluation.
