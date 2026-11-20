# A25 processing plan: from raw archives to training data

Status: proposed. Part of A25 (`A25_data_expansion.md`). Every step writes counts to a log, so the thesis data chapter can be built from the logs.

## 1. Extract and keep only what is used (done by `scripts/data/extract_inbox.py`)
- One archive at a time. Extract, check the exit code and file count, then delete the original. Stop before running out of space.
- Skip preview renders (`*renders*`) and `mirror` duplicates. Skip incomplete downloads.
- Log per archive in `data/raw/EXTRACT_LOG.json`.

## 2. Uniform contract (per source, one converter each)
- Body: SMPL-X joints to HumanML3D-263 at 60 fps. Sources at a lower rate are cubic-spline resampled. Sources at 120 fps are downsampled.
- Hands: 15 joints per hand in 6D rotation, from GRAB and Motion-X++ (AMASS hands are almost still in most sources).
- Face: jaw and expression values, from Motion-X++.
- Units and joint order are checked by a test per converter (gravity check on flight frames, foot height on stance frames).

## 3. Richer supervision (more text per motion, not just more motion)
- HumanML3D has several captions per clip: use all of them, each as a training pair.
- BABEL gives per-segment action labels: use them to split long clips into action segments, each with its own text.
- Motion-X and Motion-X++ text: check for duplicates and near-duplicates against HumanML3D (leakage).

## 4. Augmentation (applied to training only)
- Mirror left and right (flip the text words left/right).
- Time-scale between 0.8x and 1.2x by resampling.
- Random yaw rotation of the root, since the model must not depend on facing direction.
- Each augmentation is a switch in the config, so experiments can turn them on and off.

## 5. Data science checks (before any training)
- Per-source report: hours, clip count, duration histogram, action histogram, pose variance per joint.
- Deduplication: identical or near-identical clips inside and across sources.
- Leakage check: no subject or clip appears in both train and test.
- Quality scores per clip: Newton check, foot sliding, jerk, ground penetration. Calibrate the thresholds with the owner's blind ratings.
- Balance: report how the actions are distributed. Decide whether to reweight rare actions, and test that decision as an experiment.

## 6. Splits and freeze
- Split by subject, not by clip, so the same person never appears in train and test.
- Mean and Std from the training split only.
- Freeze the data version with a hash. Every model records the version it used.

## 7. Experiments that decide the choices
- Quality threshold: train on the top 100%, 75% and 50% of clips. Compare on the same test set.
- Augmentation: each switch on and off, same seed, same budget.
- Action reweighting: on and off, measured on rare actions as well as overall FID.
- Every experiment is a row in the ledger, with the data version and config hash.
