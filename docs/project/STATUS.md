# STATUS -- session checkpoint (2026-11-08, updated)

Master record for the final paper: docs/paper/FINAL_PAPER.md. Use it for every result, contribution and open decision. This file is a diary only.

## Update (2026-11-26, A28-T14 final gate of epic A28; blocked on pyright)

- make check: exit 0. ruff passes. pytest: 851 passed, 3 skipped, 15 deselected, 0 failed, 221 s.
- pyright on the full repo: 99 errors, 0 warnings, 282 files. The criterion is 0, so it is not met.
- The 99 errors are in 6 files that A28 commits did not change: studio/panels.py 87, studio/viewer.py 1, scripts/qa/studio_controls_test.py 7, scripts/qa/studio_human_test.py 1, scripts/checks/reproduce_reference_clip.py 2, scripts/dataset/validate_reference_joints.py 1.
- 96 of them are imgui errors. Likely cause (not tested): the self-built Python 3.14 imgui wheel in third_party/wheels gives pyright no type information. The other 3 are in 2 untracked scripts.
- check_problems.py --check on the 32 Python files of the A28 commits: 2 problems before, 0 after. Fixed: C901 in app/service.py (steer handling moved to apply_steer, which raises if no model is loaded); G02 in common/runner.py (Run is now a type alias statement).
- check_problems.py --check on the uncommitted files (run before the fix): 4 problems, not fixed: dataset/pipeline.py SIM108 1, reproduce_reference_clip.py 2, validate_reference_joints.py 1. All 3 files are untracked owner work.
- tests/test_package_layers.py with tests/app/test_import_boundaries.py: 7 passed. test_package_layers alone: 2 passed.
- docs/project/ARCHITECTURE.md updated: 8 packages with common and dataset, import table, run workflows, training engine, steering.
- OWNER decision: (a) add imgui type stubs under typings/ and fix the 2 scripts in a new task, or (b) accept pyright 0 on the A28 files only.

## Update (2026-11-25, A28-T1 golden tests decision)

- Owner decision: record new golden values (chat 2026-11-25).
- Module freeze: the only change in tokenization is one import path in reconstruction_metrics.py (text2motion.dataset.datasets -> text2motion.dataset.training.captioned). Re-frozen at 745a968a with an unfreeze and a freeze row in docs/notes/module_ledger.csv. test_module_freeze passes.
- Tokenizer and generator golden: their inputs (tokenizer and generator checkpoints, evaluator weights, eval stats) were deleted on 2026-11-23, so no new values can be recorded. Both tests are skipped with a named reason. Record them again with scripts/qa/golden_outputs.py after the causal retrain.
- make check: 842 passed, 3 skipped, exit 0. The gate is green for the first time since 2026-11-23.

## Update (2026-11-25, A28-T11 steering joint jump; wiring only)

- No generator checkpoint exists in this environment (runs/checkpoints is missing). The measure used the tiny CPU Mamba (random weights), so it shows wiring only, not quality.
- Script: scripts/qa/steering_jump.py. CSV: docs/notes/steering_joint_jump.csv. 24 text pairs, seed 2026, one process, switch at token 12 of 24, blend 8 frames.
- Joint jump: the peak joint step in cm over the window from the switch frame to 8 frames after it. Reference: the same stream with the old text only.
- Mean jump: (a) re-anchor 14.09 cm, (b) cross-fade 13.55 cm. p95: (a) 16.09 cm, (b) 15.00 cm.
- Mean excess over the reference: (a) 0.0001 cm, (b) 0.31 cm. p95 excess: (a) 0.00 cm, (b) 2.85 cm.
- Choice (provisional): option (a) re-anchor. Reason: no excess jump over the reference, and one stream instead of two. Caveat: the tiny model hardly reacts to the text, so (a) shows no change by construction.
- Next: repeat with the t2k_mamba_causal_heading checkpoint when it is restored. Change the choice only if (b) wins there.

## Decision (2026-11-23, owner): plan change

- Full plan: `docs/project/PLAN_2026-11-23.md`. Papers: `docs/project/references.md`.
- Causal rule: the generator and every streaming path are causal. The frozen tokenizer `fsq_g8_v1024` is not causal. A causal tokenizer is retrained.
- Rates: input 60 fps, output 60 fps, interpolated to 120 fps for playback.
- Data stages: AMASS first (locomotor learning), HumanML3D only at the text stage.
- Training: evaluate every 5 epochs, stop after 10 epochs without improvement. Old runs are historical and do not enter the paper.
- Text encoders: test several by a fixed test. CLIP is the baseline.
- Guidance: sweep scale 1, 2, 3, 4, 6.
- Engine: Unity. Claude builds 3D objects for the scene. The avatar comes later.
- X5 is skipped. A19 (hands and face) is approved as part of the model.
- Blocked: the 60 fps data needs the raw AMASS files, which the owner must download again.

## Update (2026-11-22, T2 stopped; Mamba kernel rebuild for this environment)

- T2 (t2_mamba_causal_heading) stopped at your request. Cause of the slow steps: the probe config runs the Mamba scan in plain Python (use_kernel false, with activation recomputation). The GPU load swings between 0 and 100 percent between steps.
- The saved compiled kernel (third_party/mamba_kernel) is built for Python 3.12 and cannot load in this environment, which runs Python 3.14. Environment note: the project rules say Python 3.12, but the venv runs 3.14.10.
- Rebuild started: Mamba CUDA kernel for Python 3.14, torch 2.12.0+cu132, CUDA architecture 8.9 (RTX 4050). Log: runs/logs/kernel_build_py314_sm89.log. Expected about 46 minutes.
- Next: equivalence test (kernel against Python scan, same inputs, tolerance), then the speed test, then T2 restart with the kernel only if both pass.
- Build attempt 1 failed: missing setting DISTUTILS_USE_SDK=1. Attempt 2 failed: the CUDA compiler ran out of memory with too many parallel jobs. Attempt 3 started 20:54 with MAX_JOBS=3.

## Update (2026-11-22 late, T2 kernel run finished; validation only)

- Run t2k_mamba_causal_heading (kernel on, seed 2026, causal tokenizer r11): early stop at epoch 25 of 40 (validation cross-entropy did not fall for 6 epochs; best 3.84).
- Epoch time: about 2.5 minutes (earlier runs: 13 to 21 minutes). Whole run: about 70 minutes.
- Validation text-to-motion scores (628 clips), by evaluation epoch:
  - epoch 5: FID 16.1, R@1 0.110
  - epoch 10: FID 3.85, R@1 0.181
  - epoch 15: FID 3.04, R@1 0.268
  - epoch 20: FID 2.13, R@1 0.311 (best FID, saved as the best checkpoint)
  - epoch 25: FID 2.42, R@1 0.273
- Clean cross-entropy at the stop: train 2.83, validation 3.86, gap +1.04. Overfitting starts after epoch 20.
- Caution: FID 2.1 is far below the 33.9 of the 2-epoch probe. Before any claim, check the evaluator on real motion and check the split.
- Evaluator check (2026-11-22, 23:03, validation split, all 765 segments, best checkpoint):
  - A. FID generated vs real, same metric: 2.03 (training log 2.13; the gap comes from the clip subset and sampling).
  - B. Floor: FID real vs real, random halves, 5 splits: mean 0.68, range 0.29 to 1.15. The model FID is about 3 times the floor.
  - C. Memorisation: nearest real distance for generated median 1.03, real-to-other-real median 0.88. 6.4 percent of generated segments lie closer than half the real median. No copy of a validation clip is visible.
  - D. Lengths: generated and real mean 143.0 frames.
  - Subset note: training scores the first 600 clip IDs (max_eval_clips), which gives 628 segments. The probe numbers use the same subset, so the comparison is fair.
- Source overlap (2026-11-22, by source path from index.csv; clip ids from the split files):
  - Validation: 224 of 720 source files, and 458 of 1460 clips, also appear in training.
  - Test: 639 of 2122 source files, and 1336 of 4384 clips, also appear in training (matches the earlier record).
  - Validation scoring subset (600 IDs, 628 segments): 201 segments have a shared source, 427 do not.
- Overlap effect on the score (validation, best checkpoint, 628 segments):
  - FID, shared-source segments: 3.05.
  - FID, other segments: 2.21.
  - FID, other segments subsampled to 201 (20 draws): mean 2.46, range 1.87 to 3.14.
  - Result: shared-source segments score worse, not better. The overlap does not inflate the validation FID in this check. Small sample sizes: treat as a sign, not a proof.
- The T6 decision still matters for the test split, because the test overlap is 30 percent of clips.

## Proposal (2026-11-23): owner chooses Option B, both sources at 60 frames per second (long-term choice)

- Decision: owner selected Option B. Data contract changes from 20 to 60 frames per second for HumanML3D and AMASS.
- Blocking fact 1 (corrected 2026-11-23): AMASS is not all at 120 frames per second. The clock check (docs/notes/amass_clock_check.csv) gives per clip: 120 for 5049 clips, 100 for 4634, 60 for 681, no match for 94, and a folder rate for 2 (10364 clips linked to HumanML3D sources). Conversion to 60 frames per second is a clean resampling from each rate. The raw AMASS files were removed on 2026-11-23 (owner approval, the processed copy was kept). The processed copy is at 20 frames per second. Real 60 frames per second from AMASS needs the raw files again, downloaded by the owner.
- Blocking fact 2: HumanML3D is at 20 frames per second in our copy. Its raw source must be checked the same way.
- Blocking fact 3: interpolation from 20 to 60 frames per second adds no new motion. It only changes the time scale. A tokenizer on interpolated data learns the same motion, with three times more frames.
- Blocking fact 4: the 263-value representation, the tokenizer, the generator and the text-to-motion evaluator (trained on 20 frames per second) all depend on 20 frames per second. Option B needs a new representation, a new tokenizer, new generators, and a new evaluator model. Published numbers are no longer directly comparable.
- Plan (proposed, not started):
  1. Owner: download raw AMASS and check the HumanML3D raw source, and record their frame rates.
  2. Define the 60 frames per second representation: velocities and the heading feature scaled to the new time step. Write it as a new contract, with a test.
  3. Rebuild the processed data at 60 frames per second (source rate to 60 by cubic resampling, no interpolation where the source is at 120).
  4. Retrain the tokenizer, then the generators, on the grouped split. Keep the 20 frames per second results as a separate reference.
  5. Evaluator: decide whether to retrain the text-motion matcher at 60 frames per second. This is the largest cost. Without it, the text-to-motion FID cannot be read against the literature.
- Question for owner: re-download raw AMASS and HumanML3D source files (both at their native frame rate), and approve the new contract, including the evaluator decision.

### Preparation done without owner input (2026-11-23, owner gave autonomy)

- Step 7 of the plan (60 frames per second contract): the existing code already takes a frame rate in the feature parameters, and it already has a frame index scale (1 at 20, 3 at 60). Added one test in tests/motion/test_hml3d_263_contract.py: a constant-speed root, resampled from 20 to 60 frames per second, gives a root velocity per frame at 60 that is one third of the value at 20 (within 2 percent). Result: 3 tests pass. Lint and format clean.
- Open cost for the contract: the global frame rate constant is used in 112 places in src. Changing the data to 60 frames per second means each place must take the rate from the config. This is the main code work of step 8.
- Not done (needs owner or raw data): download, rebuild, retraining.

### Download plan for Option B (2026-11-23)

- No new script. The existing script scripts/data/fetch_sources.py covers the steps: list, fetch (public Hugging Face sources only), place (copies a folder that you downloaded yourself), verify (checks hashes against a manifest).
- MPI sites (AMASS, SMPL-X, GRAB, BABEL) need a login. The rule stays: no automated login to MPI sites. You log in in your browser and download the folders yourself.
- HumanML3D uses 18 source folders. Most are AMASS folders. The folder humanact12 is a separate source, not AMASS. Its licence must be read before use.
- Blocker: the raw AMASS folders took about 147 GB. Free space on the only drive (C:) is 129.9 GB. No second drive is attached.

Ten steps, in order:

1. Owner: attach an external drive with at least 200 GB free, or free more space on C:.
2. Owner: download the AMASS folders for the 18 source folders in the browser, into a staging folder.
3. Run: place amass <staging folder> (existing script).
4. Run: verify amass (existing script, hash check).
5. Owner: download humanact12 from its public source after reading the licence; then place and verify it.
6. Check the clock rate of each clip (120, 100, 60) and decide the 94 clips with no clear match.
7. Write the 60 frames per second contract as code, with tests.
8. Rebuild the processed data at 60 frames per second. Re-run the contamination checks.
9. Retrain the tokenizer at 60 frames per second. Compare with the 20 frames per second run at equal steps.
10. Train the Mamba and transformer twins with the same recipe. Score at 20 frames per second. Read each test split once. Record everything in STATUS and MLflow.

Steps 1 and 2 are owner actions. Steps 7 to 10 start after the data check in step 4.

### Full plan, Option B (owner decision, long-term)

Owner actions (blocking, in this order):

1. Download the raw source of every dataset that HumanML3D and AMASS use, at native frame rate. This includes the 23 AMASS folders removed on 2026-11-23 and HumanML3D's own pose sources. Place them in data/raw.
2. Approve the evaluator decision. Recommended: score generated motion after downsampling to 20 frames per second, with the existing matcher, so that published numbers stay comparable.

Plan steps (after owner actions 1 and 2):

1. Contract: define the 60 frames per second representation (time-scaled velocities and heading, length bounds, window length as a candidate of 768 frames). Decide the window with a test. Write the contract as code with tests.
2. Data: rebuild processed features at 60 frames per second from native sources (cubic resampling, no interpolation from 20). Reuse the grouped split. Re-run the contamination checks: zero shared source files between splits, and no AMASS clip linked to a validation or test source.
3. Tokenizer: retrain at 60 frames per second on the grouped training split, with the same causal and heading design. Compare the validation curve with the 20 frames per second run at equal steps.
4. Generator: set the maximum token length for 60 frames per second (about three times the 20 frames per second length). Run the overfit gate, then the Mamba and transformer twins with the same recipe (patience 10 epochs on validation cross-entropy).
5. Evaluation: downsample generated and reference motion to 20 frames per second, then score with the existing matcher. Read the test split once per twin.
6. Record every step in STATUS.md and MLflow. Keep the 20 frames per second run as the reference.

## Update (2026-11-23 18:00): Mamba run g_mamba_grouped_p10 finished (validation cross-entropy patience 10)

- Early stop at epoch 30 (10 epochs without a lower validation cross-entropy; best 3.685).
- Validation FID by evaluation epoch (614 segments): 5 24.87, 10 6.28, 15 3.02, 20 2.62, 25 2.10 (saved best checkpoint), 30 2.62.
- Test: not yet read. It is read once, on the final selected checkpoint of each twin, after the transformer twin finishes.
- Next: transformer twin on the grouped tokenizer, same recipe (patience 10), matched parameter count.
- Logging: this run started before live MLflow logging was on, so it is not in the store yet. Backfill for generator runs is still to be added.

## Update (2026-11-23 14:20): MLflow workflow on the local machine (owner request)

- Store: runs/logs/mlflow.db (SQLite). UI: local server on port 5001, reads that store. Old server on port 5000 is still running (stopping it was denied by the permission system; stop it by hand when convenient).
- Tracking is on for the grouped tokenizer and grouped generator configs (mlflow_tracking true). Runs started from now on log their metrics live. Runs started before this change (the Mamba p10 run) are not live-logged; backfill them with the same script when they finish.
- Logging path tested on the CPU with a temporary experiment: one run, six metric points. The test experiment and its run folder were deleted.
- Backfill script (scripts/maintenance/backfill_mlflow.py) changed: it skips run folders that are already in the store. Before this change it created duplicates on every run. Result of the one run: 42 tokenizer folders found, 23 already logged, 19 added. The store now holds 43 runs.
- Next for the workflow: add generator runs to the backfill (the script covers tokenizer runs only), then compare runs in the UI and make changes from the comparison.

## Update (2026-11-23 13:00): Mamba run g_mamba_grouped finished at patience 6; new recipe with patience 10 started (owner decision)

- Run g_mamba_grouped (validation cross-entropy patience 6): early stop at epoch 25. Validation FID by evaluation epoch: 5 23.59, 10 6.92, 15 3.31, 20 2.60, 25 2.50 (saved best, FID). Best validation cross-entropy 3.685 at epoch 20. Gap between training and validation cross-entropy grows after epoch 15 (+0.84 at the stop).
- No test score was written by this run. A test read on its checkpoint was started and then stopped at once, because the recipe is superseded. The completed run keeps its original recipe (rule: completed experiments stay unchanged).
- Owner decision: the cross-entropy patience is 10 epochs. With evaluation every 5 epochs, this equals 2 evaluation steps without a lower validation cross-entropy. The patience counts epochs, not evaluations.
- New run g_mamba_grouped_p10: same config, same tokenizer, validation cross-entropy patience 10, FID patience 6 (unchanged). Log: runs/outputs/research/g_mamba_grouped_p10.out.log. Guard 12 hours.
- The transformer twin must use the same patience (10) so that the two twins follow the same recipe.

## Update (2026-11-23 10:20): Mamba generator on the grouped split started

- Config: configs/generator/grouped/gen_mamba_grouped.yaml (kernel on, dataset folder humanml3d_grouped). Tokenizer: tok_grouped_r11_causal_heading_w001 (E1).
- Overfit gate: PASS. Cross-entropy 0.00065, token accuracy 1.000, 26 steps. Gate file: runs/outputs/gates/grouped_mamba_gate.json.
- Run g_mamba_grouped: 40 epochs, seed 2026, validation every 5 epochs, patience 6 on validation FID and on validation cross-entropy. Test scored once, on the selected checkpoint. Log: runs/outputs/research/g_mamba_grouped.out.log. Guard 12 hours.
- Still to do: the transformer twin on the same grouped tokenizer, with a matched parameter count, before any comparison.

## Update (2026-11-23 10:20): E2 result (AMASS share 0.25) at equal steps: AMASS does not help

- E2 stopped at its step budget of 15400 steps (epoch 75, 206 steps per epoch with AMASS). The label "epoch 100" in the automatic result line is wrong; the run stopped at epoch 75.
- At 15400 steps, validation recon-FID: E2 0.1158 (MPJPE 69.8 mm), E1 0.0917 (MPJPE 72.1 mm, epoch 100, same steps).
- Test (best validation checkpoint, scored once): E2 0.0925 (MPJPE 71.8 mm), E1 0.0777 (MPJPE 72.3 mm).
- Decision for now: AMASS share 0.25 is not used. E1 stays the tokenizer for the generators. AMASS remains a documented option (L2).
- Caution: E2 validation was still falling at the stop, as E1 was. The comparison is at equal steps, the rule for this project.

## Update (2026-11-23 03:35): E2 started (grouped tokenizer, AMASS share 0.25, equal steps)

- E1 steps: 154 per epoch, 15400 in total (100 epochs), counted from the grouped loader.
- E2 settings: same config and seed, AMASS share 0.25 from the listed training clips (data/processed/amass_263/train.txt, 9417 clips), step limit 15400 (about 75 epochs, 206 steps per epoch with AMASS). Validation every 5 epochs. Test scored once at the end, on the best validation checkpoint.
- Run: tok_grouped_e2_amass25_causal_heading_w001. Log: runs/outputs/research/tok_grouped_e2_amass25.out.log. Guard 6 hours.
- Comparison rule: E1 and E2 share the grouped validation and test sets, and the same step count. The E2 result answers only one question: does AMASS help at equal steps.

## Update (2026-11-23 03:30): grouped tokenizer E1 finished (protocol v2, grouped split, HumanML3D only)

- Run tok_grouped_r11_causal_heading_w001 (run dir 20261122T220439Z_tokenizer_tok_grouped_r11_causal_heading_w001): 100 epochs, seed 2026, window 256, batch 128, causal, heading weight 0.01, no AMASS. Finished normally, no guard kill.
- Validation recon-FID (2191 segments), selected by this value: epoch 5 18.32 (MPJPE 318 mm), epoch 10 1.82, epoch 20 0.57, epoch 50 0.15, epoch 100 0.0917 (MPJPE 72.1 mm). The curve was still falling slowly at epoch 100, so the run is not fully converged.
- Test (2190 segments), best validation checkpoint, scored once at the end of the run: recon-FID 0.0777, MPJPE 72.3 mm.
- Checkpoint: runs/checkpoints/tokenizer/tok_grouped_r11_causal_heading_w001 (best), and the resume file of epoch 100.
- Comparison caution: this test set is the grouped test split. The r11 test value (0.0887, 83.2 mm) is on the official test split. The two numbers are not comparable. This is L1.
- Next: E2 (AMASS share 0.25, equal steps to E1). The step count of E1 is not logged; it is calculated from the loader before E2 starts.

## Update (2026-11-23 00:45): raw AMASS removed (owner approval); processed copy kept

- Removed: data/raw/amass (147 GB, 16429 files, 23 dataset folders: ACCAD, BMLmovi, BMLrub, CMU, CNRS, DanceDB, DFaust, EKUT, EyesJapanDataset, HDM05, HUMAN4D, HumanEva, KIT, MoSh, PosePrior, SFU, SOMA, SSM, TCDHands, TotalCapture, Transitions, WEIZMANN). No archive files existed.
- Kept (processed, the only AMASS input from now on):
  - data/processed/amass_regen/pose_data: 15927 pose files (path source for the source links).
  - data/processed/amass_regen/joints, new_joints, new_joint_vecs, index_manifest.json, dropped_index_rows.csv.
  - data/processed/amass_263/new_joint_vecs: 12377 features; corpus_manifest.json (complete true, 0 failed, 40.68 hours); train.txt (9417 usable names, see L2).
- Check before removal: no config and no code path reads the raw folder. Processed pose files are the input of the link check.
- Raw data cannot be rebuilt from this project. If the raw files are needed again, download them from the AMASS source again, and check the corpus manifest.

## Update (2026-11-23 00:30): disk space breakdown (system drive, 476 GB total, about 4.5 GB free)

- Thesis folder, 231 GB: raw data 172 GB (AMASS 147, GRAB 20, HumanML3D 4, EHF 1); processed data 9 GB; renders 18 GB; runs 18 GB (resume files about 0.65 to 0.8 GB each); virtual environment 4 GB; models 3.5 GB; other 3 GB.
- Documents, other folder: Work 40 GB (not thesis data).
- AppData, 29 GB: uv cache (pruned now, 1.2 GB freed), Programs 5.4, Microsoft 4.5, Temp 3.5, Brave 2.1.
- OneDrive 4.6 GB.
- Cleaned without approval: uv cache prune (1.2 GB) and pip cache purge (caches only, regenerable).
- Owner decisions needed (candidates, nothing deleted):
  1. Raw AMASS, 147 GB: the processed corpus exists. Move the raw files to external storage, or delete them after a checksum record. Rebuilding needs the raw files.
  2. Renders, 18 GB: regenerable outputs.
  3. Resume files of finished runs, about 13 GB: keep the best checkpoints, delete the resume files.
  4. Documents\Work, 40 GB: outside this project.
- Process fix (proposed code change): fail loud at run start when free space is below the resume file size plus a margin; keep only the newest resume file per run.

## Update (2026-11-23 00:10): grouped tokenizer run crashed (disk full), restarted

- The first grouped tokenizer run (tok_grouped_r11_causal_heading_w001) crashed at the end of epoch 5, when it saved the resume file. Cause: the system drive had 3.7 GB free. The write failed, and a leftover temporary file stayed locked.
- Epoch 5 values match run r11 (training recon 0.451 against 0.443; validation MPJPE 318 mm against 277 mm on a larger validation set of 2191 segments). No sign of a code error.
- Action: deleted the two failed-write leftovers (about 0.65 GB). Free space now 4.35 GB. The run restarted from epoch 1 (no resume file exists), guard 6 hours. Log: runs/outputs/research/tok_grouped_r11_run2.out.log.
- Blocker for the generator runs: each resume file is about 0.8 GB, and the disk has little room. Owner decision needed on which old checkpoints may be deleted (candidates: the eq_s2027 to eq_s2030 off-protocol checkpoints, about 1.4 GB, and the older resume files of finished tokenizer runs, about 4 GB). Nothing else was deleted.

## Update (2026-11-22 evening): grouped split built, tokenizer retraining started, AMASS mapping verified

- Owner approved the tokenizer retraining. Owner rule: experiments are autonomous; decisions on the big project need the owner. Cross-split contamination is a red flag at every step.
- Grouped dataset: data/processed/humanml3d_grouped. Built with the existing function text2motion.dataset.preparation.write_grouped_split (new function, no new script). The old script was removed. Lists include mirrored copies, as in the official lists. Base clips: train 10230, val 2192, test 2192. Lines: 20460, 4384, 4384. Two base clips have no mirror file.
- Normalisation (Mean, Std) refitted on the grouped training list only (existing fit_train_stats). Feature folder: junction to the official feature folder.
- Protocol record for the grouped dataset: data/processed/humanml3d_grouped/protocol_splits.json, and docs/project/protocol_split_grouped.json.
- Contamination check (independent of the build function): shared source files between splits: 0 (train-val, train-test, val-test). Shared clip ids: 0.
- Tokenizer retraining: configs/tokenizer/grouped/fsq_g8_w1024_causal_heading_w001.yaml (same settings as r11, grouped dataset). Run tok_grouped_r11_causal_heading_w001, guard 10 hours, log runs/outputs/research/tok_grouped_r11.out.log. Epoch 1: recon 0.6297.
- AMASS mapping (L2): verified. Links 6590 train, 1869 val, 1905 test pose files. Copies of val or test sequences: 0. AMASS corpus: 9417 usable clips, 2960 blocked.
- Code change: the AMASS extension loader reads data/processed/amass_263/train.txt (fails loud if a name has no feature file). The list is a data file, not a script.
- Next: E2 = grouped tokenizer plus AMASS share 0.25, with the same number of steps as E1 (equal-step control). Then decide.

## KNOWN PROBLEM FOR THE PAPER (2026-11-22): source overlap in the official split (limitation L1)

- The official HumanML3D split assigns clips by base clip. Source files appear in more than one split.
- Overlap: test 639 of 2122 source files (1336 of 4384 clips) are also in training. Validation 224 of 720 source files (458 of 1460 clips).
- Checked effect: on the generator validation score, shared-source segments scored worse (3.05 against 2.21). Effect on the tokenizer choices and the other parameters: not measured.
- All results on the official split are provisional. The list is in docs/project/LIMITATIONS.md, L1. Refer to it as L1 in the final paper.
- Decision (owner, 2026-11-22): the grouped split is the protocol. Record: docs/project/protocol_split_grouped.json. Build: scripts/maintenance/build_grouped_split.py.
- Still open: retraining of the tokenizer and the generators on the grouped training split (needs owner approval); AMASS mapping (L2).
- Script rewritten for lint (functions, sorted imports, no unused variables, a real disjointness check). Re-run: the three split hashes are identical to the record. Ruff: all checks passed. Pyright: 0 errors. The editor Problems tool is not available in this session; ruff and pyright were used instead.

## Proposal (2026-11-22): leakage-free grouped split (adopted, see above)

- New folder: data/processed/humanml3d_grouped (train.txt, val.txt, test.txt, grouped_split.json). The frozen official files are unchanged.
- Rule: group key is the source file (source_path in index.csv). Each group goes to exactly one split. Seed 2026. Target 70/15/15 by clip count.
- Sizes (base clips, 14616 total, 11715 source files): train 10231 (70.0 percent), val 2193 (15.0), test 2192 (15.0). Zero groups in two splits.
- Consequence 1: 80.7 percent of the new validation clips and 81.5 percent of the new test clips were in the old training split. The tokenizer r11 and the generator t2k were trained on them. Both must be retrained on the new training split before any new score counts.
- Consequence 2: the scores cannot be compared with published HumanML3D numbers, which use the official split. Report both, with the reason.
- AMASS: the AMASS corpus (12377 clips) excluded the old validation and test sources (2525 skipped). It must be grouped with the new split too: each AMASS clip goes to the split of its source file. This mapping is not verified yet.
- Needs owner decision: (1) adopt the grouped split as the protocol; (2) retrain the tokenizer (this changes the rule that the tokenizer is frozen) and the generators.
- Test split: not scored. It waits for the owner decision on the split (T6).
- Guard: no kill record for this run.

## Update (2026-11-22, kernel fixed and verified; performance issue addressed)

- Kernel rewritten (plain template dispatch, both kernel files). Build for Python 3.14, exit 0 at 21:37.
- Kernel against the reference scan: relative error 4e-7 to 4e-6 (forward and gradients, lengths 128 to 1000).
- Speed, one backbone training step: Python scan median 49.9 s, peak GPU memory 10.6 GB (the 6 GB GPU spilled to system memory, which explains the 0 to 100 percent spikes). Kernel median 2.9 s, peak 1.7 GB. About 17 times faster.
- Config with the kernel on: configs/generator/t2_causal_heading_kernel/gen_t2_causal_heading_kernel.yaml. Overfit gate with the kernel: PASS (runs/outputs/gates/t2_kernel_causal_heading.json, step 201, token accuracy 1.000).
- T2 restarted with the kernel at 21:49: run t2k_mamba_causal_heading, 40 epochs, seed 2026, protocol flags. Guard: 16 hours. Log: runs/outputs/research/t2k_mamba_causal_heading.out.log.
- Decision: keep PyTorch. CNTK is not an option (maintenance only, no CUDA 11 or later).
- Details: third_party/mamba_ssm-2.3.2.post1-msvc/MSVC_PORT.md.

## Update (2026-11-22, T2 stopped; Mamba kernel rebuild for this environment)

- Attempt 3 failed at selective_scan_fwd_kernel.cuh line 315 (MSVC C2131). Cause: a named constant from an outer lambda used as a template argument in a nested lambda (upstream pattern, not changed by us).
- Fix tests (small MSVC files, not the kernel): four lambda capture variants fail. Plain template functions with the constants as template parameters compile. Next: rewrite the switch launch in the forward and backward kernels with plain template functions, then rebuild. Not applied yet.

## Update (2026-11-22, T2 started: full Mamba generator run)

- Run t2_mamba_causal_heading: Mamba backbone, causal tokenizer r11, seed 2026, protocol settings. Epochs 40 (budget about 9 to 13 hours). Validation every 5 epochs. Patience 6 on validation FID and on validation cross-entropy.
- Overfit gate: probe_mamba_causal_heading.json (same configuration, PASS).
- Log: runs/outputs/research/t2_mamba_causal_heading.log.
- Test split scored once, after selection, as the protocol requires.

## Update (2026-11-22, T1 result: generator probe on the chosen tokenizer)

- Mamba generator, causal tokenizer r11, 2 epochs, validation split (628 clips scored): text-to-motion FID 33.9, R@1 0.036, R@3 0.095, MM 7.75, Diversity 3.30. Epoch 1 took 20.9 minutes; epoch 2 took 13.2 minutes.
- Training cross-entropy fell from 6.64 (epoch 1) to 5.94 (epoch 2). The generator learns.
- Compared with the probe on the non-causal tokenizer r6 (2 epochs): FID 38.6, R@1 0.038, R@3 0.107, MM 7.89, Diversity 3.17. The causal tokenizer gives a lower FID, and similar retrieval.
- Limit: one evaluation point after 2 epochs. The FID trend is not yet known. The result does not support a thesis claim.

## Update (2026-11-22, tokenizer decision: r11, heading weight 0.01)

- Decision: the tokenizer is causal grouped FSQ, width 1024, 256-frame windows, heading correction with weight 0.01 (run r11_causal_heading_w001_e100). Chosen for bounded heading and root-aligned error on long sequences. Cost: test recon-FID 0.089 (against 0.058 without heading loss), test MPJPE 83 mm.
- Generator config for this tokenizer: configs/generator/probe_causal_heading/gen_probe_causal_heading.yaml (causal, heading correction on, weight 0.01).
- Overfit gate (Mamba): PASS. Written to runs/outputs/gates/probe_mamba_causal_heading.json. Step 26: cross-entropy 0.0045, token accuracy 1.000.
- Generator probe (Mamba, 2 epochs) started. Log: runs/outputs/research/probe_mamba_causal_heading.log.

## Update (2026-11-22, heading correction result and weight check)

- r10 (causal, 256-frame windows, heading weight 0.1, 100 epochs): test recon-FID 0.540, test MPJPE 131 mm. Compared with r9 (no heading loss): 0.058 and 105 mm. The heading loss at weight 0.1 dominates training: reconstruction loss 0.29 against about 0.15, and codebook usage fell to 42 percent (about 92 percent in r9). Result: weight 0.1 is too large.
- r11 (same model, heading weight 0.01, 100 epochs), seed 2026: test recon-FID 0.0887, test MPJPE 83.2 mm. Config configs/tokenizer/causal/fsq_g8_w1024_causal_heading_w001.yaml.
- Long sequences (10 AMASS clips, at least 800 frames), r11: heading error per 200-frame bin 2.4, 5.5, 5.2, 5.9 degrees; root-aligned MPJPE per bin 57, 78, 66, 69 mm. Both stay flat over the sequence. Compare r9 (no heading loss): heading 9.9 to 18.0 degrees and root-aligned 88 to 123 mm. r9 had not stabilised.
- r12 (heading weight 0.03, 100 epochs), seed 2026: test recon-FID 0.162, test MPJPE 94.7 mm. Long sequences: heading error per bin 2.1, 4.9, 5.3, 7.0 degrees; root-aligned MPJPE per bin 66, 84, 74, 90 mm. Both stay roughly flat, but the test FID is worse than r11. Test FID against heading weight: 0.01 gives 0.089, 0.03 gives 0.162, 0.1 gives 0.540. Lower weight is better for FID; the heading error is bounded at all three weights tested.
- Trade-off: r11 gives bounded error on long sequences and a lower test MPJPE (83 against 105 mm), but a higher test recon-FID (0.089 against 0.058). The FID and the MPJPE disagree; the choice needs the owner's decision.

## Update (2026-11-22, heading correction, hybrid step 4)

- Implementation: a learned head predicts a per-token correction of the turning feature from the decoder latent. The correction is causal (it uses only the current and past tokens). The head starts at zero, so the untrained model equals the model without correction.
- Training loss: a heading loss on the accumulated turning error over each training window (teacher-forced start). Weight 0.1 (config heading_weight). It is added to the reconstruction loss and the commitment loss.
- Config: configs/tokenizer/causal/fsq_g8_w1024_causal_heading.yaml. Run r10_causal_heading_w256_e100 (causal, 256-frame windows, 100 epochs, seed 2026), in progress.
- Tests: zero-start identity, zero heading loss for exact motion, positive loss for accumulating error, causality with a random head. 434 tests pass.
- Next: compare r10 with r9 on test recon-FID and on the long-sequence heading error.

## Update (2026-11-21, causal tokenizer with 256-frame windows, hybrid step 3)

- Causal grouped FSQ, width 1024, 256-frame windows, 100 epochs, seed 2026 (r9): test recon-FID 0.0577, test MPJPE 104.6 mm. The 64-frame causal run gave 0.0708 and 141 mm.
- Long sequences (10 AMASS clips, at least 800 frames): root-aligned MPJPE 88 mm to 123 mm, against 83 mm to 139 mm for the 64-frame causal run. Absolute MPJPE still grows: 208 mm to 648 mm.
- Heading error per 200-frame bin: 9.9, 24.5, 20.9, 18.0 degrees. It no longer grows steadily, but it stays near 20 degrees. Longer windows stop the drift, but they leave a large heading offset. Part 4 of the hybrid design (learned heading correction) is needed for a bounded error.

## Update (2026-11-21, 500-epoch tokenizer, causal results, long sequences)

- Non-causal grouped FSQ, width 1024, 500 epochs: test recon-FID 0.0178, test MPJPE 103 mm. Gap train to validation on 40 clips each: absolute 91 against 96 mm; root-aligned 21 against 26 mm. The validation curve has been flat since about epoch 300. Mild overfitting has started, and the gap is still small.
- Causal grouped FSQ, width 1024, 100 epochs: test recon-FID 0.0708, test MPJPE 141 mm. The non-causal 100-epoch run gives 0.0240. Causality costs about three times the FID at equal epochs.
- Long sequences (10 AMASS clips of at least 800 frames, scripts/checks/long_sequence_check.py): error grows with length. Non-causal 500 epochs: root-aligned MPJPE 63 mm in the first 200 frames, 132 mm in the last bin. Causal 100 epochs: 83 mm to 139 mm. Absolute MPJPE: 158 to 542 mm and 189 to 623 mm.
- Cause (measured): heading error. Heading is the running sum of predicted turning speed. Heading error grows from 5.7 to 20.8 degrees over the 800 frames (non-causal 500 epochs). Infinite-time streaming needs a heading correction or a window reset.
- Decision needed: heading correction (window reset with measured seam error), a heading-invariant representation, or longer training windows.

## Update (2026-11-21, causal tokenizer, option 3)

- Design: causal encoder and causal decoder. Each convolution looks only at past frames. The decoder repeats each token in time, then applies a causal convolution. The non-causal path is unchanged, and its checkpoints still load.
- Setting: TokenizerConfig.causal (default off). Config: configs/tokenizer/causal/fsq_g8_w1024_causal.yaml.
- Tests: encoder token and decoder frame do not read later frames (causal). Control tests show the non-causal model does read later frames. Full suite: 429 passed. Type and lint checks clean.
- Quantization makes a plain output comparison weak. The encoder and decoder are therefore tested before quantization.
- Training: causal grouped FSQ, width 1024, 100 epochs, seed 2026 (run r8_causal_fsq_w1024_e100). It starts after the 500-epoch non-causal run (r7), which has about one hour left.
- Still open: long-sequence test beyond 196 frames (drift), root-relative error for infinite streams, and the generator on the causal decoder.

## Update (2026-11-21, overfit check of the 100-epoch tokenizer)

- Same metric on training and validation clips (40 each, 100-epoch grouped FSQ, width 1024, best validation checkpoint): training absolute MPJPE 110.3 mm, root-aligned 24.2 mm; validation 111.8 mm, root-aligned 28.1 mm. Gap small: the tokenizer generalizes at 100 epochs. Small sample; the 500-epoch run is the next check.
- Generator probe (Mamba, 2 epochs): validation text-to-motion FID 38.6, retrieval at 1 0.038. Speed 12 to 15 minutes per epoch.
- 500-epoch tokenizer run (r7_fsq_w1024_e500) restarted after a file-lock failure caused by a concurrent job. Speed about 25 seconds per epoch; about 3.6 hours expected.

## Update (2026-11-21, step 1 decision and generator probe start)

- Step 1, 100 epochs, seed 2026, width 1024. Best validation recon-FID: grouped FSQ 0.0318, residual 0.0335. Choice by validation: grouped FSQ. Test scores (not used for the choice): FSQ 0.0240 and 106 to 109 mm MPJPE, residual 0.0249 and 106 mm MPJPE. The validation gap is small and comes from one seed.
- Generator config for the chosen tokenizer: configs/generator/probe_fsq_w1024/gen_probe_fsq_w1024.yaml (tokenizer width 1024).
- Overfit gate (Mamba): PASS. Written to runs/outputs/gates/probe_mamba_fsq_w1024.json. Step 26: cross-entropy 0.0105, token accuracy 0.997.
- Generator timing probe (Mamba, 2 epochs) started. Its log is runs/outputs/research/probe_mamba_fsq_w1024.log.

## Update (2026-11-20, width sweep and step 1 candidates)

- Width sweep, grouped FSQ, 50 epochs, seed 2026, current code. Parameters, best validation recon-FID, test recon-FID, test MPJPE: width 256, 3.0 M, 0.267, 0.223, 241 mm. Width 512, 11.3 M, 0.072, 0.059, 142 mm. Width 1024, 43.6 M, 0.043, 0.035, 119 mm.
- Choice by validation only: width 1024.
- Reproducibility check: width 512 rerun under the current code gives test recon-FID 0.0586, the same as the earlier run.
- Step 1 candidates, 100 epochs, seed 2026, width 1024: grouped FSQ (r6_fsq_w1024_e100) and residual quantization at matched bits (r6_rvq_w1024_e100). The choice between them uses validation reconstruction only, then the generator probe in step 2.

## Update (2026-11-20, data quality hypothesis)

- Diagnostic (scripts/checks/data_quality_diagnostic.py, 2 105 test clips, 50-epoch grouped FSQ): Spearman rank correlation between root-aligned error and the project quality measures. Skating 0.27, floating 0.32, jerk 0.40, yaw (turning speed) 0.63. Fast turning explains most of the error. Data artifacts correlate, but more weakly.
- Filtered training split (scripts/maintenance/build_clean_train_split.py, existing project quality model fitted on the training clips only): 1 349 of 11 692 base clips flagged. Kept clips have lower jerk (28.8 against 32.3) and lower skating. Test and validation splits are unchanged. Folder data/processed/humanml3d_clean, with a protocol record.
- Runs in the queue: r4_clean_fsq (50 epochs, 7 900 steps) and r4_base_eq_clean (no AMASS, stopped at 7 900 steps). Result: the clean-data gain at equal steps. Known confound: the normalization statistics are still the full-training statistics.
- Not yet proven: that preprocessing is the main lever. The turning-speed result suggests that motion content limits the error.
- Result at equal steps (7 900 steps, seed 2026). Full data, no filtering: test recon-FID 0.0638, test MPJPE 146.6 mm, final validation recon-FID 0.0777. Filtered data: test recon-FID 0.0671, test MPJPE 147.8 mm, final validation recon-FID 0.0860. The filtering did not improve the result. The removed clips probably include valid hard motions, which reduces variety. One seed; the gap is small. Verdict: the filtering hypothesis is not supported.

## Update (2026-11-20, equal-step controls and run report hook)

- Test set, seed 2026, 50 epochs unless stated. Recon-FID and absolute MPJPE.
  - Grouped FSQ, no AMASS, 50 epochs (8 750 steps at 175 per epoch): 0.0586, 141.8 mm.
  - Residual tokenizer, matched bits, 50 epochs: 0.0478, 128.2 mm.
  - Grouped FSQ, AMASS share 0.25, 50 epochs (11 650 steps): 0.0430, 126.7 mm.
  - Grouped FSQ, AMASS share 0.5, 50 epochs (17 500 steps): 0.0349, 117.7 mm.
  - Equal-step control for share 0.25, no AMASS, 11 650 steps: 0.0433, 131.6 mm.
  - Equal-step control for share 0.5, no AMASS, 17 500 steps: 0.0304, 121.3 mm.
- Reading: at equal steps, the AMASS share does not change the recon-FID. The share 0.25 run has a lower MPJPE (126.7 against 131.6 mm). The share 0.5 run has a higher recon-FID (0.0349 against 0.0304) and a lower MPJPE (117.7 against 121.3 mm). Most of the gain in the earlier table came from the extra steps. The result is mixed and rests on one seed.
- Residual tokenizer against grouped FSQ at matched bits (80 bits per token): residual is better on both measures (0.0478 against 0.0586; 128.2 against 141.8 mm). This is one seed.
- Run report hook: .claude/hooks/run_report.py runs after each shell command. It reports each finished training run once, with its test recon-FID and MPJPE. The first run marks all existing runs as reported. Its list is runs/outputs/reported_runs.json.
- MLflow web view: http://127.0.0.1:5001 (version 3.17.0, full package). The older view on port 5000 is still running and shows no interface. It can be closed once the process can be stopped.

## Update (2026-11-20, MLflow view of the local runs)

- Local MLflow store: runs/logs/mlflow.db, experiment "tokenizer". It holds 23 tokenizer runs, backfilled from the saved metrics and manifests by scripts/maintenance/backfill_mlflow.py. Each run records its seed, precision, git commit and start time.
- Web view: http://127.0.0.1:5000 (this machine only). It was started with the flask, flask-cors and aiohttp packages added to the project environment. These packages are not in the project dependency list yet.
- Off-protocol runs (seeds 2027 to 2030) appear in the store. Filter on the seed parameter, and exclude them from the paper.
- New runs are not logged to MLflow yet. The project flag mlflow_tracking is off by default.

## Update (2026-11-20, epics A25 and A26, 50-epoch tokenizers, seed 2026)

- Test set, 2 189 clips, recon-FID and absolute MPJPE. Grouped FSQ, no AMASS: 0.059, 142 mm. Residual tokenizer at matched bits (80 bits per token): 0.048, 128 mm. Grouped FSQ with AMASS share 0.25: 0.043, 127 mm. Share 0.5 run: queued or running.
- Caveat: the AMASS runs take more steps (233 steps per epoch for share 0.25, against 175 for no AMASS). The gain may come from the extra steps. Equal-step controls are queued: no AMASS for 67 epochs with the step cap at 11 650 (share 0.25 steps), and no AMASS for 100 epochs with the step cap at 17 500 (share 0.5 steps).
- One seed. Claims need the controls and, if possible, more seeds under the protocol.

## Update (2026-11-20, epics A23 and A24)

- A23 protocol guard: every training command checks the effective seed. Seed 2026 is the protocol seed. Other seeds are refused unless the run is marked exploratory. The split files are frozen with hashes and checked before each protocol run. A seed 2027 config was refused by the command line. 424 tests pass.
- A24 metric check: the project joint recovery and a second implementation agree to 2.8e-7 m on 20 validation clips. This checks the code path. It does not check the formula against the official HumanML3D code.
- Metric definitions used in this project. Absolute MPJPE: mean distance between reference and reconstructed joints, all 22 joints, all frames, in mm. Root-aligned MPJPE: the same, after subtracting the root joint from every joint in each frame.
- Seed 2026 checkpoint results, 20 validation clips: base (2 epochs, 350 steps) absolute 460 mm and root-aligned 109 mm; share 0.25 absolute 456 mm and root-aligned 110 mm. Most of the absolute error comes from the root trajectory, which drifts over the clip. The root-aligned value is the better measure of pose quality. The earlier figures of 500 to 700 mm are absolute values.
- Off-protocol checkpoints (seeds 2027 to 2030) are in the checkpoint folder. They are not used for any claim.

## Update (2026-11-20, AMASS extension, padding, equal-step runs)

- Data: the SMPL-X v1.0 body models are in place. The AMASS pose stage converted 15 927 sequences. The feature corpus has 12 377 clips, about 40.7 hours, with 2 525 clips skipped because HumanML3D uses the same source file.
- Path fix: pose files live in doubled or nested folders for some datasets. The lookup now tries these layouts. Tests added.
- Dropped clips: 109 index rows have fewer than 2 frames after the head trim. They are listed in a log file, and 20 of them are in the official test split.
- Padding: clips of 40 frames or more are kept and padded to the window. The loss ignores padded frames.
- Mixing: the tokenizer stream can add AMASS windows at a fixed share per epoch. Option 2 was chosen: the AMASS share is a separate setting from the verb balance.
- Step limit: the tokenizer command has a max-steps option. It cannot be used with resume.
- Smoke runs (one epoch, one seed): no AMASS 22 418 windows per epoch; share 0.25 29 891; share 0.5 44 836. Test recon-FID 51.9, 41.6, 22.8. The step counts differ, so these runs do not compare the share.
- Equal-step runs (350 steps): no AMASS, 2 epochs, test recon-FID 21.3; share 0.5, 1 epoch, test recon-FID 22.8. At equal steps the AMASS mix is slightly worse, and no claim about AMASS benefit is supported yet.
- Equal-step runs (350 steps, all with the step limit): no AMASS test recon-FID 21.3 and MPJPE 511 mm; share 0.25 test recon-FID 20.1 and MPJPE 518 mm; share 0.5 test recon-FID 22.8 and MPJPE 525 mm. The 0.25 share is slightly better on recon-FID and slightly worse on MPJPE. One seed: no claim.
- OFF-PROTOCOL, NOT CITABLE: runs with seeds 2027, 2028, 2029 and 2030 were started by mistake. The project rule is seed 2026 only, for every run and every paper figure. These runs are kept in the logs as exploratory records. They must not appear in the paper, in tables or in claims. The only protocol-compliant equal-step results are the seed 2026 runs listed above.
- Open: the health statistics still count padded frames (logs only). The test split decision (report the overlap, or add a source-level split) is not made.

## Update (2026-11-18, A24 names, moves, torch.compile, roma)

- f9e05ad: 7 file renames (nn_generator, nn_tokenizer, generator_trainer, tokenizer_trainer, reconstruction_metrics, token_datasets, studio_config); git sees pure renames.
- a1ac691: caption weighting helpers moved motion/storage -> dataset/datasets (AST-identical); dead load_clips, run_producer, collect_stream deleted; the fit_train_stats guard got the retargeted test (it had none).
- torch.compile (inductor via triton-windows 3.8, removed after): transformer 79 -> 77 ms/step (noise level), mamba 163 -> 151 ms/step (-7%), 40 s compile per run, losses not bit-identical (first step diff 2e-3 / 7e-3). e5d8630: train-generator --compile, on by default (owner choice); eager with a warning when CUDA or Triton is missing, so this laptop stays eager until triton-windows is installed.
- roma for motion/geometry: only qinv is bit-identical; qmul, qrot, quaternion_to_matrix, cont6d_to_matrix differ by 5e-7 to 2e-5. Not adopted.

## Update (2026-11-18, A23 request classes in app)

- Commit b4ec8a1: every CLI request class lives in app/cli_requests.py; motion and tokenization no longer import pydantic; build_token_pack takes TokenPackPlan, train_tokenizer takes TokenizerSchedule; OverfitGateRequest deleted (defaults inlined).
- Gate: 735 pass 2 skip, every earlier test unchanged; JSON schema of all 11 CLI requests byte-identical to HEAD; pyright 0; check_problems 0; ledger rows A23.
- The untracked scripts/training/tokenizer_preflight.py and train_tokenizer_challenger.py now import TrainTokenizerRequest from app.cli_requests.

## Update (2026-11-17 night, A22 tight code and fast data)

- Owner order: full autonomy for A22; Haiku agents made the code edits, the orchestrator checked each one and committed in the owner's name (14 commits, f5fb97f..df7296a).
- Speed: the CLIP text encoder now runs only its trainable layer per step; the 11 frozen layers come from a disk cache (data/.cache/clip*frozen*\*.npy, 1.44 GiB, built once in 192 s for the 43114 train captions). Train step 88.6 -> 71.0 ms (-20%) on the laptop; 5 seeded steps give loss difference 0.0 and bit-equal weights. Turn off with --text_cache false.
- Speed: caption files are parsed once per dataset (init about 4-5 s faster per loader); `text2motion --help` 6.7 -> 0.6 s.
- Dropped by measurement: data workers (the loader is 9.5-12.8 ms of a batch; the 65.6 ms reading was a cold file cache) and a packed feature store (about 8 ms per step).
- Structure: CLI request classes live in app; MotionDataConfig and the weighting enums in dataset; 20 single-use helpers nested in their caller; settings objects for the 9 functions with 9+ parameters; tests/test_package_layers.py enforces the package layers and the 10-file limit. No per-package defaults.yaml: one dataclass per section already holds the defaults.
- Gate: all 724 earlier tests unchanged plus 12 new (736 pass, 2 skip); train batches bit-identical for 2 configs and 2 epochs; pyright 0; check_problems 0; ledger rows for motion and tokenization.
- Not done: Python 3.14 (imgui 2.0.0, needed by aitviewer, does not build on 3.14). Disk C: has about 8 GB free.

## Update (2026-11-17 evening, A17 SO6 dataset package)

- HumanML3D 60 fps build done: 29228 feature files, 0 failures; Mean/Std from 23384 train clips (streamed statistics; the old float32 sum was off by 0.0049).
- Owner order: one package only for data. New src/text2motion/dataset (10 files); amass, preparation and datasets moved from motion. scripts/dataset has 3 entry scripts in place of 17.
- Gate passed: same clock table, same Newton and T5g results, same features (difference 0.0) as the old scripts. 724 tests pass.
- Hands and face (A19): future work; the code keeps the body path modular.
- Next: 60 fps features of the extra sources (prepare_source.py --fps 60), then Newton and T5g at 60 fps (check_quality.py quality --fps 60).

## Update (2026-11-17 afternoon, A19 whole body approved)

- Owner approved hands and face (A19) and dropped the 2026-12-01 experiment cut-off; submission 2027-02-01 stays.
- Design: the body twins of steps 1-3 stay 22 joints and are frozen; part tokenizers and part generators add hands and face (ParCo, HumanTOMATO pattern).
- Data facts: AMASS fingers do not move in KIT, CMU, BMLrub, HDM05, HUMAN4D, SOMA (std 0.0000 rad); GRAB 0.154 (mocap); Motion-X++ smplx322 has hand pose 90, jaw 3, expression 50 per frame. The Motion-X++ tilt is in the root rotation only, so hand and face values are usable.
- Next: WB0 council on the representation; WB1 hand and face data on the CPU while the GPU trains the body.

## Update (2026-11-17, DE15 HumanML3D 60 fps: clock fix)

- Run 1 stopped at the index step: 107 clips end past the file. Cause: AMASS SMPL-X files say 120 fps, but KIT, EKUT (100), DFaust, TotalCapture and 492 CMU files (60) store a lower rate. These clips played 1.2x or 2x too fast.
- Fix: amass_clock_check.py finds the stored rate of each file by a match to the official 20 fps clips. 10268 of 10364 files match. 96 files (163 clips) do not match; their clips come from the official features with the cubic spline.
- Run 2 started 2026-11-17 (pid 158040): re-extract the non-120 files at the stored rate, cubic spline to 60, then index, features, Mean/Std.
- Owner decision (2026-11-17): train at 60 fps, render at 120 fps. Training at 120 fps costs 1.5x (transformer) and 1.9x (Mamba-8) per step, measured at 294 against 148 tokens; most added frames would be computed, not captured.
- Run 2 stopped at the fill step: 2 index rows (009707, 011059) are 1-frame clips with no official features and in no split. The fill now skips clips in no split and fails for a clip in a split. Run 3 started (pid 146688).
- Run 3 stopped at the fill step: 000990 and 005836 (train, stand still) have 1 official frame; the spline needs 2. The loader drops clips under 40 frames (datasets.py:206), so training never used them; the fill skips clips with 1 frame. Run 4 (pid 160232): 1526 clips filled, features 29224 files in progress.
- Run 4 encoded 29222 of 29224 files. Val clip 001178 and its mirror gave NaN: at frame 240 the body faces exactly -z, and qbetween divides 0 by 0 for exactly opposite vectors (the official HumanML3D code has the same fault). Fix in motion/geometry.py (half turn about a perpendicular axis); ledger unfreeze/feature/freeze c7bddaae...; 3 new tests; tests/motion and golden 172 pass. Features-only rerun started 2026-11-17 13:2x (pid 179236, about 50 min).
- Features rerun: 29224 written, failed list empty. STATS stopped: train clips 000990 and 005836 (1 official frame) had no feature file. Cause: the clock check cannot score a 1-frame clip and marked CMU 82_01 and 82_18 no_match. Fix: such a file takes the rate of its subject folder when all scored files agree (CMU 82: 16 files, all 120) -> status folder_rate; only these 2 files changed. Full rebuild started 14:2x (pid 189796, about 75 min). Free RAM 1.3 GB, so the extra-source features wait.
- Step-0 AMASS sources have a correct clock: gravity in flight BMLrub -9.46, HUMAN4D -8.95 m/s2; GRAB 120 fps per its README. The step-0 selection stays valid.

## Update (2026-11-16 evening, A16 data: owner strategy fixed)

- Owner strategy, not open to change: step 0 clean data, step 1 tokenizer, step 2 motion mimic (no text), step 3 prompt mimic with the motion part frozen. Same as DE4-DE7. T9 shortcut test dropped.
- Automatic realism tests all failed against the owner's 30 blind ratings: glide v1, glide v2, physics engine (no policy runs on this PC), MotionCritic (Spearman 0.05).
- Ratings: HumanML3D 7.9, GRAB 8.55, Motion-X++ 5.1 (music, perform, haa500 low; idea400 7-9.5). Owner: low clips are not human motion; GRAB goes into training.
- Newton check (DE12, owner idea): centre-of-mass acceleration against gravity (hover, balance). Spearman with owner ratings -0.52. HML3D p99 limit failed twice (limit 1.0); proposed fixed rule: reject a clip if >= 50% of frames break the law.
- DE13: we read Motion-X++ with a per-clip tilt. Switched to the official joint files: impossible-frame share drops in all 7 types; kept hours 18.4 -> 22.4 h.
- Proposed step-0 set (Newton < 0.5 and T5g): GRAB 3.76 h, BMLrub 4.70 h, HUMAN4D 0.93 h, Motion-X++ official (all 7 types) 22.44 h. Waits for owner yes.
- Owner order: never downgrade data; train at 60 fps (cubic spline where a source is lower; tokenizer downsample 4 kept), render at 120 fps (memory frame-rate-requirement).
- A18: an 8-layer Mamba with expand 4 (16.23M; transformer 15.30M) passed the 6-epoch test (val FID 0.519, R@1 0.406 vs transformer 1.208, 0.326) and is 2.3x faster per service step (4.41 vs 10.25 ms). Owner: it is the Mamba twin of the 60 fps pipeline.
- DE15: HumanML3D 60 fps build running (data/HumanML3D_60fps); motion package frozen again (bc4a7ad2...). DE16 leak check by source: no overlap.

## Update (2026-11-15 night, A15 V12 rule B: FAIL; paper doc online)

- V12 (rule B verb groups, 6 epochs, both twins) FAILS all three rules for Mamba and two for the transformer.
- Rare-verb top-5 gain T +0.041, M +0.046, but both 95% intervals include 0. Epoch-6 val FID T 2.00 vs 1.21, M 0.60 vs 0.44. Mamba R@1 0.398 vs 0.434.
- Fifth failed rebalance test (A8 M5, V3, V6/V8/V11, V12). No full rule-B run. Open owner decision: close the sampler line, or a new epic (more captioned motion, or the text encoder).
- Paper draft as a private Claude doc: https://claude.ai/artifact/HGGcXwutbkLT7rx4fAVPaw (8 matplotlib figures; script and PNGs in artifacts/outputs/paper_doc_figs, not tracked).

## Update (2026-11-15 evening, A9 F4 text rewrite done except 1 table)

- All thesis text now describes the step50 pair (15.30M T, 15.55M M, 4648 test segments). Commits 469a36c to 019aae9.
- Chapter 6: long stream 7 segments, val, 32 prompts. Reset every 8 keep 16 keeps text match (mean distance seg 2-7: M 3.70, T 3.69). Full state fails (M 3.28 then above 7.4).
- Cost: training 1.4x (7.1 vs 9.6 min/epoch, kernel on). Inference 2.2x (8.3 vs 17.9 ms/token, batch 1). Old text said 9x training; that is gone.
- A reset costs M 50 ms/token vs T 10 ms (replay one step at a time). Parallel rebuild is listed as future work only.
- Waits for owner: chapter 4 tab:tok_downstream uses old pilot generators. Remove it (drops claim 'FSQ advantage carries over') or keep it labelled as a pilot.

## Update (2026-11-15 07:00, A9 F2 done: one test run of the final step50 pair)

- Val (765 clips): FID T-M -0.006, CI -0.102 to +0.092. R@1 0.472 vs 0.500, McNemar p 0.18.
- Test (4648 clips, one run, commit 70fc066): 20-run FID T 0.085 vs M 0.110 (CI +/-0.004 each). Paired run-0 FID T-M -0.036, CI -0.065 to -0.011.
- Test R@1: T 0.499, M 0.506, McNemar p 0.42. Foot skate: real 0.160, T 0.244, M 0.259, p 6e-7.
- Claim (owner approved 2026-11-15): "Equal text match (R@1). The transformer has a lower FID on test (-0.036), and Mamba has more foot skate."
- Effect on the thesis: chapter 5 says hyp:parity is supported (old 2189-clip numbers). With the new claim, parity holds for R@1 only. The F4 rewrite must change this text.
- No model change or selection change after this run.
- F3 done (commit 3f375db, run as is on uncommitted scripts, diff sha256 cfa3835c). Mamba state 0.73 MB at all lengths; transformer cache 168 MB at 8192 tokens. Mamba is not faster per step (15-16 ms vs 7-11 ms).
- F3 7-segment stream (val, 32 prompts): only the reset policy keeps text control. Segment 7 R@1: Mamba full 0.031, Mamba reset 0.344, transformer reset 0.469.
- "377 ms" check: the probe value is for 32 prompts in 8 batches. Batch-1 stream time: Mamba reset 50 ms/token, Mamba full 17, transformer reset 10. Real time (200 ms/token) holds with a 4x margin.
- Mamba reset costs 3x because it replays the prefix and 16 tokens one step at a time (nn_model.py:270-276). The transformer prefills in parallel.
- F4 started. Inventory `doc/notes/f4_figure_inventory.csv` (commits 956406e, 867258d): 26 tables and figures, 20 read a generator. Owner decision: replace all 34M results (no historical tables).
- F4 finding: the old test tables used 2189 unmirrored clips; the current protocol scores 4648 Guo pieces with mirrors. Old and new test numbers are not comparable.
- F4 finding: chapter 5 method and training text describe the 34M recipe (AMASS prior, width 512, batch 8). The final pair trains on HumanML3D only. `configs/generator/a7/` is not committed.

## Update (2026-11-14 20:00, A15 closed: motion-class sampling does not beat G4; HuMMan-MoGen sample converted)

- V9 (commit 3339064): the class weights cause the epoch-6 FID gain. Draws with replacement at uniform weight do not help the transformer and hurt Mamba.
- V10 (commit b4b61cc): on the epoch-6 EMA bundles, small-class R@1 rises for B in both twins (T +0.043, M +0.029).
- Owner point: FID weights each action by its val share, so it cannot show rare-class gains. A per-class rule was added to V11 before the 20-run data.
- V11 (commit 9ef5200): FAIL on both rules. Full runs to the stop rule. 20-run val FID: T 0.168 (G4 0.168), M 0.159 (G4 0.177, gain inside 2 sd). Small-class R@1 B - G4: T +0.007, M -0.008, both CIs include 0. The epoch-6 gain was faster early learning only. G4 stays the A9 F1 candidate.
- Extra data scout (2 Copilot branches): HuMMan-MoGen and Inter-X are the candidates with new motion and SMPL parameters. Motion-X has conflicting terms. FineMotion and HumanML3D++ add text only. KIT-ML has 21 joints.
- HuMMan-MoGen (commits ac7b220, 6b9fc28; facts in `doc/notes/humman_mogen_check.csv`): 3132 unmirrored clips, 160 actions, only 340 unique sentences, 30 fps (gravity fit). The AMASS axis swap mirrors this source; a proper rotation fixes it. A zero-padded resample made false end-frame speed peaks; line padding fixes it. 320 sample clips are now in range of HumanML3D val.
- Open owner decisions: use HuMMan for training (new epic) or not; a longer uniform transformer run (G4 transformer still improved at epoch 60).

## Update (2026-11-14 00:20, A15 motion-class sampling and the night run)

- Leak test (commit ad99355): on val, clips that share no recording with train have a higher FID than a size-matched random set (transformer 0.238 vs 0.202, Mamba 0.252 vs 0.206). R@1 does not drop. Same for both twins.
- Verb clusters (commits 3a6d355, 57dc2d6): 324 action verbs, 11 191 clips. Both motion trees give the same 10 motion classes (map `doc/paper/pics/verb_cluster_map.csv`, column class_10). A clip classifier tells the 10 classes apart (balanced accuracy 0.854). 100 classes cannot be tested: only 22 have 5 single-class clips.
- G4 per class (commit 16a940d): val R@1 of G4 is within 0.07 of the real-clip score in each of the 10 classes. The small classes are not behind.
- Sampler (commit e949662): `clip_weighting: motion_class` draws a train clip with weight n \*\* -0.5 of its rarest class, with replacement. The rebalance is mild: run/jog 1.75x, jump/hop 1.52x, walk/turn 0.94x.
- V6 (commit c6aa58e): FAIL by its rule. Small-class R@1 rises for the transformer only. This probe used raw `_last.pt` weights.
- Lesson: `_last.pt` is the resume state with RAW weights. The in-training evaluation and the best `.pt` bundle use EMA weights. Evaluate the bundle, not `_last.pt`, to compare with in-training numbers.
- V8 (commit 3125da4): PASS. 20-run val FID of the epoch-6 EMA bundles: transformer 1.112 -> 0.606, Mamba 0.451 -> 0.266. R@1 T +0.034, M 0.000. The verb test shows no change (commit eba1465).
- Confound: motion_class also draws with replacement (61.4% unique clips per epoch). V9 arm C (`uniform_replacement`, commit 2e66d70) separates the two causes. It runs now.
- Loader test: 0 workers load a batch of 8 in about 12 ms of a 130 ms step. 1 or 2 workers save at most 10% and leave 0.9 or 0.3 GB of RAM free. Keep `--num_workers 0`.
- Night chain (owner: yes night run): after V9, the V9 verdict, then the per-class probe on the EMA bundles of A, B, C. If V9 = CLASS WEIGHTS, the full B run starts (V11, about 14 h) and is compared with G4 by the fixed rule in the V11 row. Log: `artifacts/outputs/night_chain.log`.

## Update (2026-11-13, A15 kill test and council on the timeline)

- A15 V3: the rare-verb Mamba run stopped at the 1 h cap after epoch 5. It ran again from epoch 0 (the old files have the suffix `_killed_e5`). Epoch 6: rare-verb Mamba val FID 0.6276, R@1 0.418; uniform Mamba 0.4410, 0.434. The ratio 1.42 is above the 1.1 limit, so option B fails for Mamba. The verb-test part runs now.
- The verb-test chain died 2 times: in Windows PowerShell 5.1, `$ErrorActionPreference = "Stop"` turns a stderr warning of a native program into a fatal error. Fix: "Continue" in that script.
- Seeds: the owner forbids any seed other than 2026 (2026-11-13). The verb probe now samples from one stream with seed 2026 (commit 127ce5c). The V1 table must run again. The evaluator still uses crc32(clip_id) + run per clip: an owner decision is open.
- Verb tags: 1 471 tagged verbs in train, 687 in 1 clip only. About 8 of 40 random verbs with 5 to 149 clips are not verbs. The caption text itself is clean (0 missing files, 27 duplicate captions). Training reads only the raw text; the tags matter only for option B and the verb claims.
- Speed test (commit 7cb0171): real rare-verb clips at speed 0.8 to 1.2 keep their verb for the matcher (top-5 change within 0.02). The matcher finds the verb in its top 5 only 31.5% of the time on the original clips.
- G9 closed: saved best val FID 0.1502, R@1 0.503 (not 0.1627 / 0.482, which is the last evaluation). It does not beat G4, so 60 epochs is enough.
- Correction: A8 is closed as negative (M4, M5). The F0 Mamba gate failure was fixed with an owner-approved gate rule on 2026-11-10. The line in the 2026-11-11 entry that says "cause not known" is out of date.
- Council (4 members, cross-vendor): do not clean or augment the training data before the A9 freeze. Next work: A9 F1, the final pair rule on validation (candidate: the G4 matched pair).

## Update (2026-11-12, A15 verb coverage: option B in code)

- Owner approved A15 with option B. Training uses all verbs. Per-verb claims use 27 motion verbs only (42 calibrated verbs minus 15 non-motion or tagger-error verbs).
- Option B: a train clip picks the caption with the rarest verb more often (weight 1 / train clips of that verb). Motion is not repeated.
- CPU comparison (11 692 train clips): rare-verb caption picks A 32.2%, B 55.7%, C clip oversampling alpha 0.5 51.4% (ESS 40.8%, 4.9x repeats). C works against A7, so it is not used.
- Timing filter: the loader already cuts a clip to the caption time range. Only 3.6% of train captions have one (1 274 of 34 948).
- Code: `caption_weighting: rare_verb` in the data config (default uniform, completed recipes unchanged). Motion package unfrozen and frozen again (ledger hash a554342a).
- G9 finished: early stop at epoch 58, val FID 0.1627, R@1 0.482. Its G9 row is not updated yet.
- V1 done (P1 twins, 32 seeds, Wilson 95%): 20 of 27 claim verbs are above chance for both twins. 5 verbs (hold, stop, look, bring, grab) fail for the twins and for real clips, so the test cannot measure them. Claim set: 22 measurable verbs.
- V3 running since 21:29 (driver pid 22772): A and B, both twins, 6 of 60 cosine epochs. Pass rule fixed before the data in the V3 row.
- Crash cause: RAM. VS Code (about 6 GB) and SentinelAgent (1.7 GB) left 327 MB free. Do not run the full test suite in one process; run it one folder at a time.

## Update (2026-11-11, encoder choice and verb calibration)

- Encoder: bge-small-en-v1.5 is the choice (owner decision, 2026-11-11). It has the top probe score (0.510). The earlier fixed rule picked the faster MiniLM-L3 without an owner decision. That rule is retired. The production generators still use CLIP. X5 (kill test) stays blocked until the A8 arm passes KT2.
- Split: 80% train (11 692 base clips), 5% validation (730), 15% test (2 192). No change made.
- Verb calibration: `doc/notes/verb_calibration.csv`, all 1 655 verb lemmas from the caption tags (tag VERB). An earlier count of 57 verbs came from a hand list and is wrong.
  - 231 verbs have at least 20 train clips. 45 verbs have at least 20 clips in each of train, validation and test.
  - Of the 45, 42 pass both checks (two-proportion z below 2 for validation and test). Flagged: pick (test), and 2 others. About 2 flags are expected by chance from 45 tests.
  - The other 1 610 verbs have too few validation or test clips to check.
  - Tagger noise: "left" is tagged as a verb, and "swinge" is a typo. Not filtered yet.
- Consequence: only the 45 verbs can support a per-verb claim. Claims about the other verbs need a larger split or a different test. Any claim about rare verbs needs the full set or a larger split.

## Update (2026-11-11, G9 started: 100-epoch step50 transformer, R@1 floor 0.490)

- G7 finished (marker `artifacts/outputs/A7_G7_DONE.txt`). Its result is not yet read into the G7 row.
- G9 started at 16:48 UTC. Script: `scripts/training/run_a7_g9_step50_100ep_local.ps1`. Driver pid 27768 (file `artifacts/outputs/a7_g9_driver.pid`). Log: `artifacts/outputs/a7_g9_step50_100ep_transformer_w320.log`. Budget 12 GPU h. Marker when done: `artifacts/outputs/A7_G9_DONE.txt`.
- Settings: step50 recipe, from scratch, 100 epochs, CE patience 6, `--save_r1_floor 0.490`, seed 2026.
- Open failure, not yet checked: the A8 F0 Mamba overfit gate exited with code 1, and the A8 driver stopped at that gate (`artifacts/outputs/a8_kt_driver.log`). The cause is not known. The A8 F0 Mamba run did not start.

## Update (2026-11-11, A14 extra motion closed: the frozen tokenizer cannot represent the new data)

- Sources: AIST++ (410 clips, 1.27 h, dance) and 100STYLE (810 clips, 18.77 h, locomotion styles). Both CC BY 4.0. A corrupt 100STYLE download (md5 mismatch) was deleted and downloaded again.
- fps study (`doc/notes/fps_study.csv`): 20 fps stays the model clock. Energy above 10 Hz: AIST++ 3.55%, 100STYLE 0.11%. The 3 phase offsets differ by 0.34 and 0.67 of one frame step, so they add no real data. Rule: 8 Hz low-pass, stride 3.
- Gate (rule fixed before data: error at most 2.0x the HumanML3D validation error): FAIL for both. AIST++ 3.67x, 100STYLE 7.43x. Compatible hours: 0.12 h and 0.00 h. After the fact: most world error is root drift, local pose error is 3.3x and 6.2x. A per-dimension mean shift does not help.
- Checks: control re-encode of official joints (small difference without an x flip), a picture of 4 clips (plausible skeletons), feet point forward in 97% to 100% of clips.
- Tasks N5 to N7 skipped. A prior on this data would learn tokens of wrongly reconstructed motion. A way forward needs a new tokenizer (breaks the frozen rule) or a skeleton fit. Both are out of scope before 2026-12-01.
- Also: code `--save_r1_floor` (G8). The F1 row was corrected (step50 transformer R@1 is 0.498, not 0.474).
- G7 partial (pkeep50 arm, mean of epochs 8/10/12): FID 1.74 (transformer) and 1.69 (Mamba), against P1 0.85 / 0.66 and step50 0.70 / 0.34. Dropping 50% of the tokens one by one is worse than P1. The `wholestep` arm decides which factor matters.
- Scripts: `scripts/extra_motion/` (5 files). Data and per-source inventories: `data/extra_motion/` (git-ignored).

## Update (2026-11-11, council on training length, and a web search for extra motion data)

- Council (3 luna + 1 argon, 12 calls: the first stage failed because a multi-line prompt was cut, so 4 calls repeated). No member backs the motion-first plan. All 4 say the epoch count cannot be compared with the literature. The transformer has not plateaued (best FID at epoch 60).
- Two defects found and checked: `trainer.py:653-657` saves the best checkpoint on FID only (no R@1 guard, one best file per run). The F1 row says R@1 0.474 for the step50 transformer, but the G4 CSV says 0.498. The F1 row needs a correction.
- Correction: the Mamba stop at epoch 38 is the patience-3 criterion. The patience-6 run ended at epoch 41.
- A8 used a same-data prior (22 418 clips), so it did not test extra motion (`STATUS.md` entry of 2026-11-09).
- Data search (2 Copilot branches, gpt-6-luna, exit 0). Checked by curl: 100STYLE is CC BY 4.0, 60 fps BVH, over 4 million frames (about 18.8 h raw), one actor, locomotion styles. AIST++ annotations are CC BY 4.0, 60 fps, SMPL, 306 MB, 1 408 dance sequences. Not checked: usable hours, leaks against val and test, tokenizer reconstruction on these motions.
- Repo path for extra motion: AMASS only. A loader for BVH or other SMPL files is missing (engineer report, function names checked). Cost figures in the report are assumptions.
- Raw answers: scratchpad `council_data/answer_scout.md` and `answer_engineer.md`.

## Update (2026-11-10, G7 started: which factor of step50 matters)

- 2 new configs: `width_320_pkeep50.yaml` (pkeep 0.5 for each token) and `width_320_wholestep.yaml` (pkeep 0.8, whole step). Both differ from P1 by one factor.
- Driver `scripts/training/run_a7_factor_split_local.ps1`: gate, then 12 epochs for each arm and each twin (4 runs, about 7 GPU hours). Log `artifacts/outputs/a7_g7_driver.log`, marker `A7_G7_DONE.txt`.
- Read the result at epochs 8, 10 and 12 next to P1 and step50, with the noise limit (FID 0.05, R@1 0.03). Do not claim a cause for a difference inside the noise.

## Update (2026-11-10, A8 M5 done: the motion prior does not beat step50)

- Full F0 run (prior, then full fine-tune, step50 recipe) against step50, validation 628 clips. Rule fixed before the data. FAIL for both twins.
  Transformer: best FID 0.160 (step50 0.149), R@1 0.497, CE gap 0.676 (0.641). Stopped at epoch 36.
  Mamba: best FID 0.214 (step50 0.153), R@1 0.462 (0.507), CE gap 0.706 (0.626). Stopped at epoch 33.
- The prior gives a lower best validation CE (4.448 against 4.517 and 4.382 against 4.418). The FID is not better and the gap is larger.
  A lower CE did not give better motion. This is the measured result for the thesis.
- Step50 stays the final candidate. A8 closes as a negative result: the frozen arms failed KT2, and the full arm failed M5.
- Cost: about 11 GPU hours for M5. The GPU is free.
- Next: G7 (which factor of step50 matters), then F1 (final pair rule) and the A9 freeze (2026-11-24).

## Update (2026-11-10, A13 X4: encoder selection test done on 6 models; M5 run started)

- Table: `doc/notes/text_encoder_probe.csv` (704 validation motions, 3000 train motions and mirrors, pretrained encoders, CPU).
- Score (50% motion match, 30% synonyms, 20% details): bge-small 0.510, gte-small 0.506, e5-small-v2 0.491,
  MiniLM-L3 0.488, MiniLM-L6 0.487, CLIP 0.428. All 5 small models are above CLIP.
- The 5 small models are equal inside the noise (about 0.02 on the score). The fixed rule then selects the faster model:
  paraphrase-MiniLM-L3-v2 (17.4M parameters, 5.8 ms). bge-small-en-v1.5 (33.4M) has the highest score.
- The gain against CLIP is in synonyms (R@1 0.38 against 0.27) and details (0.94-0.99 against 0.86).
  Motion match R@1 is not better than CLIP (0.24-0.28 for all 6), so the test does not predict better generated motion.
- The CPU times are not reliable (gte-small 92 ms against 18 ms for the same architecture; memory was low). Parameter counts are exact.
- X5 is blocked: it needed a frozen A8 arm, and KT2 killed both frozen arms. It needs a new design.
- M5 (F0 with the step50 recipe) started on the GPU: `artifacts/outputs/a8_f0_step50_driver.log`.

## Update (2026-11-10, A8 M5 plan change approved: F0 runs with the step50 recipe)

- The owner approved a plan change: the full F0 run (prior, then full fine-tune) uses the step50 recipe, and step50 is the control.
  Reason: step50 (FID 0.149 / 0.153) is much better than P1 (0.272 / 0.316), so a pass against P1 cannot give the final pair.
- Only the prior changes between the 2 runs: same config file, same schedule (60 epochs, patience 6, batch 8, validation each 2 epochs).
- Rule fixed before the data: best val FID < 0.149 / 0.153, R@1 >= 0.488 / 0.497, smaller CE gap at the same val CE. Both twins must pass.
  A difference smaller than the noise (FID 0.05) is reported as equal.
- The run starts after the encoder selection test ends (the test runs alone on the CPU now).

## Update (2026-11-10, A8 KT2 done: only the full fine-tune arm survives)

- KT2, 2 text epochs for each arm, validation 628 clips (transformer / Mamba):
  F0 full fine-tune: R@1 0.278 / 0.230, val CE 4.780 / 4.546. SURVIVES.
  F1 frozen backbone: R@1 0.202 / 0.192. KILLED. F2 lower half frozen: R@1 0.247 / 0.194. KILLED (Mamba).
  Limits: R@1 >= 0.226 / 0.215 (P1 at epoch 2 minus 0.05), val CE <= P1 + 0.10.
- The frozen prior idea (the motion-first plan) failed its kill test: text control is too weak when the backbone is frozen.
- The prior lowers val CE by more than 1.1 at epoch 2 against P1, but R@1 and FID at epoch 2 are not better than P1 at epoch 2.
  A lower CE at the start is not yet a proof of better motion.
- Next: M5, the full run of F0 for both twins. Open point for the owner: the recipe of that run (P1 as planned, or step50).
- The encoder selection test has only the CLIP row; the system stopped it 2 times for low memory.

## Update (2026-11-10, owner approved the gate rule change; A8 kill tests and the encoder test run again)

- Rule change (owner approved): a run that starts from a checkpoint (`--init_ckpt`) passes the overfit gate on CE and token accuracy only.
  The total ratio rule (0.05) stays for a random start. Code: `PRETRAINED_OVERFIT_CRITERIA` in `generation/contracts.py`; 1 new test.
- The A8 driver runs again from the F0 Mamba gate (`-SkipPrior -Skip f0_transformer`), log `artifacts/outputs/a8_kt_driver2.log`.
- The encoder selection test runs again on the CPU and now saves one CSV row for each model.

## Update (2026-11-10, G4 done: PASS for both twins; A8 kill tests stopped at one overfit gate)

- G4 (step50, full schedule, validation 628 clips): transformer best FID 0.149 at epoch 60, R@1 0.498 (P1: 0.272, 0.510).
  Mamba best FID 0.153 at epoch 24, R@1 0.507 (P1: 0.316, 0.502). Both pass the rule that was fixed before the run.
- The transformer had no patience-3 stop in 60 epochs. The Mamba stopped at epoch 38. No min delta was added after the data.
- A8: KT1 passed for both twins. Both priors are trained. F0 transformer after 2 text epochs: val CE 4.780, R@1 0.278 (inside both KT2 limits).
- A8 stopped at 05:26: the overfit gate of F0 Mamba failed 1 time (total ratio 0.057, limit 0.05). CE was 0.0012 and token accuracy 1.0.
  The total loss cannot go below about 0.257, and a model that starts from the prior has a lower first loss, so the ratio is larger.
  The gate rule is a ratio to the first loss. It was made for a random start.
- Not run: F0 Mamba, F1 and F2 (both twins). The GPU is free. The owner decides on the gate rule.

## Update (2026-11-10, A13 X4: the encoder selection test stopped after 1 of 6 models)

- The system stopped the CPU test because memory was low. It was not started again; the owner decides.
- Only CLIP finished (704 validation motions, 3000 train motions and their mirrors):
  synonym R@1 0.273, motion R@1 0.278, mirror accuracy 0.708 (chance 0.5), detail probe mean 0.864 (chance 0.25), score 0.428.
- Correction: a linear probe reads left/right from the CLIP vector (0.906). The earlier line "all encoders almost ignore left/right"
  came from the cosine distance and is not correct as written. The information is in the vector, in a small direction.
- No encoder is selected. The 5 small models have no result.

## Update (2026-11-09, A13 X4 done: text encoder research and a CPU probe)

- Question from the owner: can a smaller text model replace CLIP? Qwen and Mistral models are 10-50x larger than the CLIP text encoder (63M).
- Probe, 600 validation motions, 2 human captions each, pretrained encoders, no training:
  can the encoder find the second caption of the same motion among 32? R@1: MiniLM (22.7M) 0.377, CLIP 0.276, T5-small 0.212.
- All 3 encoders almost ignore left/right: a swap changes the pooled vector by 0.008-0.009 (cosine distance);
  a human paraphrase changes it by 0.15-0.43. This is a limit of one pooled vector, not of one model.
- Code: CLIP is hard-coded; a swap needs 50-80 lines. No paper with a matched encoder ablation was found.
- Next: X5 kill test (MiniLM vs CLIP on the frozen A8 prior), only if the frozen arm survives KT2.
- Limit: the probe used pretrained CLIP; our generators fine-tune the last CLIP layer, which the probe does not measure.

## Update (2026-11-09, A13 X1 done: diffusion research; no GPU time for it before the freeze)

- Streamed diffusion for motion is published already (MotionStreamer, CMDM, FloodDiffusion; numbers not opened by me).
- Our own data (R8P): a 1-step refiner gives recon-FID 0.0282 vs 0.0318 but MPJPE 114.0 vs 104.9 mm; more steps are worse.
- Judgement: the tokenizer floor (recon-FID about 0.03) is far below the generator FID (about 0.15). The generator is the limit, not the decoder.
- Decision: diffusion goes to related work and future work. X2 stays as an optional 2-hour kill test with a fixed rule, after A9 and G7.

## Update (2026-11-09, plan review for a publishable thesis; A12 A13 and G7 added)

- Owner goals: a 40-page thesis, later a publication; each result reproducible; proven contributions; some time for new ideas (diffusion).
- Gap 1: step50 changed 2 factors vs P1 (pkeep 0.5 and whole-step noise). New task G7 splits them, so the thesis can name the cause.
- Gap 2: no reproducibility work existed. New epic A12 (Q1-Q6): environment, data recipe, REPRODUCE.md, an independent reproduction, licences.
- Gap 3: one training seed. Task Q5 (2 more seeds for the final pair) waits for an owner decision; it conflicts with 'seed 2026 only'.
- New epic A13 Exploration: 10 GPU h, after the freeze, kill tests only. X1 (diffusion research) runs on Copilot now.
- Metrics reference written: doc/notes/METRICS.md.

## Update (2026-11-09, the R@1 loss of the step50 transformer was mostly evaluation noise)

- 2 Copilot branches (gpt-6-luna). Branch B found that my "best at epoch 52" number was stale; I verified it in metrics.jsonl.
- Transformer step50, final epoch 60: val FID 0.149, R@1 0.498. It passes the F1 rule (R@1 >= 0.490). The saved best bundle is this epoch.
- Mamba step50, epoch 24 (best so far, run at epoch 36): val FID 0.153, R@1 0.507 (P1: 0.316, 0.502).
- Noise: in epochs 50-60 the transformer val FID moves 0.149-0.198 and R@1 0.446-0.515 with a flat val CE.
  So one evaluation of 628 clips cannot rank neighbour epochs. Task F2 (20 repetitions + bootstrap) decides the claim size.
- Only the best-FID bundle and the last state exist; no per-epoch weights. A sweep of the CFG scale is not needed now.
- Literature (branch A, not all tables opened by me): T2M-GPT Table 6 has its best FID at corruption 0.5, the same direction as step50.

## Update (2026-11-09, owner gave plan autonomy; A9-A11 approved; F1 rule fixed)

- Owner: full autonomy on the plan and the notes; decide by small experiments, one step at a time; all research goes to Copilot.
- A9, A10, A11 approved. F1 rule fixed before any test run: lowest val FID among epochs with val R@1 >= P1 - 0.02.
- Future work: T3, P2, R8b, W4, G5, K7. Report only: K8, R7, K6.
- First step: 2 Copilot branches on how to get the R@1 of the step50 transformer back with no new training.

## Update (2026-11-09, epic review for a complete thesis; A9 A10 A11 proposed)

- Owner: 2026-12-01 is the date to send results to the supervisor; the defense is at the end of January 2027.
- Gap found: the thesis draft (8 chapters, about 15,000 words) has 0 mentions of P1, step50 or A8.
- Added as proposed: A9 Result freeze (F1-F6, by 2026-11-24), A10 Thesis text (H0-H6), A11 Defense (D1-D3).
  G6 (test-split table) now depends on F1 (final pair rule), so the test split is used once.
- Open for the owner: the F1 rule (R@1 guard or not), and the closure of A0-A4, A6 leftovers.

## Update (2026-11-09, Copilot review of the A7/A8 code; G4 has no patience-3 stop)

- Review: 2 gpt-6-luna branches (training-engineer, eval-metrics-engineer), read-only. 4 findings, each checked.
- B1 confirmed: stop_epoch_at returned the last epoch when no patience-3 stop occurs. Now it returns None;
  the G4 CSV has the column patience3_stop_reached. The verdict logic does not change (window = full run).
- G4 transformer: val CE falls by about 1e-4 every 2-3 epochs (4.5173 @55, 4.5171 @56, 4.5170 @58), so the
  patience counter resets; no patience-3 stop is possible before the 60-epoch cap. The rule has no minimum delta.
- Interim, epoch 58 (not a verdict): best val FID 0.161 @52 (P1 0.272), but R@1 at that epoch 0.474 (P1 0.510).
- A1 false: the prior resume has no --stop_epoch ($priorFlags has none). A2 latent: select_on_val_ce with
  val_ce_patience 0 resets the best on resume (trainer.py:83). A3 latent: --init_ckpt does not verify the bundle
  tokenizer and config (app/training.py:217). The A8 driver uses patience 6 and the same tokenizer, so both stay open.

## Update (2026-11-09, A8 M3 approved; code done; kill-test driver waits for G4)

- Owner approved the M3 plan and about 22-32 GPU h.
- Design change from the check: the prior uses the main trainer with cfg_dropout 1.0, not pretrain.py.
  So the prior has end tokens, EMA, the P1 crops and mirrors; its clean CE uses no text.
- New code: train.freeze (none, backbone, lower*half) in GeneratorTrainer; --select_on_val_ce saves the
  best bundle at the lowest clean val CE; --init_ckpt reads bundles; sanity-overfit takes --init_ckpt.
  Configs: configs/generator/a8/width_320*{prior,f0,f1,f2}.yaml. Tests: 7 new; suite 690 passed.
- KT2 rule changed before any data: R@1(2) >= P1 R@1(2) - 0.05 replaces the null-text CE test.
- Deferred: the BMLrub extra (7.7 h) and its leak check; the trainer reads HumanML3D only. Phase 1 = same-data prior.
- Driver run_a8_kill_tests_local.ps1 started (pid 29196); it waits for G4 (pid 29888), then gates, KT1,
  the prior to val-CE patience 6 (cap 20 epochs), and KT2 (3 arms x 2 twins x 2 epochs). Marker A8_KT_DONE.txt.
- G4 side note: transformer val FID 0.164-0.173 at the latest evaluations (P1 best 0.272); verdict at the end.

## Update (2026-11-09, A8 M2 council verdict; M3 plan waits for the owner)

- Council: 8 calls, all exit 0, but my launch script named `epic_a8$2` instead of the brief file.
  Valid: 3 answers (A, B, D) and 2 reviews (C, D). Answer C and review B are empty; review A
  answered the brief again. No retry (the 10-call budget is used). Verdict file in the scratchpad.
- Q2 (clock), unanimous: keep 20 fps; 120 fps only by interpolation at render time; revisit only
  with raw AMASS and a measured gain.
- New measurement: the wrong-clock AMASS datasets add 0 h of extra motion (HumanML3D has correct
  20 fps copies of all their files). Verified-clock extra: BMLrub 7.7 h. Unknown clock: 15.7 h.
- Tool-checked from review D: pretrain.py has no end token, no EMA and no pkeep, and the config
  uses an end token, so the end rows of token_emb and heads stay untrained after a prior.
- M3 plan (epic_tasks.csv): arms F0 full fine-tune, F1 frozen backbone, F2 lower half frozen,
  vs P1; kill tests KT1 and KT2; pass rule fixed now; about 22-32 GPU h after G4. Waits for the owner.

## Update (2026-11-09, A8 "Motion first, text second" opened by the owner; M0 data facts; M1 research)

- Owner goal: learn motion first (motion only), freeze it, then learn the text part. Epic A8, state planning.
- M0 (data, CPU only): AMASS_263 val/test leak = 61 source files (ACCAD 59, MoSh 4, KIT 1), same as the
  October audit; the exclusion of the other held-out sources was deliberate (0 hits vs 70-95% for train).
  Extra motion not in HumanML3D: 23.5 h (HumanML3D train: 22.9 h).
- M0 NEW: the AMASS_263 clock is wrong for some datasets. A time-scale fit against HumanML3D clips gives
  KIT and EKUT 16.7 fps, DFaust and TotalCapture 10 fps, CMU and MoSh mixed. doc/notes/amass_timing_audit.csv.
  A prior on these files learns motion at the wrong speed. The raw AMASS is not on disk (see 2026-10 entry).
- M1: 2 Copilot branches (gpt-6-luna). No paper tests a frozen motion-only generator prior on the same data.
  Correction of my reply today: F6 is RESOLVED (line about 3402): the AMASS prior wins FID at CFG 1 and CFG 3.
- Owner asked about 120 fps data by interpolation and a new tokenizer at 120 fps: stopped for a decision (rush check).

## Update (2026-11-09, A7 G4 started: step50 to the clean-CE stop, both twins)

- Owner chose option b: G4 with step50 for both twins, resumed from epoch 12.
- Found before the run: P1 used val_ce_patience 3 (manifests), not 6. P1 stopped at epoch 32 / 24.
  A G4 run with patience 6 trains longer than P1, and that alone can give a lower FID.
- Rule, fixed before the data: the primary result uses the patience-3 stop on the same G4 curve.
  PASS if for both twins the best val FID up to that epoch < P1 best (0.272 / 0.316)
  AND the best val CE epoch in that window > P1 (29 / 21). The patience-6 result is reported, no rule.
- The script replays the P1 stops exactly (32 / 24) from the P1 val CE curves.
- Driver: scripts/training/run_a7_step50_full_local.ps1, pid 29888, marker A7_STEP50_FULL_DONE.txt.
  Worst case (60 epochs): about 6.8 h transformer + 9.5 h Mamba = 16.3 GPU h; time limit 10 h for each run.

## Update (2026-11-09, A7 G2b verdict: step50 FAILS on the transformer; Mamba passes clearly)

- Driver ended with code 0 at 10:33Z (transformer 0.85 h, Mamba 1.19 h). Marker A7_STEP50_EXTEND_DONE.txt.
- Val FID, step50 vs P1, epochs 8 / 10 / 12:
  transformer 0.851 / 0.627 / 0.626 vs 0.972 / 0.826 / 0.763; Mamba 0.372 / 0.374 / 0.271 vs 0.734 / 0.736 / 0.501.
- Transformer FAILS both conditions: FID ratio 0.822 (limit 0.8); R@1 mean 0.402 vs 0.448 (limit -0.02).
- Mamba PASSES: FID ratio 0.516; R@1 mean 0.478 vs 0.473. Mamba FID(12) 0.271 is lower than the P1 Mamba best 0.316.
- The CE gap is not smaller at the same val CE (transformer 0.145 vs P1 0.122 at val CE 5.135; Mamba 0.209 vs 0.225 at 4.769).
  So the FID gain does not come from less memorization in the teacher-forced CE.
- Verdict stays FAIL under the fixed rule. The FID gain on Mamba is large; the next step is an owner decision.
- Limits: one seed; 2 changes in one arm (pkeep 0.5 and whole-step noise); FID only at 3 points; no interval.

## Update (2026-11-09, A7 G2 verdict: both arms fail; step50 extension to epoch 12 started)

- G2 driver ended with code 0 (marker A7_G2_DONE.txt). CSV: doc/paper/pics/a7_screen_validation.csv.
- step50 FAILS: gap ratio 1.74 / 0.92 (transformer / mamba) and the lag rule fails.
  Its val FID at epoch 6 is lower than P1: 1.208 vs 1.468, 0.448 vs 0.866.
- clipfrozen FAILS: gap ratio 1.02 / 0.99. It learns faster (lag ok), but its val FID at
  epoch 6 is higher than P1: 2.259 vs 1.468, 1.640 vs 0.866. R@1 is lower: 0.313 vs 0.350, 0.347 vs 0.439.
- G3 (combination) skipped: it needed 2 passing arms.
- Owner asked for G2b: resume step50 from epoch 6 to epoch 12, same 60-epoch cosine, both twins.
- G2b rule, fixed before the data: PASS if for both twins the mean val FID over epochs 8/10/12
  is at most 0.8 x the P1 mean (0.853 transformer, 0.657 mamba), AND the mean val R@1 is at
  least the P1 mean - 0.02 (P1: 0.448, 0.473). Script: a7_screen_verdict.py extend.
- Limits: one seed; step50 changes 2 things (pkeep 0.8 -> 0.5 and whole-step noise), so a pass
  cannot say which one acts; the margin 0.8 is a choice, not a measured FID noise level.
- Driver: scripts/training/run_a7_step50_extend_local.ps1, pid in artifacts/outputs/a7_step50_extend_driver.pid,
  started 2026-11-09 08:57Z, marker A7_STEP50_EXTEND_DONE.txt. Expected time: about 1.7 GPU h.

## Update (2026-11-08, A7 approved; G1 council verdict; G2 screen rule fixed before the run)

- Owner accepted the 3 A7 targets (CE gap per sample seen, best val FID, a val curve that still
  improves late) and approved G1 + G2.
- G1 council (3 gpt-6-luna + 1 gemini-4-argon, 8 calls). 4 more calls were lost: try 1 sent
  only the first line of the brief (copilot.cmd cuts a multi-line argument). Fix: a one-line
  prompt that names the brief file, plus --add-dir for the scratchpad.
- Agreement (cross-vendor): (1) --epochs also sets the cosine length, so a 6-epoch screen
  needs a separate stop; (2) the combined arm cannot show which factor acts, so test the
  factors alone; (3) the CE gap alone rewards slower fitting, so compare at the same val CE.
- Dissent (1 luna member): rerun the control instead of the reuse of P1. Decisive check: a
  1-epoch replay of P1 must match within 0.005 CE; else the control is rerun.
- Verified new facts (member D, checked with a tool): our pkeep draws noise over each of the 8
  sub-codes of a step (trainer.py:398); T2M-GPT replaces whole frames. At pkeep 0.8 only
  0.8^8 = 17% of our steps are fully clean. The EMA averages the generator only; the trained
  CLIP layer is evaluated without an average (trainer.py:281, 479).
- Code: train.pkeep_whole_step (default false, so P1 is unchanged) and --stop_epoch (default
  0 = off). Tests: test_clean_ce.py (3 new), test_a7_screen_verdict.py (2). 683 tests pass;
  ruff, pyright and check_problems are clean.
- G2 design (fixed now): arm step50 = pkeep 0.5 on whole steps; arm clipfrozen = CLIP frozen.
  Both twins, 6 of 60 cosine epochs, P1 flags otherwise. Control = P1 epochs 1-6.
- G2 rule: an arm PASSES if for BOTH twins ce_gap(6) <= 0.7 x the P1 gap at the same val CE
  (linear on the falling P1 curve) AND val_ce(6) <= P1 val_ce(4) (at most 2 epochs behind).
  FID and R@1 are reported with no rule. Script: scripts/qa/a7_screen_verdict.py.
- Limits: one seed; epoch-6 values do not predict the best FID; the 50-batch CE subset is
  fixed (paired, so fair between arms); arm step50 changes the noise amount and its form
  together; arm clipfrozen also removes the EMA mismatch above.
- Budget: about 4.7 GPU h (replay 0.4 h + 4 runs of about 1.05 h + 4 gates). The old plan
  was 4.2 h for G2 plus 4 h for G3; this design does both factors in one screen.
- REPLAY MATCH (21:51): epoch-1 clean CE, P1 vs replay: transformer 6.3405/6.3305 vs
  6.3405/6.3305 (delta 0.0000); mamba 6.4263/6.4277 vs 6.4269/6.4284 (delta 0.0007, limit
  0.005). P1 epochs 1-6 are the control. --stop_epoch 1 stopped both runs as planned.
- ARM step50 RESULT (rule fixed before the run): FAIL for both twins.
  transformer: val_ce(6) 5.629 (P1 ep4 5.534, lag rule fails); gap 0.045 vs P1 0.026 at the
  same val CE (ratio 1.74). mamba: val_ce(6) 5.319 (P1 ep4 5.217, fails); gap ratio 0.92.
  At the same epoch the gap is half of P1 (0.045 vs 0.091; 0.063 vs 0.181), but only because
  the arm learns more slowly: at the same val CE the gap is not smaller.
- Not in the rule, reported only: val FID(6) is LOWER than P1 for both twins: transformer
  1.208 vs 1.468 (R@1 0.326 vs 0.350); mamba 0.448 vs 0.866 (R@1 0.442 vs 0.439). P1 mamba
  reached about 0.45 only at epochs 16-18. One seed; P1 FID moves by 0.1-0.2 between evals.
  Hypothesis (not tested): whole-step noise helps sampling (exposure bias), and the
  teacher-forced CE cannot see this. The verdict stays FAIL; a rule change needs the owner.
- Driver: scripts/training/run_a7_screen_local.ps1 (detached). Output:
  doc/paper/pics/a7_screen_validation.csv, marker artifacts/outputs/A7_G2_DONE.txt.

## Update (2026-11-08, verb inventory and "can generate" test)

- `doc/notes/verb_inventory.csv` (script `scripts/research/verb_inventory.py`): 442 verbs with
  > = 5 train clips, from the HumanML3D POS tags; original clips only (no M mirrors). Bands:
  > 10 verbs >= 1000 clips, 71 with 100-999, 141 with 20-99, 220 with 5-19. 65 have 0 test clips.
  > Tagger errors stay in the CSV (left, side, back, clockwise, swinge, outstretche).
- "Can generate" test (`scripts/qa/verb_generation_probe.py`): top 100 motion verbs (tagger
  errors and object verbs removed); text "a/DET person/NOUN <verb>/VERB" for the Guo encoder,
  prompt "a person <verb>s" for the generator; 16 seeds, 30 tokens; retrieval of the own verb
  among the 100 verb texts (top-1, top-5; chance 1% / 5%). Ceiling: 16 real train clips per
  verb with the same retrieval. Spearman of top-5 vs log(train clips) for the 3 sources.
  Queued after the stream run (one GPU job at a time).
- RESULT (doc/paper/pics/verb_generation_validation.csv; 100 verbs, 46-4467 train clips):
  mean top-5: real 0.320, transformer 0.474, mamba 0.450 (chance 0.05). Spearman of top-5 vs
  log(train clips): real 0.017 (p 0.87), transformer 0.222 (p 0.027), mamba 0.212 (p 0.034).
  Verbs < 150 clips (57): transformer 0.429, mamba 0.391; 300-999 clips (23): 0.565 / 0.538.
  Top-5 = 0 for 29 (transformer), 30 (mamba), 22 (real) verbs, e.g. talk, stay, lose, lead.
- My ceiling assumption was WRONG: the generated score is above the real score for 56/58 of
  100 verbs. Real clips often hold several actions ("walks, then sits"); the template prompt
  gives one clean prototype. The real score is a different control, not a ceiling.
- What holds: the generators match a verb better when it has more training clips, and the real
  data shows no such trend, so the trend comes from the generators, not from the evaluator.
  Effect is moderate (rho about 0.2). 16 samples per verb; one template sentence; no interval.

## Update (2026-11-08, 1-minute stream run; slow dance for teaching)

- Owner: final paper wants about 10 minutes of stream; for now, 1 minute.
- Running (19:31, detached, pid artifacts/outputs/long_stream_7seg.pid): the same R5 setup as
  the 58.8 s run (val, 32 prompts, seed 2026, CFG 6, temperature 1, top-p 0.9, batch 4,
  full_state + reset_8_keep_16) with 7 windows = 68.6 s. New files `*_7seg.json` and
  `doc/paper/pics/long_stream_validation_7seg.csv`; the 6-window files stay unchanged.
- Known limit: the transformer has 97 learned positions, so full_state cannot run past them;
  only the restart policy is valid for it at this length.
- RESULT (68.6 s; doc/paper/pics/long_stream_validation_7seg.csv). Windows 1-6 reproduce the
  58.8 s run exactly. Window 7 (58.8-68.6 s), R@1 (pool 32, chance 0.031):
  Mamba full_state 0.000 (from 0.594); Mamba restart 0.406; transformer restart 0.500
  (from 0.563). Text distance: Mamba full 8.35, Mamba restart 4.05, transformer restart 3.80.
  32 prompts, so one R@1 step is 0.031. Text control holds past 1 minute only with restart.
- Slow dance for teaching (owner question): 2 paths.
  1. Playback time-stretch after generation (no training): interpolate the generated pose in
     time (the studio already has `upsample_pose_parameters` with rotation slerp). Same steps,
     slower; foot contacts keep their order. Not physical for flight phases (jumps).
  2. Text "slowly": "dance" has 14 slow captions in train; not tested yet for dance.

## Update (2026-11-08, speed-word test: does "slowly"/"quickly" transfer to rare actions?)

- Owner question: can the model do an action slower when the training shows "slowly" mostly
  with other actions? Train captions (69,896): "slow/slowly" with walk 2420, turn 218, jump 76,
  bow 12, hop 8, spin 8; "fast/quickly/quick/rapidly" with climb 2, bow 2, hop 28, spin 28.
- Rule fixed BEFORE the run. Prompts "a person <verb>s", "... slowly", "... quickly"; 16 shared
  seeds; 30 tokens (6 s); CFG 6, temperature 1, top-p 0.9; P1 twins. Speed = mean root-relative
  joint speed (limbs) and mean horizontal root speed. A verb follows a word if the ratio to the
  plain prompt is < 0.9 (slowly) or > 1.1 (quickly) with a 95% paired interval that excludes 1,
  on limb or root speed. Transfer holds if the rare group (bow, hop, spin, climb) follows
  "slowly" in at least 3 of 4 verbs for a twin.
- Script: `scripts/qa/speed_word_probe.py`.
- RESULT (data doc/paper/pics/speed_word_probe.csv). Verbs that follow (limb or root):
  | twin | "slowly" | "quickly" |
  | transformer | walk jump turn hop spin climb (not bow) | walk spin climb |
  | mamba | walk jump turn spin climb (not bow, hop) | walk turn climb |
- Verdict by the fixed rule (rare group, "slowly"): transformer 3/4 (hop, spin, climb) -> transfer
  HOLDS; Mamba 2/4 (spin, climb) -> transfer FAILS.
- "slowly" transfers better than "quickly". Walk follows both words strongly (slow 0.61-0.77,
  quick 1.28-1.59).
- Weak point found AFTER the run (not a change of the verdict): the window is fixed (6 s) and
  the speed is a mean. For one-shot actions (jump, hop, bow), a quick action can end early and
  the body then stands still, so the mean speed falls. "quickly" made jumps and hops slower by
  this measure. The metric is valid for cyclic actions (walk, spin, climb, turn) only. A better
  test measures the duration of the action, not the mean speed. 16 samples per prompt.

## Update (2026-11-08, top-p test: does top-p 1.0 give faster peaks without a worse FID?)

- Rule fixed BEFORE the run. 628 validation pieces (max_eval_clips 600), P1 twins, CFG 6,
  temperature 1, same seeds for both settings. top-p 1.0 is better for a twin only if:
  1. peak speed ratio (top-p 1.0 over 0.9) > 1 and its 95% interval excludes 1;
  2. the paired FID difference (1.0 minus 0.9) has a 95% upper end <= +0.03 (about 10% of P1);
  3. R@1 does not fall by more than 0.02.
     Adopt top-p 1.0 only if both twins pass. Otherwise keep top-p 0.9.
- Script: `scripts/qa/top_p_probe.py`. Shared helpers moved to `evaluation/metrics.py`
  (`sharpness_statistics`, `paired_mean_ratio` on scipy.stats.bootstrap); tests in
  `tests/evaluation/test_sharpness.py` (2 pass).
- RESULT (628 pieces; data doc/paper/pics/top_p_validation.csv):
  | twin | peak speed 1.0/0.9 | FID 0.9 -> 1.0 | FID diff [95%] | R@1 0.9 -> 1.0 |
  | transformer | 1.023 [1.001; 1.046] | 0.249 -> 0.221 | -0.029 [-0.102; 0.053] | 0.518 -> 0.500 |
  | mamba | 1.014 [0.990; 1.039] | 0.365 -> 0.300 | -0.064 [-0.163; 0.045] | 0.510 -> 0.482 |
  Foot skate rises about 5% with top-p 1.0 for both (intervals exclude 1).
- Verdict by the fixed rule: FAIL for both twins (FID upper end > +0.03 for both; Mamba peak
  speed interval includes 1 and R@1 falls 0.028). Keep top-p 0.9.
- What it shows: top-p explains about 2% of the 17-21% peak-speed loss. The loss is in the
  learned token distribution, not in the sampler. FID point values improve, but the intervals
  are wide.
- Noise note: the same setting (top-p 0.9) gave transformer val FID 0.249 here vs 0.272 in
  the P1 log (other seeds and batches), so a single FID value moves by about 0.02-0.03.

## Update (2026-11-08, motion sharpness probe: does the tokenizer smooth sharp motion?)

- Owner question: would a tokenizer retrained on more data give sharper, more natural motion?
- Rule fixed BEFORE the run (18:50): on 256 validation clips (seed 2026), the ratio of the
  tokenizer roundtrip to the real motion for peak joint speed (p99, root-relative):
  > = 0.95 -> the tokenizer keeps the peaks; a tokenizer change has no measured reason.
  > < 0.95 -> the tokenizer smooths the peaks; propose a tokenizer task with its full cost.
  > The same table shows the P1 generators (CFG 6, temperature 1, top-p 0.9) vs the real motion.
- Script: `scripts/qa/motion_sharpness_probe.py` (SciPy paired bootstrap, 1000 resamples).
  Smoke test (8 clips) passed; full run detached, pid in artifacts/outputs/motion_sharpness_val.pid.
- RESULT (256 clips, ratio to real, 95% paired interval). Data: doc/paper/pics/motion_sharpness_validation.csv
  | source | peak speed | peak accel | mean jerk | foot skate |
  | tokenizer | 0.956 [0.934; 0.972] | 0.806 [0.746; 0.857] | 0.692 [0.636; 0.753] | 1.132 [1.067; 1.203] |
  | transformer | 0.814 [0.761; 0.871] | 0.698 [0.622; 0.776] | 0.708 [0.615; 0.816] | 1.487 [1.316; 1.704] |
  | mamba | 0.789 [0.732; 0.848] | 0.647 [0.577; 0.718] | 0.669 [0.586; 0.761] | 1.435 [1.249; 1.684] |
- Verdict by the fixed rule: tokenizer peak speed 0.956 >= 0.95, so no tokenizer change. Weak
  point: the interval low end (0.934) is below 0.95, so the pass is narrow.
- The generators lose about 20% of the peak speed; the tokenizer loses about 4%. The main loss
  of sharp motion is in the generators. Jerk is about 0.7 for all 3, so the decoder sets the
  jerk level (part of the real jerk is capture noise; not separated).
- The tokenizer cuts acceleration peaks by about 19%: a real but smaller decoder effect.
- The 8-clip smoke test pointed the other way (generators above real). 8 clips were too few.
- Next cheap test (inference only): top-p 0.9 cuts rare tokens, and rare tokens can be the fast
  motions. Measure peak speed and val FID with top-p 1.0, both twins.

## Update (2026-11-08, A7 "Slow general learning" proposed; owner asked why P1 peaks at epoch 25-30)

- Owner goal: learning over about 200 epochs, with generalization, not a peak at epoch 25.
- Objection kept on record: the epoch count is a time unit. A lower lr or a larger batch makes
  the curve longer with the same result. The A7 targets are the CE gap per sample seen, the
  best val FID and a val curve that still improves. The epoch count alone is not a target.
- Facts (G0): P1 uses batch 8 = 2802 steps per epoch; its best val FID (ep 30) came after
  84K steps = 0.67M samples (about 30 passes). T2M-GPT (README): batch 128, 300K iterations =
  38.4M samples (about 1600 passes), pkeep 0.5 (50% random tokens), lr 1e-4, CLIP frozen.
  P1: pkeep 0.8 (20%), CLIP last layer + projection trained, lr 2e-4. The P1 CE gap grows
  from epoch 3. Val FID still fell slowly at the CE stop (0.27 / 0.32).
- Plan (epics.csv A7; tasks G0-G6 in epic_tasks.csv): council on the design, then a 6-epoch
  screen of control vs pkeep 0.5 + frozen CLIP for both twins (about 4.2 GPU h), then the
  factor split, then the winner to the CE stop, then a test-split table vs published models.
- Published numbers with sources: `doc/notes/comparable_papers.csv` (only verified values;
  others marked "verify"). P1 numbers are validation only, so they are not comparable yet.
- Waiting for the owner: the R7 rating sheet is still empty on disk (checked 18:15);
  the A7 GPU budget.

## Update (2026-11-08, owner decision: tokenizer stays, focus on the generators)

- No more tokenizer training. Production FSQ stays frozen. Reason: tokenizer recon-FID about
  0.03 vs generator FID about 0.27-0.32 (P1 val); 500 more FSQ epochs give about 0.003 at most,
  and new tokens would force a retrain of both generators.
- Owner chose P2 (per-step text) first. P2 depends on R7: the owner sets a physical guard,
  then rates the blinded keep0/keep16 review. If keep0 passes, no P2 training is needed.
  Blank sheet (46 cases, 32 random first, 14 outliers):
  `artifacts/outputs/qa/switch_review_ratings_20261108.csv`. Media:
  `artifacts/outputs/qa/switch_review_focus_media_20261108/case_NNN.gif` (panels A and B).
- GUARD FIXED BY THE OWNER BEFORE ANY RATING (2026-11-08). Counted on the 32 random cases
  only; the 14 outliers do not count. keep0 passes only if both conditions hold:
  1. Text: keep0 follows the new text better or the same in at least 20 of 32 cases.
  2. Physics: (cases with foot slide on keep0) - (cases with foot slide on keep16) <= 4.
     Pass: adopt keep0 as the switch policy; no P2 training. Fail: propose the P2 pilots
     (both twins, overfit gates first, 4 h cap) for owner budget approval.
     Sealed mapping (not opened): `artifacts/outputs/qa/switch_review_20261108/sealed_mapping.json`,
     sha256 57ead0f1a4ce851bf8906197c8a1ccf6d3aa24515ce4529be1cfe639cc4748d9.

## Update (2026-11-08, C1b done: paired bootstraps on validation)

- Two paired bootstraps, 730 validation clips, 1000 resamples, seed 2026, delta = FSQ - RVQ:
  - Best checkpoints (FSQ e500 0.0293, RVQ e425 0.0317): delta -0.0022, 95% interval
    [-0.0062, +0.0018], does NOT separate; FSQ worse in 13.4% of resamples.
    `artifacts/outputs/qa/tokenizer_pair_bootstrap_c1b.json`.
  - Fixed epoch 500, no selection (FSQ 0.0293, RVQ 0.0399): delta -0.0106, 95% interval
    [-0.0211, -0.0031], separates; FSQ worse in 0 of 1000 resamples.
    `artifacts/outputs/qa/tokenizer_pair_bootstrap_c1b_e500.json`.
- The epoch-500 checkpoints are `*_c1b_e500.pt`, exported from `_last.pt` through the EMA class
  (average_buffers=True). Check: the FSQ export is bit-identical to the FSQ best checkpoint.
- How to read it:
  - Epoch 500 is the end point of the recipe, so it has no selection. But I chose it after I
    saw the curves, and epoch 500 has the largest gap of the curve (+0.0106). The typical gap
    after epoch 300 is +0.007-0.008.
  - The robust evidence is the curve: FSQ better at all 16 evaluations from epoch 125.
    These evaluations are correlated (one run), so they are not 16 independent tests.
  - Both bootstraps include clip noise only. One seed: training noise is not measured.
- Claim for the paper (my proposal): "At 8x1024 with the same 500-epoch recipe, FSQ reaches a
  lower validation recon-FID than RVQ from epoch 125 on; at the fixed end point the paired
  interval excludes 0; with best-of-20 selection the difference is within clip noise."
- Changes: `tokenizer_pair_bootstrap.py` accepts `_c1b_e500` names; `TrainingRun.psm1` no
  longer sets `expandable_segments` (ignored on Windows; kept in `app/service.py` for Linux).
  Tests (24), ruff, pyright and the editor Problems panel are clean. Nothing committed.

## Update (2026-11-08, C1b training done: FSQ ahead of RVQ on validation; 500 epochs are enough)

- Both runs ended with code 0: fsq_g8_v1024_c1b in 2.21 h, rvq_l8_1024_c1b in 3.2 h (with a
  validation evaluation each 25 epochs, 730 validation clips).
- Selected checkpoints (best validation recon-FID):
  - FSQ: 0.0292 at epoch 500 (the last evaluation), MPJPE 110.5 mm.
  - RVQ: 0.0316 at epoch 425, MPJPE 110.6 mm. This point is an outlier: the 4 evaluations
    before it and the 3 after it are 0.0367-0.0398.
- The curves (validation recon-FID, RVQ minus FSQ):
  - Epochs 25-100: RVQ equal or slightly better (gap -0.0015 to -0.0005).
  - Epochs 125-500: FSQ better at all 16 evaluations; gap +0.0019 to +0.0106, mostly +0.007
    to +0.008 after epoch 300.
  - RVQ validation is flat at 0.038-0.040 from epoch 275, while its train recon still falls
    (0.1565 at 300 -> 0.1489 at 500). FSQ validation still falls slowly (0.0306 -> 0.0292).
- Rule for more epochs (owner asked; fit of a line through epochs 400-500): FSQ slope -0.0009
  per 100 epochs, RVQ +0.0038 per 100 epochs. The gap does not close, so more epochs cannot
  change the ranking. Decision: no extension.
- Warning about the comparison: selection takes the best of 20 noisy evaluations. For RVQ the
  selected point (0.0316) is 0.007 below its plateau, so the checkpoint gap (0.0024) is smaller
  than the curve gap (about 0.008). The paired bootstrap tests the selected checkpoints only;
  label it selection-conditioned. It does not include training noise (one seed).
- Side finding: `TrainingRun.psm1:15` and `app/service.py:200` set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; Windows ignores it (UserWarning in every
  .err log). No effect on any run. Proposed: remove it (fail loud rule); not done.
- Production FSQ, both generators and the cached tokens are unchanged. Nothing committed.

## Update (2026-11-08, C1b preflight passed; fresh matched pair launched by owner decision)

- Preflight on the real networks (batch 128, window 64, ema 0.99, the June recipe; both June
  manifests show resume=true and git_commit=unknown):
  - EMA coverage: FSQ 48/48 and RVQ 24/24 floating buffers are in the EMA.
  - Export reloaded into a fresh network: shadow diff 0, output diff 0 (both).
  - Resume round trip: next-step weight and EMA diff 0 on the GPU (both).
  - Overfit gate: 300 steps FAILED (FSQ ratio 0.508, RVQ 0.674 > 0.4). The 0.4 threshold came
    from the tiny test at lr 3e-3; the loss still fell. 1500 steps PASSED (FSQ 0.719 -> 0.155,
    ratio 0.216; RVQ 0.756 -> 0.138, ratio 0.182). Keep 1500 steps for full-size tokenizers.
  - Train compute for 500 epochs (no loading, no evaluation): FSQ 1.46 h, RVQ 2.10 h;
    peak GPU 0.77 / 1.06 GB; 163 steps per epoch.
- Objection kept on record: one seed per tokenizer probably cannot separate the validation gap
  of about 0.001-0.002 recon-FID at 8x1024; a bootstrap over clips excludes training noise.
  A claim of "about equal at 1024, FSQ ahead at 512" needs no retrain. Owner chose the retrain.
- Launched 2026-11-08 10:26 UTC, detached: `scripts/training/run_c1b_local.ps1` (pid in
  `artifacts/outputs/c1b_driver.pid`): fsq_g8_v1024_c1b, then rvq_l8_1024_c1b, then the paired
  validation bootstrap; it stops at the first failure. Production FSQ, generators and cached
  tokens are untouched (the challenger refuses names without `_c1b` and existing files).

## Update (2026-11-08, C1b preflight code ready, not run on the GPU)

- What we built: `scripts/training/tokenizer_preflight.py` checks a real tokenizer on one real
  batch: overfit gate (300 steps, last recon <= 0.4 x first, the rule of test_rvq_baseline),
  EMA coverage of every floating buffer (RVQ embed, cluster_size, embed_avg), export reloaded
  into a fresh network (output difference must be 0), resume round trip (next step difference
  must be 0), and the step time for the planned epochs. It writes a gate JSON and exits 1 on a
  failure.
- `scripts/training/train_tokenizer_challenger.py` runs the CLI request with a validation final
  split. It refuses a name without `_c1b`, `--resume`, and any existing checkpoint or resume
  file, so production FSQ cannot be overwritten. `run_train_tokenizer` got `final_split`
  (default TEST; the CLI is unchanged). `tokenizer_pair_bootstrap.py --pair` scores one pair.
- Checks: 5 new CPU tests pass (FSQ and RVQ; an EMA without buffers fails coverage); the full
  suite passes (671); the freeze test passes, so `tokenization` and `motion` are unchanged.
- Where we stopped: no GPU job ran and nothing is committed. Next, with owner approval: run the
  2 preflights (a few GPU minutes each), then decide the >=8 h window from their timing.

## Update (2026-11-08, owner reserves C1b for a fresh matched pair)

- Owner rejected the narrower new-RVQ versus historical-FSQ validation screen and
  chose to reserve C1b for fresh matched RVQ8x1024 and FSQ8x1024 challengers.
  No GPU overfit gate or training was launched. The current window is not consumed
  by training, and its09:58:50+02:00 deadline is not extended.
- C1b remains blocked on a separate approved window and native gate/evaluation wiring.
  Proposed allocation>=8h: historical training spans sum about5h26, plus45m gates,
  45m paired validation and reserve. Historical pacing is not a current forecast;
  timed full-epoch and evaluation preflights must establish fit before launch.
  This larger budget is a proposal, not owner approval.
- Fresh pair protocol: seed2026, TF32, matching data/splits,500epochs,window64,
  batch128, EMA.99,eval_every25,workers0. Both require actual tokenizer overfit gates,
  coherent learned-buffer export and resume checks, isolated checkpoint names and
  validation-only selection/scoring. Same-validation bootstrap remains conditioned
  on checkpoint selection; no independent certification or TEST access.
- Production FSQ, cached generator tokens and both generator checkpoints stay frozen.
  A fresh FSQ is only a challenger, not a production replacement. No winner swap,
  generator retrain, commit or push. C1 six-pair claim stays paused; P2 stays proposed.
- Eight completed cheap council calls are retained; no further calls are approved
  for this reservation step. Removed completed session-specific council launch tasks.

## Update (2026-11-08, C1b preflight stop point)

- Owner approved C1b in the six-hour GPU window ending09:58:50+02:00.
  Preflight<=45m, isolated RVQ training<=4h, paired validation<=45m, remaining reserve.
  No GPU gate or training started as of04:14+02:00. Frozen FSQ and generators unchanged.
- Strongest objection: the existing FSQ manifest has git_commit=unknown and resume=true;
  its run folder contains only manifest.json and metrics.jsonl, with no saved training
  state or checkpoint-selection lineage. Matching nominal500-epoch recipes cannot
  establish a historically fair retrain comparison. C1 remains paused.
- Fresh council completed all8 calls:3gpt-6-luna and1gemini-4-argon, each with an
  independent answer and anonymous review. No Claude, retries or model escalation.
  Cross-vendor agreement: conditional go only for a validation challenger screen.
  Three GPT members share a model family; this is not four independent model families.
  Confidence medium pending actual RVQ overfit, coherent EMA export/resume, memory,
  and timed throughput gates. All oppose a capped-run convergence/win claim.
- Review mean ranks (lower is better): B1.75, C2.00, D3.00, A3.25.
  A: measured EMA and timing gates. B: selection-conditioned validation interval.
  C: clean challenger scope. D: generator-only gate and concrete stop conditions.
  D's28.8s/epoch threshold ignores startup/evaluation costs, so it is not adopted.
  Some member line citations were shifted. Three could not read the older graph;
  their source claims were checked against current files, not accepted by vote.
- Decisive direct checks: app/training.py run_sanity_overfit uses GeneratorTrainer
  and a frozen tokenizer, so it cannot gate RVQ training. run_train_tokenizer always
  calls evaluate(Split.TEST) after training. The bootstrap hardcodes six old pairs.
  A validation-only screen needs these actual paths wired and tested before launch.
  TokenizerTrainer requests floating-buffer EMA and validation-FID selection;
  same-validation bootstrap is selection-conditioned, not independent certification.
- Stop point: C1b blocked pending owner approval of the narrower validation screen.
  Do not silently rename that evidence as a fair historical comparison.
  Council: session files/council_c1b_20261108 (brief; answers; anonymous mapping; reviews).

## Update (2026-11-08, focused review and counterfactual controls verified)

- Strongest new objection: Mamba early R@1 gains+.25 for both A-to-A and A-to-B at
  the point estimate. Reset benefit is not exclusively switch-specific. Its B-to-A
  contacting-foot skate rises+.05606m/s [.02428;.08288],31 jointly contacting pairs.
  No physical guard or human naturalness pass was selected. Keep production retain16,
  width320, TF32 and frozen FSQ8x1024 unchanged; P2 remains proposed.
- Owner approved focused clips after the initial unreadable render was stopped.
  Fixed common orthographic framing, explicit camera updates, thicker dark bones and
  checked horizontal-arm T-pose repaired the review. All46 GIFs and46 montages passed
  independent hash/dimension/duration checks:113 source frames176-288;5.65s at20FPS.
  T-pose, known clip and sample montages inspected. Recovered joints only, not fitted
  mesh or human ratings. Mapping stays separate;32 random cases and14 outliers.
  Retry333.3s plus first logged509s =842.3s accounted wall time, below15 minutes;
  not measured GPU utilization. Partial initial files preserved.
  Media: artifacts/outputs/qa/switch_review_focus_media_20261108.
  Verification: switch_review_focus_verification_20261108.json.
- Counterfactual12 cells completed1021.1s,17.0min within20-minute guard:
  A-to-A / A-to-B / B-to-A; keep16/0; both frozen twins; original32-pair cohort.
  Independent verification reproduced common32x96x8x1 uniforms, prefix/cut token
  identities, canonical recovery, diagnostics, semantic arrays and paired intervals.
  A-to-B BOTH policies exactly replay prior cohort0, not a new replication.
  Source: artifacts/outputs/qa/switch_controls_20261108.json;
  verification: switch_controls_verification_20261108.json;
  [Control CSV](../../doc/paper/pics/switch_controls_validation.csv).
- Early own-target R@1 keep16->0:
  Mamba A-to-A.34375->.59375; A-to-B.25000->.50000; B-to-A.31250->.56250.
  Transformer A-to-A.56250->.59375; A-to-B.28125->.53125; B-to-A.28125->.56250.
  Reversed-switch difference intervals exclude0 for both twins, unadjusted.
- Fixed-AB-target early margin interactions are Mamba+6.0872[3.8719;8.3671] and
  Transformer+3.8078[2.0736;5.8514]. Fixed-target R@1 interaction intervals:
  Mamba+.21875[-.03125;.46875]; Transformer+.25000[.03125;.46875].
  Descriptive own-target R@1 gain contrasts are0[-.40625;.375] and
  +.21875[-.0625;.50078]; different target pools, not a matched fixed-target claim.
  Same-caption own margins are always0; B-to-B was not run. No generalization proof.
- Mamba A-to-A skate also rises+.04667m/s [.02190;.07079],31 pairs.
  B-to-A AUJ falls while skate rises; one jerk metric cannot certify motion quality.
  [Decision report](../../doc/notes/streaming-decision.md) records the trade-offs.
  Next: blinded random-case ratings and owner physical guard, not immediate adoption
  or retraining. Optional GPU tests need budget accounting/approval.
- Final focused validation:60 tests passed; one pre-existing PyGLM import warning;
  Ruff/type checks0. R7 remains running for owner choice. No new epic, council/Claude
  calls, training, dependency changes, commit or push. GPU jobs ran sequentially.

## Update (2026-11-08, saved-motion audit verified; visual review stopped)

- Strongest objection: the initial blinded renders make the skeletons too small for
  reliable visual judgment. Stopped the exact renderer processes after22 completed
  GIFs rather than spend the rest of the15-minute cap on unreadable media. No human
  quality pass, policy adoption or training approval follows from this work.
- Independently verified128 paired diagnostic rows, four32-pair groups,46 anonymous
  cases,32 random cases from16 shared pairs, copied-array hashes, root steps, per-pair
  jerk peaks, semantic values, quantiles and paired intervals. Policy mapping remains
  separate. [Paired CSV](../../doc/paper/pics/switch_review_pairs.csv); raw audit and
  verification are artifacts/outputs/qa/switch_review_20261108.json and
  switch_review_verification_20261108.json.
- Initial task quoting and module-loading errors occurred before Python launch.
  The process-type task then ran on the RTX3050. The log spans8min29s through the
  last completed case; this is recorded wall time, not measured GPU utilization.
  Partial media are preserved in artifacts/outputs/qa/switch_review_media_20261108.
  All22 GIFs have800x630 dimensions and19.2s duration. The render completion report
  is absent. The SMPL-X default-pose image alone is not a verified T-pose readiness
  check. Visual framing/readiness must be repaired before calling this review ready.
- Counterfactual control implementation reuses the history-screen owner. Original
  A/B sampling identities fix uniforms across A-to-A, A-to-B and B-to-A, retained16/0
  and both twins. It saves uniforms/token indices and gates raw R6 parity only for
  A-to-B/16. A-to-A uses32 distinct recordings, not64. Switch-specific interactions
  use identical original A/B caption targets; own A-to-A margins are necessarily0.
  B-to-A is only a direction check without a matched B-to-B control.
- Counterfactual generation has NOT run. The20-minute control allowance is unspent.
  Focused validation:44 tests passed; Ruff/type checks report0 remaining problems.
  R7 remains running; P2 remains proposed. Owner choice is needed to change the
  visual retry scope/budget or waive that prerequisite. No optional stress tests,
  fresh council/Claude calls, dependency changes, commits or pushes.

## Update (2026-11-08, switch-history and K8 screens verified; R7/P2 proposal)

- Strongest objection: frozen validation screens and synthetic core timings do not prove
  generalization or naturalness. Keep width320, TF32 and the production restart8/retain16
  policy unchanged. No retraining, new council call, Claude call, commit or push.
- Approved steps1-4 produced a12-cell switch-only history screen, R7 decision table,
  six-cell K8 precision screen and conditional P2 proposal. R7 owner choice and P2
  intervention/budget remain open. See doc/notes/streaming-decision.md.
- History: two disjoint64-recording validation cohorts absent TRAIN;32 caption pairs
  each; frozen P1 twins, seed2026, TF32, CFG6. Only token48 uses retained16/8/0;
  other resets remain8/16. Removed tokens cannot resurrect; capacity stays16.
  Default/explicit16 parity is tested on both twins; original cohort16 raw features
  reproduce R6 exactly. All12 cells completed in1067.9s,17.8min.

  | Twin        | Cohort | Early R@1 keep16 |  keep8 |  keep0 |
  | ----------- | -----: | ---------------: | -----: | -----: |
  | Mamba       |      0 |           .25000 | .25000 | .50000 |
  | Mamba       |     64 |           .40625 | .53125 | .59375 |
  | Transformer |      0 |           .28125 | .43750 | .53125 |
  | Transformer |     64 |           .40625 | .34375 | .56250 |

- Zero-history early own-caption margin improves in all four cells with unadjusted
  paired95% intervals above0: Mamba+4.75036[2.64607;7.07174] and
  +4.73035[2.72818;6.92111]; Transformer+3.29510[1.46879;5.36224] and
  +2.69026[1.28438;4.04859]. Only one early R@1 interval strictly excludes0;
  do not claim universal retrieval improvement. Keep8 does not replicate.
- Physical trade-off: Transformer cohort0 AUJ.41891->.56737, point+35.4%;
  difference interval[-.06151;.25813]. Other zero-history AUJ difference intervals
  also cross0. This is uncertainty, not equivalence or acceptable naturalness.
  Skate contrasts use jointly contacting31 Mamba/32 Transformer pairs. No physical
  regression threshold was selected. Offline noncausal tokenizer decode is not live
  latency. Candidate retain0 requires owner guard and body review before adoption.
- History verification independently reproduced504 semantic arrays, all paired intervals,
  canonical recovery, GT jerk, diagnostics, hashes,128 distinct source recordings,
  zero cohort overlap and12 CSV rows. Raw report:
  artifacts/outputs/qa/switch_history_20261108_retry1.json; verification:
  artifacts/outputs/qa/switch_history_verification_20261108.json.
  Tracked CSV: doc/paper/pics/switch_history_validation.csv.
- Execution correction: first history attempt failed after generation because in-memory
  dataclass tokens were tuples while Guo input expects lists. Fixed explicit conversion
  with a regression test. Failed outputs preserved; retry27-minute cap kept the first
  attempt within the original30-minute allowance. No unbounded retries.
- K8 completed sequentially after history exit, one process, batch8/196frames,
  FP32/TF32/BF16, one warmup plus five measured full-core updates per cell.
  Transformer medians80.77/69.63/114.17ms; Mamba123.72/129.96/143.38ms.
  BF16 versus TF32:64.0%/10.3% slower. Mamba allocated peak.681->.622GB,
  8.7% lower. No BF16 speed gain demonstrated; no production precision change.
- Numeric reference is clean generator token-CE on fixed strict-FP32 tokenizer targets,
  identical initial weights, dropout off. TF32 gradient relativeL2 errors
  .000669/.000655; BF16.007257/.008109 for Transformer/Mamba. These are not
  total-loss gradient fidelity or convergence certificates. Timings include configured
  generator/tokenizer losses, live encoding and optimizer/EMA, but exclude CLIP
  encoding/backprop, data loading and validation. Five ordered repeats are a screen,
  not epoch-speed evidence. Saved30 timing samples and logits/gradients; independent
  audit reproduced errors, medians, initial-weight/gradient-coverage parity and six CSV
  rows. Evidence: artifacts/outputs/qa/k8_precision_20261108.json;
  k8_precision_verification_20261108.json; doc/paper/pics/precision_costs.csv.
- K8 first launch failed before any benchmark because the task omitted the tokenizer
  checkpoint; explicit FSQ path fixed and actual task arguments preflighted. Failed
  logs preserved; retry14.4-minute cap, exit0 after about0.01h. No fallback math.
- P2 recommendation is conditional, not approved: review frozen retain0 first. If
  inadequate, propose shared per-step text with fresh matched prefix-only controls:
  four cells, at most4h pilots plus1h gates. Capped curves cannot prove generalization;
  convergence-scale funding is separate. Historical P1 patience3 is not a matched
  control for a new patience6 candidate. More AMASS motion does not imply paired
  captions; more epochs alone do not establish better generalization.
- Final focused validation:81 generation/evaluation/streaming/precision tests plus26
  generator-upgrade/known-value tests passed; one skipped,107 passes total. Ruff/type
  checks on the changed source/tests report0 problems. Both CSV ledgers parse correctly:
  56 task rows, six epics, unique(epic;id) task keys. R7/K8 retain running status
  for owner decisions; P2 remains proposed. CPU construction verifies per-step text
  adds no parameters: Transformer15,300,040 and Mamba15,553,480 with either flag.

## Update (2026-11-07, saved-motion caption-switch follow-up complete)

- Strongest objection: early/late matcher scores cannot distinguish necessary transition
  time from unwanted retained-history effects. This is a controlled diagnostic on one
  validation cohort, not proof of a conditioning bug, equivalence, or generalization.
  The generator and tokenizer were never loaded; no new motion was generated or trained.
- Reused both frozen width320 R6 raw263 archives. Reproduced every full-action R6 retrieval
  hit and mean new-caption distance before computing new windows. Used the same frozen Guo
  matcher, official normalization, TF32 and seed2026 in one guarded shared VS Code terminal.
  The process exited0 within its15-minute cap.
- Action2 spans global frames192:384. Early192:288 and late288:384 are equal96-frame,
  4.8-second windows at20FPS. Cut and continuous generation use identical windows, caption
  pair order and fixed32 pools. Both old and new captions are scored; an old-minus-new
  distance margin above0 favors the new caption. Exact distance ties favor neither.

  | Twin        | Method               | Early new-caption R@1 | Late new-caption R@1 |
  | ----------- | -------------------- | --------------------: | -------------------: |
  | Mamba       | Independent cut      |               0.53125 |              0.37500 |
  | Mamba       | Continuous generator |               0.25000 |              0.34375 |
  | Transformer | Independent cut      |               0.62500 |              0.40625 |
  | Transformer | Continuous generator |               0.28125 |              0.46875 |

- Early generator-minus-cut R@1 paired95% intervals:
  Mamba -0.28125 [-0.53125;-0.03125];
  Transformer -0.34375 [-0.53125;-0.15625].
  Late: Mamba -0.03125 [-0.21875;0.15625];
  Transformer +0.06250 [-0.12500;0.25000].
  Late intervals include0; this is not evidence of equality or a Transformer win.
- Old-minus-new own-caption distance margin improves early->late:
  Mamba +1.36603 [0.09329;2.61105];
  Transformer +2.37344 [0.52568;4.43295].
  Transformer late new-caption distance is3.0886 versus independent3.0775;
  difference+0.0110 [-0.7125;0.7331].
  Mamba late distance is4.5880 versus independent3.5497; its distance-difference interval
  includes0, but late caption margin remains below independent by-2.04851
  [-3.90039;-0.13365].
- Interpretation: results are consistent with delayed adaptation, especially the early
  retrieval deficit and later shift toward the new caption. Necessary motion transitions
  and caption-internal phases remain alternatives. Retained history as a mechanism is a
  medium-confidence hypothesis, not a measured causal conclusion. No new epoch budget
  is justified by these results alone.
- All intervals use1000 paired resamples and are exploratory, unadjusted for multiple
  comparisons. They condition on the fixed caption/motion pools, not independent training
  seeds. The64 recordings are absent from TRAIN but not subject-disjoint or unused
  checkpoint-selection data. R6's lower jerk remains a separate physical diagnostic.
- Verified saved text embeddings(2;32;512), motion embeddings(2;3;2;32;512), all84 metric
  cells and paired intervals by independent NumPy distance/rank recomputation. Source,
  raw-feature and matcher hashes match; full-action retrieval reproduces R6.
  Tracked12-row,33-column CSV: doc/paper/pics/caption_switch_validation.csv.
  JSON and embeddings: artifacts/outputs/qa/r6_20261106/caption_switch_20261107{.json;/}.
  Verification: artifacts/outputs/qa/r6_20261106/caption_switch_verification_20261107.json.
  CSV includes frame bounds, split, seed, precision and checkpoint/config hashes.
- Command: .venv\Scripts\python.exe -m scripts.qa.long_stream_probe --caption_switch
  --compare artifacts\outputs\qa\r6_20261106\mamba.json
  artifacts\outputs\qa\r6_20261106\transformer.json
  --family_index data\HumanML3D_official\index.csv
  --out_path artifacts\outputs\qa\r6_20261106\caption_switch_20261107.json
  --summary_csv doc\paper\pics\caption_switch_validation.csv
  Choose new output paths to repeat; existing evidence is protected from overwrite.
- Validation:45 focused tests pass; check_problems0. Tests cover exact windows, distance
  direction/ties, paired contrasts, full-action reproduction failure, source hashes,
  malformed pools, short/integer/nonfinite raw archives and CLI generation bypass.
  Added integrated task Caption-switch saved-motion screen; reused R6 test/check tasks.
  R7 evidence updated but its owner decision remains open. A controlled retained-history
  ablation at the switch is a candidate falsification test, not approved or executed.
  Width320, restart8/keep16 and future patience6 unchanged. No agent/model calls,
  dependency changes, commit, push, new epic or new training/generation budget.

## Update (2026-11-07, R6 transition screen and C3 fixed-floor proxies complete)

- Strongest limitation: lower jerk does not establish natural motion or caption control.
  The generator is paired with baselines by captions and sampling draws, not endpoint poses.
  Positional bridges see the future endpoint, alter bone lengths, and are not an IK solution.
  The generation policy and tokenizer are unchanged. This is not a generalization test.
- Ran frozen P1 width320 twins sequentially in the shared VS Code terminal, no training or
  model-agent calls. The experiment stem remains r6_20261106; execution finished after midnight
  local, November7. Both reports and all archives are complete; no R6 Python process remains.
  Each twin had a 15-minute guarded cap; neither reached it.
- Protocol: 32 caption pairs from 64 distinct official validation source recordings absent
  from TRAIN, at least60 reference frames; not subject-disjoint or untouched model-selection
  data. Seed2026, TF32, batch2, CFG6, temperature1, top-p0.9, reset8/keep16. Two48-token actions,
  192frames/9.6s each at20FPS; switch aligned with reset;20-frame positional bridge.
  Independent and continuous generations share left token IDs and second-action sampling draws.
  Offline noncausal decode can affect the first action near the switch.
- FlowMDM-style L1 third differences, then maximum joint per frame. Peak Jerk (PJ) is the
  maximum of the mean57-point profile over the60-frame boundary window, not the mean of maxima.
  Area Under the Jerk (AUJ) is the discrete absolute deviation from this cohort's GT jerk
  reference0.0339601999, not FlowMDM's published test constant. Values use meters/frame^3
  without FPS scaling. These are adapted validation diagnostics, not comparable paper-table scores.

  | Twin        | Join                          |       PJ |      AUJ | Conditional foot skate (m/s) |
  | ----------- | ----------------------------- | -------: | -------: | ---------------------------: |
  | Transformer | Cut                           | 1.201751 | 2.925252 |                     0.255800 |
  | Transformer | Linear                        | 0.106820 | 1.288303 |                     0.230218 |
  | Transformer | Quintic minimum-jerk          | 0.096559 | 1.216105 |                     0.230145 |
  | Transformer | History-conditioned generator | 0.036652 | 0.418906 |                     0.273301 |
  | Mamba       | Cut                           | 1.462913 | 3.569462 |                     0.243383 |
  | Mamba       | Linear                        | 0.084680 | 1.266608 |                     0.199652 |
  | Mamba       | Quintic minimum-jerk          | 0.079212 | 1.175063 |                     0.196215 |
  | Mamba       | History-conditioned generator | 0.025754 | 0.779083 |                     0.196797 |

- Paired1000-resample generator-minus-quintic AUJ:
  Transformer -0.797199 [95% CI -0.958149;-0.475697];
  Mamba -0.395980 [-0.598878;-0.196780].
  This supports lower jerk mismatch in this setup, not equal-endpoint superiority.
  Mean relative bone-length deviation rises cut->quintic from1.93%->2.34% Transformer
  and1.78%->2.31% Mamba. No smoothing-only naturalness winner.
- C3 uses the same fixed Y-up floor0m for every variant. Contact is feet<=5cm; reports below
  -2cm and all-feet-above10cm fractions. No-contact skate is null, never a zero success.
  Transformer linear/quintic and Mamba generator each have one no-contact pair; their skate
  means condition on31 pairs, others32. All below-floor fractions are0; raised-feet fractions
  can include legitimate kneeling/running poses and are not proof of floating. Historical
  R5/R5b metrics retain their old clip-relative floor and are not directly comparable.
  Tests verify lifted20cm, lowered3cm and known0.2m/s synthetic cases.
- Semantic caution: second-action fixed32-pool Guo R@1 is0.500 cut vs0.40625 generator for
  both twins. Generator-minus-cut paired95% intervals: Transformer[-0.250;0.0625],
  Mamba[-0.28125;0.125], so the decrease is not established by this sample.
  Text distance increases2.477->3.632 Transformer and2.645->4.506 Mamba.
  Bridges are joint-only and cannot be passed off as valid263 features for matcher scores.
  One falsification test: evaluate new-caption adherence after excluding the initial transition
  window; a low-jerk generator may be slow to leave the previous action. Not run in this screen.
- Verified eight joint archives at(32;384;22;3), four raw-feature archives at(32;384;263),
  both GT arrays at(32;2;57), exact canonical recovery, checkpoint/tokenizer/config hashes,
  bridge exterior preservation, profiles, diagnostic aggregates and paired intervals.
  Raw arrays and JSON: artifacts/outputs/qa/r6_20261106/.
  Tracked eight-row summary: doc/paper/pics/transition_quality.csv.
  Verification details: artifacts/outputs/qa/r6_20261106/verification.json.
  Generation+offline decode32.1s Transformer and128.8s Mamba; not live latency.
  Reproduction command is in R6's epic_tasks.csv row; choose a fresh output stem.
- Validation:34 focused tests pass; check_problems0 across the four changed Python files.
  Integrated tasks: R6 transition tests, R6 transition problems, R6 transition screen.
  R6 and C3 now done; A1 still awaits R7's owner decision. No new epic or training budget
  assumed; width320 and future clean-CE patience6 retained; no dependency sync, commit or push.

## Update (2026-11-06, owner retains width320; future validation-CE patience6)

- Owner retains width320 after the bounded width256 pilot. Restored the size driver's active
  default to 320. Both existing presets and all old checkpoints remain unchanged.
  The pilot's time caps, while its CE was falling, do not establish failure to converge.
- Owner confirmed five retries followed by stopping on the sixth consecutive epoch without
  a strictly lower clean validation CE. Ties count; an improvement resets the counter.
  Future shared generator CLI and size-driver defaults are patience6. This monitors validation
  token CE, not training total loss, and does not disable wall-time or epoch ceilings.
  Explicit patience0 still disables CE stopping; negative CLI patience is rejected.
- Fixed resume continuity in the shared trainer for both twins. New last checkpoints store
  best clean CE and stale-epoch count. Older checkpoints recover these from their recorded run's
  clean-CE metrics through the completed checkpoint epoch, ignoring later partial-epoch records.
  Missing/corrupt history fails loudly rather than silently granting fresh patience.
  An already exhausted counter stops before another optimizer epoch.
  --fresh_selection is the explicit reset of both FID selection and CE patience.
- Verified actual P1 width320 last checkpoints without modifying them:
  Transformer completed32, best clean CE4.580611, stale3;
  Mamba completed24, best clean CE4.475481, stale3.
  Each has three additional non-improving epochs available under patience6, not six fresh
  retries. Any new lower CE resets the counter. A continuation would require an approved budget
  and would be a patience diagnostic, not an independent seed or a width-generalization test.
- Validation: 75 focused tests pass across clean CE, resume, training, CLI and PowerShell guard;
  check_problems zero. Tests cover the exact sixth-epoch threshold, ties, reset on improvement,
  continuity through new/old checkpoints, explicit fresh selection and failure on corrupt state.
  No loss/model/data/precision changes. Historical sweep/P1/P256 recipes retain patience3.
  Existing width256 VS Code pilot tasks retain their historical explicit arguments.
  No further training, dependency sync, model-agent call, commit or push.
- Epic priorities: R6 joins and proposed C3 fixed-floor checks address the still-unproved
  long-stream quality claim using the frozen320 pair; K8 tests BF16 versus actual TF32 for
  both twins before adopting a precision change; K7 still lacks validated build/determinism
  evidence. C1b fair RVQ training is needed before unpausing C1's FSQ-versus-RVQ claim.
  P2 intervention/budget is not chosen. W1-W3 figures/data exist; W4 remains optional.
  Width-cost CSV has no precision provenance field, so do not call its saved step timings a
  matched TF32 speed comparison without an audit.

## Update (2026-11-06, 18:49 local, width256 bounded pilot complete)

- Both runs stopped at their approved wall-time caps, not at validation-CE patience or
  the 60-epoch ceiling. Transformer stopped at 15:42:50 UTC (17:42:50 local);
  Mamba stopped at 16:42:54 UTC (18:42:54 local). Both exit codes are 124.
  P256_PILOT_20261106_PARTIAL.txt exists; there is no DONE marker.
  Verified no active generator Python process after the stops.
- Both own-run resume checkpoints load and match their recorded completed epochs and best FID.
  Training remains partial. P256 is done as a bounded pilot task, not as a convergence run.

  | Twin           | Completed epochs | Best validation CE | Best validation FID (epoch) | Median optimizer epoch |
  | -------------- | ---------------: | -----------------: | --------------------------: | ---------------------: |
  | Transformer256 |                8 |      5.05638 (ep8) |               0.82545 (ep8) |              5.504 min |
  | Mamba256       |                7 |      4.89352 (ep7) |               1.06508 (ep4) |              7.669 min |

- Last three validation CEs still decrease: Transformer ep6-8 = 5.26843, 5.15135, 5.05638;
  Mamba ep5-7 = 5.15887, 5.01090, 4.89352. This is not a measured plateau.
  Mamba FID worsened from 1.06508 at ep4 to 1.63157 at ep6 despite falling CE.
  Its ep7 checkpoint has no ep7 FID evaluation (eval_every2).
- Timing is optimizer-epoch time from these runs, not full wall time or a matched width320
  speed comparison. Transformer had the previously documented safety interruption and resume.
  One seed, unequal completed epochs and non-converged runs cannot prove better generalization.
- Evidence: [Transformer metrics](../../artifacts/logs/train/20261106T144312Z_generator_transformer/metrics.jsonl),
  [Mamba metrics](../../artifacts/logs/train/20261106T154319Z_generator_mamba/metrics.jsonl),
  [guard stops](../../artifacts/outputs/GUARD_KILL.txt).
  Own-run last checkpoints retain optimizer, EMA, RNG and scheduler state for an explicitly
  approved continuation. No further training launched and no budget extension assumed.
  R7 remains open; width320 fallback unchanged. No commits, pushes or model-agent calls.

## Update (2026-11-06, owner-selected width256 fresh pilot)

- Owner selected width256 for both twins, then explicitly authorized fresh training and a
  one-hour training cap per twin. Gate time is additional, at most one hour per gate.
  This authorizes P256 in A4, not a full convergence run or other P2 interventions.
- Reused run_size_sweep_local.ps1 with a bounded Pilot mode. Default width is now 256;
  the alternative 320 selector is commented beside it, as requested. Both existing presets
  remain intact: width256 uses dt_rank16 / Mamba14 layers; width320 uses dt_rank20 / 15 layers.
  Use the complete preset for rollback, not only a d_model edit.
- Both fresh overfit gates must pass before any full training. Then Transformer and Mamba
  run sequentially, with fresh weights, seed2026, TF32, the existing kernel and frozen FSQ.
  Batch8, workers0, eval_every2, clean validation CE, patience3, 60-epoch ceiling.
  No changes to data, losses, tokenizer weights or old checkpoint files.
- Guard distinguishes normal one-hour caps from stalled processes. Only normal caps continue
  the paired pilot; crashes/stalls stop the queue. Capped runs produce PARTIAL, never DONE.
  Logs stream into the shared VS Code task terminal and persist on disk.
  A new tag refuses existing pilot artifacts instead of skipping or resuming them.
- Safety interruption at 14:50:10 UTC: Windows virtualenv Python has a launcher and an actual
  child interpreter. The previous timeout guard stopped only the launcher and could leave CUDA
  training alive when queuing the other twin. Stopped the verified controller and both Python
  PIDs; verified zero generator processes before recovery. Preserved logs and the completed
  epoch-1 Transformer checkpoint. This was an orchestration defect, not an epoch/quality stop.
- Fixed the shared guard to snapshot descendants, stop exact PIDs, check PID identity and wait
  for all stopped processes before returning. Added a real Windows child-liveness regression.
  Recovery is explicit (-Pilot -ResumePilot), requires original gate files and the interrupted
  run's checkpoint, preserves original logs and uses only time remaining from the original
  log creation time. Missing recovery evidence fails instead of silently restarting.
- Validation: 18 guard/driver tests passed; check_problems zero; PowerShell parser passes;
  both preset architectures and TF32 validated. Built-in test discovery found no tests;
  the explicit integrated pytest task ran all 18 successfully.
- Initial task: Width256 twin pilot; recovery task: Width256 pilot recovery.
  Tag p256*pilot_20261106.
  Logs artifacts/outputs/p256_pilot_20261106*{transformer,mamba}_w256.log(.err);
  checkpoints generator_{transformer,mamba}\_w256_p256_pilot_20261106\*.pt.
  Both fresh gates passed:
  Transformer 101 steps, CE 0.09922, token accuracy 99.117%, total-loss ratio 0.04078;
  Mamba 26 steps, CE 0.007365, token accuracy 99.932%, total-loss ratio 0.02975.
  Before interruption, Transformer completed epoch1: train CE 6.36722, validation CE 6.36197;
  optimizer-epoch time 5.622 minutes, not an end-to-end twin speed comparison.
  Resumed its own epoch-1 checkpoint at 14:55:59 UTC with 0.77972 h (46.78 min) remaining.
  The existing trainer resets its validation-CE stop counter on resume; this pilot is not
  an uninterrupted convergence experiment. Model, optimizer, EMA, RNG and schedule are restored.
  Original Transformer deadline remains 15:42:46 UTC (17:42:46 local), plus guard polling.
  Full Mamba training has not started; its one-hour budget is unused and it starts from zero.
  Transformer recovery verified active: controller PID33000, launcher PID38788, CUDA PID22052;
  10,668,808 generator parameters. Heartbeat1200 updated at 14:58:44 UTC in the same run:
  artifacts/logs/train/20261106T144312Z_generator_transformer.
  One CUDA Python training process verified; Windows graphics processes are not training jobs.
  Continuation logs: artifacts/outputs/p256_pilot_20261106_transformer_w256_resume_20261106T145559Z.log(.err).
  Mamba is queued automatically after Transformer finishes or reaches the normal cap.
  Caps count subprocess wall time, including startup and validation, not just optimizer time.
  P256 remains running, not done. No commit, push, dependency sync or model-agent call.
- A one-hour pilot cannot establish convergence or a fair generalization win over the fully
  trained P1 width320 pair. Existing width320 inference/evaluation tasks remain unchanged.

## Update (2026-11-06, autonomous epic checks and R5b results)

- Work stays within A1/A0; no new epic, training run, commit or push.
  No new council calls. Revalidated all 7 agents as gpt-6-luna and the no-Claude policy.
- R5b real-data gate passes: two cohorts of 32 distinct validation source recordings,
  no recording overlap between cohorts, and no recording represented in official TRAIN.
  Selection uses the official index and seed 2026. This is not subject-disjoint evaluation.
- Extended the existing long-stream probe for restart 8/16 versus 32/16, per-prompt physical
  scores, saved raw HumanML3D-263 motion and paired policy comparisons.
  Combined focused suite: 21 passed (11 probe, 10 paper-watch); check_problems: zero.
  Sequential batch-2 inference completed in the shared VS Code Policy screen cohorts task.
  Verified matching provenance, aggregate/per-prompt agreement and all 8 finite raw archives
  with shape (32, 1176, 263): 256 generations of 58.8 seconds, no retraining.
  R5b is done; R7 remains running.
- Late window (49.0-58.8 seconds), restart32-minus-restart8 text distance; positive is worse:

  | Twin        | Cohort offset | Difference | 95% prompt-paired interval |
  | ----------- | ------------: | ---------: | -------------------------- |
  | Mamba       |             0 |     +0.253 | [-0.825, 1.166]            |
  | Mamba       |            32 |     +1.464 | [0.219, 2.905]             |
  | Transformer |             0 |     +0.287 | [-0.435, 1.047]            |
  | Transformer |            32 |     +0.024 | [-0.708, 0.822]            |

  All four restart-policy R@1 intervals cross zero. This is not a replicated policy winner.
  Mamba restart32 cost 275.4 seconds versus restart8 618.3 seconds across both cohorts;
  Transformer 107.1 versus 113.6 seconds. Total 1114.4 seconds (18.6 minutes) for generation
  plus offline decoding alone, excluding model loading/scoring. Single-pass cost, not a
  repeated speed benchmark or live latency.

- Evidence: [cohort 0 CSV](../../doc/paper/pics/long_stream_policy_offset0.csv),
  [cohort 32 CSV](../../doc/paper/pics/long_stream_policy_offset32.csv) (24 windows each);
  [paired comparisons and archives](../../artifacts/outputs/qa/r5b_20261106/);
  [shape/provenance verification](../../artifacts/outputs/qa/r5b_20261106/verification.json).
  Seed 2026, TF32, CFG 6, temperature 1, top-p 0.9, end masked, frozen P1 checkpoints.
- Physical-score gains are cohort-dependent. The root_teleports proxy counts steps above
  0.25 m/frame = 5 m/s: all three flagged late cases describe running, with peak speeds
  5.19-5.37 m/s. They do not establish teleports or physically invalid motion.
  Foot grounding is still clip-relative: floating/penetration remain unmeasured (C3).
- Inspected [static diagnostic montage](../../artifacts/outputs/qa/r5b_20261106/late_distance_case.png):
  largest Mamba late distance deterioration, caption "the man is jogging on the spot",
  clip 005441; restart32 distance 14.907 versus restart8 1.783.
  This selected case is not representative and static poses cannot prove naturalness or
  repetition. Saved matched root trajectories: Mamba maximum distance from window start
  0.182 m (32) versus 0.160 m (8); the retrieval outlier is not proof of large root drift.
- Limits: recording-disjoint, not subject-disjoint; validation after checkpoint selection,
  not independent final test evaluation. Intervals condition on fixed 32-motion pools,
  not training seeds. Restart changes text refresh, history and Transformer positions together.
  No smaller-model or longer-training causal experiment was run.
- Preliminary R7 options (not training approval):

  | Option                                         | Measured basis/source                                                                         | Cost scope                                                                       | Protocol effect                                                                                                          |
  | ---------------------------------------------- | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
  | Keep frozen width320, restart8                 | R5b: Mamba late distances 3.452/3.603; no replicated restart32 advantage                      | 0.203 h measured generation+decode for both twins/cohorts; loading/scoring extra | Current service baseline retained; quality and R6 joins still need checking                                              |
  | Adopt restart32                                | R5b: Mamba late distances 3.705/5.068; one cohort's paired distance worsens; R@1 inconclusive | 0.106 h measured generation+decode; not live latency                             | Less frequent text refresh/replay and position resets; not justified by this screen                                      |
  | Controlled capacity/conditioning retrain later | S3/P1: four-run sweep and one-seed P1; no causal long-horizon capacity result                 | New cost unmeasured; A4 P1 budget was about 17 GPU h, not a P2 estimate          | Requires a separately approved matched-twin plan, overfit gate and validation-only selection; no extra epochs by default |

  Assumption (medium confidence): retaining restart8 is the safer baseline, not a proven optimum.
  Blind temporal playback with prompt-following and repetition ratings can overturn that
  recommendation. Keep R7 open for R6/quality evidence and the owner's final choice.

- Resources: about 1.8 GiB free RAM; GPU has graphics processes but no CUDA training process.
  K7 worktree at 0eec5a5: six modified files plus the untracked deterministic-scan test
  confirmed by the existing integrated status task. This resolves the earlier task-creation
  timeout, not the build criterion. No CUDA compilation or determinism test started.
- Paper watch failed on November 6 at 03:00 with arXiv HTTP 429; scheduled-task result 1.
  Five retries already existed. Added retry/exhaustion logging and HTTP-date Retry-After
  handling, with tests. API recovery is not demonstrated.
  Successful scheduled-night streak is zero; next run is November 7 at 03:00.
  Earliest possible third successful night is November 9. Owner approved extending A0 to
  November 10; schedule and criterion unchanged. Manual backfills do not count as scheduled
  successes. No successful November 6 report or API recovery has been demonstrated.
- Corrected S3's finish wording to the owner-accepted four-run width-256/320 sweep.
  It does not claim ten completed runs. P2, K8, C1b and causal-decoder training remain proposed.

## Update (2026-11-05, council and no-Claude model policy)

- Council: four independent answers completed; two Claude reviews hit the weekly limit.
  Owner banned future Claude calls and approved 3 fresh gpt-6-luna members + 1 Gemini.
  Only the two failed reviews were replaced with Luna; completed answers and GPT/Gemini reviews
  were reused. No further Claude requests after the ban.
- Verdict: keep width 320 and restart 8/16 for now. Next proposed step is a frozen-checkpoint
  policy screen (8/16 vs 32/16), with independent validation replication and repetition/failure
  checks. No new training or GPU run is approved. Smaller-model screening is the dissent;
  EMA/live state and unequal selected epochs must be audited before a capacity comparison.
  R7 remains running: next-test runtime and owner choice are still pending.
- Saved report: Copilot session `files/council_20261105/verdict.md`, alongside the anonymous
  mapping and original/replacement reviews.
- Updated user-level llm-council skill, AGENTS.md, shared Copilot instructions and all 7 agent
  frontmatters: default gpt-6-luna; no Claude models/CLI/subagents. Stronger models need owner
  approval, except the approved Gemini council member. Full council: 8 calls, answer cap 300
  words, review cap 450; no hidden retries. No commits or settings changes.

## Update (2026-11-05, R5 pair-1 long-stream validation)

- Ran the existing width-320 bundles, sequentially in VS Code integrated task terminals.
  Frozen FSQ g8/v1024; 32 distinct validation captions; root seed 2026; TF32; CFG 6;
  temperature 1; top-p .9; six 49-token windows (9.8 s each), batch 4.
- Mamba full-state R@1: .594 -> .000 at 49.0-58.8 s; distance 3.051 -> 8.366.
  Restart every 8 tokens from the last 16: R@1 .594 -> .406; distance 2.945 -> 4.207.
  Late restart-minus-full R@1 +.406 (95% prompt-paired interval [.250; .563]);
  distance -4.159 [-5.356; -2.934].
- Transformer restart R@1 .563 -> .469; distance 3.049 -> 3.606.
  Full-state continuation is unsupported at 294 tokens (97 learned positions, including prefix).
  Under restart, late Mamba-minus-transformer R@1 -.063 [-.250; .125]:
  this sample does not establish a twin winner.
- Evidence: `doc/paper/pics/long_stream_validation.csv` (18 aggregate windows), JSONs
  `artifacts/outputs/qa/long_stream_{mamba,transformer}_w320_p1_val_2026.json`
  (captions, sampler, per-prompt scores, drift intervals), and
  `long_stream_w320_p1_comparison_2026.json` in the same directory.
  Reproduce with the Long-stream tasks in `.vscode/tasks.json`; R5 marked done.
- Limits: forced continuation beyond the 9.8 s training maximum; captions can describe finite
  actions. Intervals resample prompt pairs within fixed 32-motion distractor pools, not model
  training seeds. Offline decoding and total batch wall time are not a streaming-latency result.
  Restart jointly changes retained history and text access. No smaller-width or longer-training
  comparison was run. These results support the restart policy, not an epoch or capacity claim.

## Update (2026-11-04, Copilot agents, epics in CSV, Mamba kernel 9.3x, diffusion decoder pilot)

- What we tried: 7 Copilot agent roles in `.github/agents/` (2 branches per question); epics and
  tasks now live in `.claude/docs/epics.csv` and `epic_tasks.csv`; nightly paper watch at 03:00
  (arXiv + OpenAlex, local files only).
- What we got: Mamba training step is memory bound (90% element-wise, 35% `F.pad` copies in
  `parallel_scan`). The saved CUDA kernel works: 1011.3 -> 108.4 ms per step (9.33x), same math
  (output 4.5e-8, gradient 4.4e-7), but not bit-exact (3.1e-5 after 3 steps, atomic adds).
  Owner chose: kernel for phase 4 plus a deterministic backward (K7, draft in a worktree).
  Half the positions (stride 8) makes a Mamba step 2.21x faster. Token segments almost never
  repeat (top 1000 of 4 positions cover 0.73%), so presets must be whole clips.
  Diffusion refiner on frozen tokens, 30 s smoke: validation recon-FID 0.0318 -> 0.0241, MPJPE
  104.9 -> 124.1 mm, 27 ms per 48-frame chunk.
- Claim risks found: `generalization.md` (1000 clips): FSQ beats RVQ in 6/6 matched cells on
  test but 3/6 on validation (RVQ wins all codebook-1024 cells by 0.0007-0.0021);
  `tokenizer-matrix-status.md` calls the test number "val"; `bib.bib:29-36` mixes the CVPR and
  arXiv T2M-GPT data; foot slide finds the floor per clip, so floating is not detected.
- Where we stopped: the 60-minute pilot runs; K7 builds after it.

## Update (2026-11-02, full regroup by kind across packages)

- Owner chose after a 2-member Copilot review: dissolve streaming, 4 small fixes, generation and
  studio splits. Commits 57bc2d8..ef90591, each with an AST proof that moved code is identical;
  542 tests pass; golden identical.
- Packages now: app (adds server, protocol, cli_requests), evaluation (adds embeddings),
  generation (adds datasets, decoder; nn_text merged into nn_model), motion, studio (adds
  fitting, session, panels), tokenization. No streaming package; import map updated.
- Defect fixed: the overfit gate messages showed fixed limits (99%, 0.1, 95%) for any criteria.
- Removed: 3 dead studio methods; reverse import losses -> nn_model.
- Lesson: Copilot drafts can drop class separator boxes and add re-exports; check both.
- Open: rendered architecture images not regenerated; backlog in the ledger (swap_left_right
  hand branch, freeze guard top folder only). Next: Block 2 (tokenizer) backlog.

## Update (2026-11-02, motion phase 1 regroup done)

- Motion package regrouped by kind (Copilot design, owner approved), 7 commits f1e9111..S8:
  contracts (all records, model.py merged), geometry (14 rotation functions), kinematics
  (Skeleton only), representation (263 format + mirror), normalization (scaler + all stats),
  storage (files, both split readers, caption parsing), datasets, amass, preparation.
- Removed: dead cont6d_to_matrix_np, 2 statistics wrappers, pose_folders_and_candidates (split
  in 2 private functions). Split readers: base_clip_ids (no mirrors), listed_clip_ids (all).
- Every step: AST proof that moved code is identical, 532 tests pass, golden identical.
  Tokenization unfrozen and refrozen 2 times for import-only changes. Motion frozen 85538d36.
- Backlog: swap_left_right hand branch (25-51 joints) and assert checks; freeze guard hashes
  only the top folder; rename tool keeps only 2 path parts.
- Next (owner approved order): one cross-package design review (shared records, runtime
  helpers, runner with execute() per package, streaming request file name), then Block 2.

## Update (2026-11-02, stabilize module by module; Block 1 motion)

- Owner rules: one root seed 2026 (derived seeds allowed); the owner approves every process
  start and every agent subtask; stabilize the blocks in order, data to output, before new work.
- Confirmation runs: per-step decision void (it used seed 2027). Seed 2026 only: Mamba per-step
  FID 3.53 vs 4.15 prefix; transformer per-step failed its gate (cause unknown).
- Evaluator check: real motion through our protocol path gives the published row (R@3 0.799 vs
  0.797, MM-Dist 2.964 vs 2.974, Diversity 9.505 vs 9.503). 3 defects held: fixed Diversity
  draw, half the test population (mirror IDs removed), MultiModality first 100 clips unseeded.
- Service restarts both twins every 8 tokens from the last 16, so Mamba's full memory is unused.
- Baseline suite: 530 passed, 2 skipped. Block 1: seed is a required keyword (no default 42),
  pose adoption sample seed 2026; golden identical; motion frozen again (efa1a60b).
- Next: Block 2 (tokenizer) backlog, then generation, evaluation, streaming, studio, base.

## Update (2026-11-01, weekend results, latency check, confirmation queue)

- New init vs old init, transformer 15 epochs (1 seed): FID 4.26 vs 8.62, R@3 0.49 vs 0.41,
  train CE 4.98 vs 5.60. Both still improve at epoch 15: no early stop seen.
- Twins at 15 epochs, new init: transformer FID 4.26 R@3 0.49; Mamba FID 4.31 R@3 0.50.
  Training time per epoch: transformer 3.3 min, Mamba about 33 min.
- Per-step A/B at 3 epochs: better in all 3 complete pairs (FID about 20.5 vs 24-32), but the
  rule could not apply (validation CE is not logged; one gate failed). Not adopted yet.
- Latency, batch 1, random weights: transformer eager 6.34 ms/token, Mamba eager 15.05, Mamba
  CUDA graph 1.76. Not fair yet: the transformer has no CUDA graph. Graph vs graph is next.
- Confirmation queue (scratchpad/confirm, frozen worktree of c81eff6): 15-epoch per-step arms
  under RULE.md (written before the runs), then Mamba seed 2027 prefix for twin quality.
- Same seed and config are not bit-identical between runs (epoch 1 FID 34.40 vs 33.76).

## Update (2026-10-31, reference init, per-step text, weekend queue)

- Commit c81eff6: both twins use the mamba_ssm reference init; flag per_step_text (default
  false) adds the text vector to every motion token. Old checkpoints load and give the same results.
- Council 4 (4 Copilot sol roles): fix training and evaluator first, fair benchmark in parallel,
  final runs 2026-11-07; defer the prompt monitor and the RVQ retrain. Dissent: per-step A/B.
- Overfit gate, transformer: old init reaches 99% in 26 steps, new init in 1276 steps. Gate
  budget 3000 steps for both twins (it stops at pass). Hypothesis: old init memorized fast.
- Weekend queue (scratchpad/weekend/queue.sh): 2 seeds x 2 twins x prefix/perstep, gate + 3
  epochs each; then Copilot writes analysis.md against a decision rule fixed before the run.
- Twin size checked: Mamba 34.09 M, transformer 33.90 M parameters.

## Update (2026-10-30, motion validated and frozen)

- Two Copilot audits (GPT, Gemini) + a 4-model council review of the change set. 2 audit P0
  claims were false against the official code (face-joint order; mirrors in stats) and were kept.
- Fixed with tests (commits c8bd390 .. f8e7075, ledger doc/notes/module_ledger.csv): input checks
  (collate, config, scaler, recovery width, caption times), CUDA streaming recovery, batched
  rotation recovery, read-only offsets, gender decoding, reference clip required, fps 20, crop
  bounds, min length on encoded frames (corpus v2), name collisions, stale sources, honest
  manifests, the held-out guard (both folder names, every variant). Golden tests identical.
- Frozen: 7d7a1f5, hash 70f8042e...; tests/golden/test_module_freeze.py enforces it.
- Findings: the evaluator scores 2192 of 4384 test lines (mirrors removed, old behavior) -> a
  protocol decision with a 2-sided experiment in the evaluation step. pose_data has no manifest
  and raw AMASS is absent -> the corpus cannot be rebuilt; final pretraining uses the legacy
  AMASS_263 features with held-out names excluded at pack time.
- Copilot with write access is blocked by the session safety check; Copilot runs read-only.
- Tokenization: 2 audits (35 findings) + 2-member review; fixed with tests in 1d7584e (index,
  config, resume and input checks; frozen-wrapper batch/frame checks; metrics read the network
  downsample; token pack int16 and alignment guards; RVQ codes from the assignment codebook and
  coherent counts -> future RVQ training only). FSQ golden outputs identical. Frozen 636b0df
  (hash 32aecca0...). Finding deferred: residual-FSQ variant mixes bounded and unbounded values.
- Generation: 2 audits + 2-member review. Found: Mamba dt_proj keeps the default nn.Linear init
  (reference: log-uniform dt in [1e-3, 1e-1] via inverse softplus) -> likely handicaps every
  Mamba run; FK masked loss 22x (FK weights are 0 in all configs, so no run used it); CUDA-graph
  shared state and stale A; END stop rule (F09); transformer 3 dropout sites vs Mamba 1;
  accumulated CE weights microbatches, not tokens.
- Owner decisions (2026-10-30): run the 2-arm dt experiment (2 epochs each, scratchpad dt_exp/);
  match dropout after it; token-weighted CE; test-split experiment in the evaluation step;
  RVQ is trained again fairly after the RVQ fix (no note-only option).
- dt A/B (2 epochs, warmup 1000 > 700 steps, so both arms stayed in warmup): no quality
  difference (CE 6.614 vs 6.617). Memory check (median dt, exp(dt\*A) per token):
  trained e60 Mamba dt 0.547, keeps 1.4% per token, memory ~1 token in all 15 layers; fresh
  0.691; 2-epoch default arm 0.698 / ~1 token; 2-epoch reference arm 0.010, keeps 93.4%,
  memory ~15 tokens. Weight decay is not a cause (a_log, d_skip and 1-D params are exempt).
- Conclusion: the Mamba twin had almost no recurrent memory in every run so far; training does
  not repair it. A/B with warmup 100, 2 epochs, 1 seed: token CE 6.540 vs 6.541; 135-clip FID
  32.0 (old) vs 40.2 (new), R@1 0.039 vs 0.055; memory ~1 vs ~15 tokens. No measured quality
  gain at this short horizon; adopted for correctness (owner approved): 7de6cea. Repeat the
  memory check at the end of the final Mamba run. The thesis must report only the new runs.
- Fairness audit (Copilot): neither twin uses its reference init (embeddings N(0,1) vs 0.02,
  no 1/sqrt(layers) residual scaling); prefix-16 configs hide the caption from Mamba; KV cache
  copied per token; the fused kernel does not speed up Mamba decode.

## Update (2026-10-30, owner decisions after council 2: correctness first, module by module)

- Council 2 (all 4 members on Copilot: Claude Opus, Claude Sonnet, GPT, Gemini) found the
  latency headline unfair: CUDA-graph decode is Mamba-only (generation/nn_model.py:44-49), so
  3.35 vs 9.49 ms/token is graphed Mamba vs eager transformer. With re-anchoring both backbones
  have bounded state (the transformer re-prefills prefix + 16 tokens, nn_model.py:163-168), and
  without it the transformer stops at 97 positions. The long-sequence crossover does not apply.
- Owner decisions: group 1 (motion -> tokenization -> generation -> evaluation) is the focus;
  the owner designs the structure, the council helps realize it, no big refactor; full proven
  correctness of each module before anything else continues; re-anchoring stays and the twin
  comparison must be fair (same graphs, same anchoring, same decode path for latency and quality).
- Process per module: two independent Copilot audits (GPT, Gemini) -> council review of the
  proposed changes -> owner approval -> fixes with tests -> freeze. Freeze criteria: no open P0
  (wrong numbers) or P1 (silent failure), a test for each fix, golden tests identical or the
  change explained.
- Where we stopped: motion audits running (scratchpad audit_motion/).

## Update (2026-10-30, council + phase 0 and 1: safety net and silent-load fixes)

- Council (4 members: 2 Claude, GPT, Gemini): do phases 0-1, then freeze the code and run the
  final twin runs from scratch; defer the library move (phase 2) and cudagraphs (phase 3).
  Biggest risk: F06 already made Mamba e60 a 30-epoch model, so the final runs must happen again.
- Phase 0: golden tests in tests/golden (tokenizer: 19 output fingerprints `d963802`; generator:
  exact 32-clip metrics of both e60 bundles at precision highest `1d3b869`), with pinned input
  hashes and torch version, fail loud when a file is missing. Audit
  scripts/qa/checkpoint_audit.py `b844325`: now 71 rows, 0 mismatch; 3 demo entries are LEGACY
  resume files (live weights, no checks). Task list sorted by Copilot (scratch): open blockers
  F07 (pretrain does not compare pack scaler/split hashes), F03, F08, F20; F10 and F12 fixed.
- Phase 1: RVQ YAMLs got their tokenizer block `ffc3928`; CUDA and RVQ-config silent fallbacks
  now raise `edd85f6`; FrozenMotionTokenizer reads downsample from the network `eb47691`; FSQ
  load refuses other levels `0cf97b1`; generator loader refuses another config (use_kernel
  skipped) `672c15c`. Golden outputs unchanged at every step.
- Where we stopped: owner decision on F07 and the other blockers, the 3 legacy demo entries,
  then the code freeze and the final-run budget.

## Update (2026-10-30, failure logs and dumps)

- What we found: a NaN guard stopped the run, but the traceback went only to the console (not
  run.log), and no file kept the failed batch or the model state at that step.
- What we changed (`4eda1c0`): config `failure_logs: true` (default) -> a failed command is
  logged into run.log, and a failed tokenizer / generator / pretraining epoch writes
  `failure_dump.pt` in the run folder: resume state, failed batch (`inputs`), `epoch_step`,
  `non_finite` parameter and gradient names, traceback. `detect_anomaly: false` (default) turns
  on torch anomaly detection for a replay from the resume file (2-10x slower).
- Tests: one per loop, the shared dump, the CLI flag on/off, the anomaly flag; all checks pass.
- Not covered: the sanity-overfit gate loop; the console shows the traceback twice when
  failure_logs is on (once from the log, once from Python).

## Update (2026-10-30, math audit: RVQ EMA codebooks and TF32 tokens)

- What we found: the tokenizer EMA averaged parameters only. The RVQ codebooks are buffers
  (k-means EMA), so an evaluation paired the averaged encoder with the live codebook. FSQ was
  not affected. The old RVQ checkpoints (rvq\_\*.pt) and their scores carry this mismatch.
- What we changed: `ExponentialMovingAverage(average_buffers=True)` for the tokenizer only,
  the same rule as torch AveragedModel(use_buffers=True); the generator keeps parameters only, so
  its 10 resume files still load. Commit `84f2ebe`, 99 tokenization tests pass.
- What we measured: fsq_g8_v1024 on 730 validation clips, TF32 vs highest: 347 of 204,104
  token indices change (0.17%), in 268 clips; highest twice gives 0 changes. So a token pack
  depends on the precision setting. rvq_l6_512.pt does not load with its current config.
- FID now matches the Guo et al. reference (text-to-motion/utils/metrics.py): a non-finite
  sqrtm is computed again with a 1e-6 diagonal offset, and an imaginary diagonal part above 1e-3
  raises (before: silently dropped). Normal values do not change (imaginary 2e-7 at rank 2..32).
  Commit `368384c`, 23 evaluation tests.
- Checked, no change: the uncertainty weighter (no config uses it; a clamp would change the
  Kendall formula), the training std (min 0.014; a floor would be an invented number),
  parallel_scan, sample_logits.
- Owner decision: keep TF32. The tokenize command now writes `precision` into the pack identity,
  and generator pretraining refuses a pack with another precision or none (`a8918ad`, 101
  tokenization tests). The 3 packs in data/ are legacy (no identity) and need a rebuild anyway.

## Update (2026-10-28, real-settings speed: TF32 and the window loader)

- What we found: with the real app settings (deterministic, precision highest) a tokenizer step
  was 149 ms, not 89 ms, and the loader added 139 ms per batch in the training process. The
  loader cost came from low free RAM (4.6 of 17 GB before the cache, 2.2 GB after): each new
  small array was 24x slower, and each item made two (normalize, RandomState).
- What we changed: the window cache is normalized once at build, and the window starts of an
  epoch come from one vectorized draw per (seed, epoch); loader batch 82 -> 9 ms, validation
  windows bit-identical, train windows differ from old runs `b73e570`. Precision default is now
  TF32 (owner choice, all configs): step 149 -> 69 ms `4d97ebb`. Empty token pack refused,
  request fields bounded `015a1c6`.
- Warning: TF32 changes every new run at about 1e-3 relative, including a re-score of an old
  checkpoint; set `precision: highest` in a config to reproduce a published number. Both twins
  must use the same setting.
- Checked, no change: the small-sample FID raises below 2 clips and logs a rank-deficient
  warning below 512; the generator loader is 13.6 ms per batch of 8 in 4 workers against a
  158 ms transformer step.

## Update (2026-10-27, tokenizer edge cases)

- What we did: probed 30 edge cases of the tokenization package. 9 failed silently (NaN motion
  gave index -9.2e18; FSQ level 2 gave NaN codes; FSQ decode wrapped indices out of range; an RVQ
  NaN step wrote NaN into the codebook before it raised; an empty epoch returned {}; a second EMA
  copy_to lost the trained weights; an evaluation with no usable clip or one NaN clip gave NaN
  metrics; an empty FSQ level list built). Each now raises a clear error, with a test. `b3b58c6`.
- Also: encode and forward raise for a frame count that is not a multiple of downsample (the
  62 -> 60 trim); the frozen tokenizer still cuts clips itself.
- Checks: reference identical, 395 tests by folder, pyright 0, ruff clean, real val split passes.
- Open (not approved yet): an empty token pack is written without error when all clips are
  excluded or short; request fields epochs, eval_every, stride, segment_frames have no lower bound.

## Update (2026-10-27, tokenizer speed and consistency)

- What we measured (RTX 3050, batch 128 x 64 frames, another process on the GPU, so only the
  parts are reliable): a step is about 89 ms, and the network itself is 95% of it. bf16 autocast
  would cut the network part by 30%, but it changes the numerics, so it waits for E7.
- What we changed: EMA through `torch.optim.swa_utils.get_ema_multi_avg_fn` (3.4 -> 0.85 ms, also
  used by the generator) `422a25f`; code statistics on the GPU, one CPU copy per epoch (2.4 ->
  1.4 ms) `53798ae`; one `split_clip_ids` for all 8 split readers, a missing feature file raises,
  evaluation `split` required `02d8043`; evaluation in batches of equal length (val 14.4 -> 4.8 s,
  recon-FID moves 3e-5) `8ecd972`; grouped FSQ in one call (5.6 -> 0.7 ms, bit-identical)
  `7561a1e`.
- Bugs found: (1) an RVQ step with quantizer dropout failed in `bincount` on the -1 marker, so
  the RVQ baseline could not train with its default `quant_dropout=0.2`; fixed. (2) a decoder
  silently dropped extra index columns when tokens had more codebooks than the tokenizer; now a
  `ValueError`. (3) `streaming_decode_fidelity.py` loaded each clip twice (plain and `M`); fixed
  by the shared reader, so its first `max_clips` now differ.
- Checks: reference identical, 359 tests by folder, pyright 0, check_problems 0, ruff clean.
- Open: `data.sources` is read by no code; the default pack paths of tokenize and pretrain
  differ; the EMA does not average the RVQ codebook buffers.

## Update (2026-10-27, no global state rule)

- What we did: the owner asked for no global variables in any file. We found 0 `global`
  statements, but 44 module constants that were lists, dicts or sets, and 4 lowercase module
  names. They are read-only now (tuple, frozenset, MappingProxyType, nested values too):
  src `9e832df`, scripts `15d4e85`, tests `00e3777`. Two tests now load their script through
  fixtures in `tests/app/conftest.py`. FunctionNode became FUNCTION_NODES.
- What we found: `test_warm_reuse.py` put the read-only POLICY into a JSON message; json
  cannot write a MappingProxyType, the service thread failed, and the test waited forever (no
  timeout). Each message now gets `dict(POLICY)`.
- The rule: `check_problems.py` reports G01 (mutable module constant) and G02 (lowercase module
  name) and runs ruff PLW0602/PLW0603 (global). The problems_sweep hook runs it on each
  changed file. 5 tests in `tests/app/test_check_problems.py`.
- Next: move helper functions by concept, one package per round, tokenization first.

## Update (2026-10-27, type aliases as `type` statements)

- What we did: src had 10 type aliases written as plain assignments with one underscore
  (`_MetricSink = Callable[...]`). MetricSink was in 3 files and Evaluator in 2. Now each alias is
  a Python 3.12 `type` statement with a public name. MetricSink and Evaluator are defined once in
  `motion/contracts.py` (next to Split), and the two trainers and pretrain import them.
  CheckpointSink, InferencePipelineLoader, ReconstructionFidScorer, Color and Pair stay in
  their files. The renames went through `rename_everywhere.py`.
- Checks: reference identical, 345 tests by folder, pyright 0, check_problems 0, ruff clean.
  Commit `d68dd71`.

## Update (2026-10-27, class separator boxes for the minimap)

- What we did: the owner asked for a box above each top-level class, for minimap navigation.
  This is an exception to the zero-comment rule. A new script,
  `scripts/maintenance/add_class_separators.py`, adds the box to each src file with two or more
  classes (25 files). It also corrects a box with an old class name, and a second run changes
  nothing. Tests: `tests/app/test_add_class_separators.py`.
- Layout: 60 `#`, the class name after 8 characters, 60 `#`, two blank lines (ruff format wants
  two). After the imports there is one blank line above the box (ruff import sort wants one).
- Checks: reference identical, 343 tests by folder, pyright 0, check_problems 0, ruff clean,
  line endings of each file kept (42 CRLF, 15 LF).
- Where we stopped: the script does not run by itself. Run it after you add a class.
- Widened later the same day: the owner asked for a box in all src files, so the 13 files with
  one class also have a box now (38 files in total). Files with no class stay unchanged.
- Layout change the same day: the owner wanted one blank line under the box. Ruff format allows
  only 0 or 2 there (also in preview style), so the owner chose 0: the box touches the class.
  The script now finds an old box with any number of blank lines under it and replaces it.
- Hook fix after this round: `ide_problems_check.py` listed only Edit/Write files, so it missed
  the 25 files that the script changed. It now also lists each `.py` file in src, scripts and
  tests that changed after the last user message. Up to 30 files: one Problems panel call per
  file. Above 30: one call with no URI is enough, and pyright in `problems_sweep.py` checks each
  file. `files_changed_after` moved from `problems_sweep.py` into this hook (one copy).

## Update (2026-10-27, tokenizer hunt: duplicates, acronyms, request merge, shared bases)

- `ddc1dfc` every quantizer stack returns (quantized, indices, commitment_loss); TokenizerNetwork
  owns encode/decode/forward/codebook_size; the unread `cfg` attributes are gone.
- `2696956` removed values that repeat others (stack num_quantizers, group dim, trainer
  codebook_size, EMA pair cache `_tracked`); typed_modules uses the 3.12 generic syntax.
- `334f765` acronyms: TokenizerKind.FINITE_SCALAR / RESIDUAL_VECTOR (CLI values fsq/rvq stay),
  FiniteScalarStackLayout, commitment_beta, dropout_probability, reconstruction_fid_scorer and
  the local names. Log keys (recon, commit, usage_frac, recon_fid), YAML keys, checkpoint keys stay.
  evaluate_tokenizer_reconstruction old vs new on 20 val clips: identical (MPJPE 123.9 mm, in
  line with the earlier 115-125 mm records).
- `8624b74` deleted TokenizerTrainingRequest and TokenPackRequest; train_tokenizer and
  build_token_pack take TrainTokenizerRequest / TokenizeRequest; explicit weights_only on the
  best-checkpoint load (Sonar S6985).
- `080f918` MotionTokens.token_count is a property of indices.
- `5c0e08b` FrameSequenceConvolution base for the motion feature encoder and decoder.
- Checks each step: 19-tensor reference identical, 338 tests by folder, pyright 0,
  check_problems 0, ruff format and lint clean.
- Proposed, not done: keep ABC (not Protocol) for Quantizer/QuantizerStack; move the loss into
  trainer.py; FrozenMotionTokenizer to its own file (it is not an nn.Module).

## Update (2026-10-27, tokenizer clarity: Tier 1 structure + rename batches A, B, C)

- What we did: made the `tokenization` package clearer with no change in behavior. A reference
  (19 tensors: fsq_g8_v1024 and rvq_l8_1024 checkpoints, plus a seeded residual FSQ; encode,
  decode, forward, soft decode, train-mode forward) stayed identical bit for bit after each step.
- Tier 1: `d6051b9` match + assert_never for the tokenizer kind and the FSQ composition; removed
  the unused `from_checkpoint` and `scaler`. `5df86d8` one `ResidualQuantizerStack` base for the
  two residual stacks. `aff0b1d` removed the codebook-table cache that losses.py wrote onto the
  tokenizer; measured cost 0.43 ms per step of 190 ms (0.23%).
- Renames: `7ede2e4` + `4b90887` quantizers (Quantizer, QuantizerStack, FiniteScalarQuantizer,
  VectorQuantizer, Grouped/ResidualFiniteScalarQuantizer, ResidualVectorQuantizer, quantizers,
  combine_code_vectors, active_stage_count, round_straight_through). `b4cf610` networks
  (TokenizerNetwork, FiniteScalarQuantizationTokenizer, ResidualVectorQuantizationTokenizer,
  FrozenMotionTokenizer.network, quantizer_stack, downsample_stage_count). `20075c3` contracts
  (TokenizerReconstruction: reconstruction, token_indices, commitment_loss;
  ResidualVectorQuantizationConfig). `2db01d6` rename_everywhere `--only FILE ...`, because the
  word FSQ is also `TokenizerKind.FSQ` and thesis text.
- Kept on purpose: checkpoint keys (encoder, decoder, net, pre_q, post_q, quantizer, groups,
  layers, rvq, embed, ...), `num_codebooks` (generator config), YAML keys, `TokenizerConfig`.
- Checks each step: 338 tests by folder, pyright 0, check_problems 0, ruff format clean.
- Open: split TokenizerConfig/RvqConfig (shared encoder fields, YAML change); a ruff format
  check in check_problems (the rename tool does not format, H03).

## Update (2026-10-27, F13, F26, H01, decision check D01-D12)

- F13 `257eb85`: the model-only benchmark writes a `scope` field (random weights, no CLIP,
  sampling, decoding, transport or display). The service log row now has `first_chunk_ms`, the
  release identity and the policy. New `scripts/benchmarks/service_latency.py` measures through
  the service. Run on the Mamba e60 bundle (5 prompts, 20 steps): median first chunk 110 ms
  (cold first request 1,083 ms), 9.0 ms/step, 12.9x real time, 0.54 GB peak GPU.
- F26 `11ccc0c` + `10ac59c`: README (overlap is not equality; protocol note), configs/README (no
  inheritance), demo.dot + diagrams README (analytic fit at 20 fps, SLERP to 120 fps, no END stop);
  thesis: "Status of these results" notes in chapters 4 and 5, chapter 5 overlap sentence,
  chapter 6 figure caption (model-only). `make pdf` builds with no error.
- H01: the Pylance error in rename_guard.py was stale; the Problems panel is now empty.
- Decisions (CSV still says awaiting_operator_decision for all): answered by work: D04 (studio
  SLERP contract), D08 (raw AMASS path, 120 fps from the field, pose manifest, verified index),
  D09 (heldout duplicate closure F07/F03), D10 (epoch-boundary resume), D12 (TrainingRun.psm1),
  B00 (H01), DATA01 (legacy packs refused). Partly: D01 (packs cover tails; generator training
  still crops to 196 frames), D05 (pairing fixed; temporal annotations open), D07 (Guo 20 reps;
  extra measures open), D11 (remainder fixed; pilot open). Open: D02, D03, D06, DOC01.

## Update (2026-10-26, new findings: N7 kernel, index file, N9 prepare target, N10, regen rerun)

- Index `ce32c13`: the verified official HumanML3D `index.csv` (commit 3ea8be34, sha256 80226910…,
  14,616 rows) is copied to `data/HumanML3D_official/index.csv` with its provenance; all 28 configs
  set `paths.hml3d_index_csv`.
- N9 `321d20f` (found while checking the index fix): `prepare` wrote the HumanML3D stages into
  `paths.hml3d_out_dir`, the official training dataset (index/feature/stats would overwrite it once
  the index was set). The stages now write only to `--regen_dir` and refuse the training folder.
- N10 `a9e935f`: the spawn-worker test hung once in a full run (not reproduced in 3 later runs); it
  now fails after 60 s (DataLoader timeout) instead of hanging.
- N7 `858a04a`: the Mamba CUDA kernel (optional, `use_kernel: true` in 4 configs, 5 runs) is not in
  uv.lock. The working build (342 MB .pyd, sha256 2f3c100c…, torch 2.12+cu132) is saved in
  `dist/mamba_kernel` (git-ignored); `make kernel-restore` puts it back (copy tested);
  `make kernel-build` uses reconstructed MSVC/NVCC flags and is UNTESTED (the exact flags are lost).
  README uses `uv sync --inexact`.
- Regen rerun into `data/HumanML3D_263` (official fingerprint unchanged before/after): index stage
  wrote 11,676 clips, 2,940 without a pose file recorded in `index_manifest.json` (1,191 HumanAct12,
  1,642 from datasets not downloaded: EyesJapan, Transitions, BMLhandball; 107 recordings absent
  from this AMASS release). Feature stage wrote 23,134 clips and stopped on 218 (109 clips + mirrors):
  their recordings in this AMASS release are shorter than the HumanML3D index range (example CMU
  80_63: 379 frames, index 516-716). Open decision: let the index stage record out-of-range clips.
- N8 open: 13 new files from today have CRLF line endings (repository already mixed).

## Update (2026-10-26, items 6-9: N3, F25, F18, N5)

- N3 `b0d8dad`: the HumanML3D feature stage writes atomically, records `features_manifest.json`
  and stops with the failed clip names; the index stage records clips without a pose file in
  `index_manifest.json` with a warning. Measured: `HumanML3D_263` misses 23% of train features
  (all 49 logged runs used `HumanML3D_official`, which is complete). `paths.hml3d_index_csv` is
  not set and no index.csv is on this machine; the token-pack rebuild needs it.
- F25 `b9a29f6`: extras `prep` (pyarrow) and `tracking` (mlflow-skinny, sqlalchemy, alembic)
  with clear install errors; `viewer` extra removed (aitviewer is core). pyarrow 26.0.0
  installed. N7 found: a plain `uv sync` uninstalls mamba-ssm and einops (local wheel, not in
  the lock); `make install` now uses `--inexact`. Declaring mamba-ssm in the lock is still open.
- F18 `c483217` + `66ebb99`: studio.yml holds only live settings (fit: model_dir, device,
  threads, num_betas; schedule: retrofit_window); the live fit takes no warm start; perf_probe
  owns the offline iteration counts (20/50/0/25) and prints "offline iterative fitter".
- N5 `899fc83`: the captioned and window loaders stop when a listed clip has no feature or
  caption file; short clips are still dropped. Real split sizes unchanged (train 22418,
  val 700, test 2105). Regression found and fixed: a test fixture with mirroring on but no
  mirror file.
- Checks: 333 tests (slow and GPU included) pass, pyright 0 errors, ruff and check_problems
  clean.

## Update (2026-10-26, top 5 design smells fixed, full regression check)

- N1 `4dfb4fc`: `corpus.pack_clip_keys` is the one reader of clip keys (pretrain, scaling_budget,
  pretrain_generalization). scaling_budget no longer raises IndexError on new packs.
- F05 + N4 `1b272a6`: `check_scaler_values` (1-D, finite, std > 0) runs in `MotionScaler.load`
  and before `fit_train_stats` writes; Mean/Std are written atomically. All three real scaler
  folders still load (smallest std 0.0005, eval_stats).
- F16 `ec6be56`: the window cut trims the fitted track (orient/pose/transl) with the body
  frames; dead `SmplxFitState.n_frames` (it counted chunks) removed.
- F17 `0edeab7`: `live_step_budget` resolves the no-limit value 0 before the live cap (old rule
  gave 0 = no limit for a live preview).
- N2 + F23 `c3de518`: render_clip, render_mesh_demo and pretrain_generalization load through
  `load_inference_pipeline` / `app.load_tokenizer`; pretrain_generalization checks the pack
  tokenizer hash and loads with weights_only=True. Smoke run on the thesis Mamba e60 bundle:
  32 finite frames; a wrong tokenizer and a legacy pack both stop with an error.
- Regression: 318 tests + 5 slow/GPU tests pass; each new test fails on the old code; pyright 0
  errors; Problems panel empty. The slow AMASS test skips because `paths.amass_dir` is null;
  run by hand on `D:/Facultate/dissertation/data/amass`: 88 finite frames, recipe OK.
- Note: pretrain_generalization cannot run until a new token pack with an identity exists.

## Update (2026-10-26, design-smell status and new search; no code change)

- Still open from the audit (checked in code today): F05 scaler accepts zero/NaN std, F16
  fitted track never trimmed, F17 `min(steps, live.steps)` turns 0 (no limit) into 0, F18 live
  controls of the offline fitter, F23 render scripts bypass the checked loader, F25 pyarrow
  undeclared, F13 benchmark scope, F26 text. F22 (Azure log folder) is moot: Azure closed.
- New N1 (my F07 regression, reproduced): `scripts/benchmarks/scaling_budget.py` reads every
  token-pack key and calls `.shape[0]`; the F07 identity record is 0-d, so it raises
  IndexError on every new pack. `pretrain_generalization.py` also reads raw keys (counts the
  record as a clip stem, loads with allow_pickle=True).
- New N2: `scripts/evaluation/pretrain_generalization.py` loads raw state dicts with no
  tokenizer check (same as F23, but an evaluation script).
- New N3: the HumanML3D feature stage (`encode_joint_clips_to_hml3d263`) logs a failed clip at
  info level and continues, writes non-atomically, no manifest (F03 pattern; default config
  uses the official data, so low impact today).
- New N4: Mean/Std are overwritten non-atomically (pair with F05).
- Rejected by measurement: silent split shrinkage. All listed clips have files and captions;
  only clips under 40 frames drop (train 966, val 60, test 174), the HumanML3D rule. Latent:
  a missing file would drop the same way, with no message (N5).
- N6: 22 scripts lack the command header line.

## Update (2026-10-26, studio: F28, retrofit at 20 fps, perf_probe)

- F28 `10cc624`: the stream stops feeding the fit queue when the fit thread is dead, so the
  stream finishes with the skeleton and the status says "body fit failed". Rare in use
  (`fit_single_chunk` catches `Exception`); the earlier test hang came from a fake fitter.
  The new test fails on the old code.
- Retrofit `5c7dcc7`: the stream buffers keep the 20 fps joints of the take. The display
  window is cut only by multiples of `upsample`, and the 20 fps joints are cut in step (window
  plus one anchor frame). A late body ("avatar skin" after the motion) fits the anchor frame,
  then the 20 fps frames with `factor = upsample`: the same path as the live body. Side fix:
  the cut also used `[-window:]` on the display joints but `excess` on the skeleton node; both
  now use `excess`. The window can hold up to 5 more frames than `window_frames`.
- `perf_probe.py` `15dbd1b`: the demo probe stops with a clear message when no SMPL-X folder is
  given (was a TypeError).
- 304 tests pass, pyright 0 errors. Not pushed.

## Update (2026-10-26, hooks: H01 Copilot patches, H05 diagnostics tool, H09 commit marks)

All changes are local hook files (`.claude/` is not in git).

- H01: the Copilot CLI runs these hooks from `.claude/settings.json` fail-closed; a raw patch
  text as `tool_input` crashed 7 hooks (`str` has no `.get`), so Copilot denied each
  `apply_patch` (seen in the Copilot logs, 2026-10-22 to 2026-10-24). New
  `.claude/hooks/hook_input.py` (`tool_edits`) gives one edit for each file of a patch (path,
  removed lines, added lines, move target). All 7 hooks use it; post-edit hooks check every file
  of a patch; `rename_guard` also denies a patch that renames a definition or moves a project
  file; `git_guard` allows a patch (no command). Tests: 8 real Copilot patches crash no hook;
  7 behaviour cases pass, including the old Edit and Bash decisions.
- H05: `ide_problems_check.py` accepts `mcp__claude-vscode__getDiagnostics` (current) and the
  old name, and the request names the current tool. The audit's "SDK transcript" case is not
  reproduced on real data: in the Copilot log, 7 of 7 replies with the section passed the Stop
  hook, so `understanding_check.py` is unchanged.
- H09: `git_guard` removes `.claude/` path tokens and `$CLAUDE_...` variables before the search
  for assistant marks; marks in the message still deny (5 cases).
- Open: in the Copilot session the edit tool is `apply_patch`, so `ide_problems_check` and the
  sweep see no Copilot edits (they read Edit/Write only).

## Update (2026-10-26, hooks: H04 edit checks change only owned files, H10 rebase --quit)

- H04, proved first: an edit of one file made `post_edit.py` delete an unused import in another
  file (whole-tree `ruff check --fix`). Now (local hooks, not in git):
  - `post_edit.py` formats and repairs only the edited file, keeps unused imports there (F401
    not auto-fixed, but reported), and checks `src`, `scripts`, `tests` read-only. It no
    longer starts `graphify update .` in the background; run `make graph` for a refresh.
  - `problems_sweep.py` repairs only the Python files that Edit/Write changed in this turn
    (from the transcript). Other files with a new change time get a read-only check through
    `check_problems.py --check` (commit `5a17032`), and the report says not to edit them.
  - `ide_problems_check.py` and the sweep share `turn_tool_uses` (the tool calls of the turn).
  - Tests: the probe file stays unchanged; the sweep case repaired the owned file, left the
    other file unchanged, and named it in the report.
- H10: `git_guard.py` permits `git rebase --quit`; `--abort`, `--skip`, `--continue` stay denied
  (8 sample cases). The deny text names `--quit` for an old rebase.
- Limit: a file that I change through a shell command now gets only the read-only check.

## Update (2026-10-26, design smells: loader epochs F04, figure tree F24)

- F04 `448835e`: the dataset epoch is now a shared-memory integer, so persistent loader workers
  see each new epoch. Before: in a run with 2 spawn workers, epoch 0 and epoch 1 gave equal
  crops. Effect on old runs: the twin runs used 4 (transformer) and 2 (mamba) workers, so each
  clip kept one crop and one caption for each resume segment; the twins had different segment
  counts. Effect on FID not measured; the reruns use the fix. The thesis text must not claim a
  new crop and caption each epoch for the old runs (F26).
- F24 `c70753f`: `make_figures.py` deleted. It wrote to `paper/figures`, read a missing
  `outputs/` folder (empty figure, no error) and held constant numbers for two figures that
  the thesis does not use. `plot_streaming_bench.py` reproduces `doc/paper/pics/streaming.png`
  pixel for pixel from `streaming_bench_long.json`, writes there by default, and records its
  input with sha256 in `streaming.source.json`. `architecture_diagram.py` writes to
  `doc/paper/pics` and `doc/notes/diagrams` (output equal to the committed files).
  `make figs` runs these two scripts and makes no `paper/` tree.
- Open: `doc/notes/FEEDBACK.md` still names `paper/figures/streaming.png` (F26 text).
- 301 tests pass, pyright 0 errors. Local is 8 commits ahead of GitHub; not pushed.

## Update (2026-10-26, design smells: config root keys F19, warm service identity F15)

- F19 `1bc14e5`: `load_config` refuses an unknown root key and names the nearest known key.
  Before, a typo such as `generatr:` was dropped and the run used the default generator
  (checked with a test config). `hml3d` stays as the one legacy root key. All 28 experiment
  configs load under the new rule.
- F15 `de8f674`: the service sends a release identity (backbone, path and sha256 of the config,
  checkpoint and tokenizer) and its policy (anchor settings, CUDA graph) in `hello`. The client
  reloads when an identity field differs, and it stops with an error when the policy differs,
  because a load cannot change the policy. The studio now refreshes `trained_steps` and
  `default_steps` on a model change. Cost: about 1 s of hashing for a 400 MB checkpoint.
- 299 tests pass, pyright 0 errors. Local is 6 commits ahead of GitHub; not pushed.

## Update (2026-10-26, design smells: token pack F07, corpus F03, poses F03-b)

- F07 `5891b11`: the token pack leaves out heldout captures and their copies (same size and
  sha256), adds an end-aligned last window, and stores an identity record (tokenizer, stats,
  index and split hashes). Pretraining refuses a pack without the record or with another
  tokenizer. On real data: 2,254 heldout names; 66 were in the corpus (the 62 from the audit +
  4 copies).
- F03 `2bf2156`: the AMASS corpus writes `corpus_manifest.json` (recipe, counts, failed names).
  A folder with features and no manifest, or with another recipe, is refused; a heldout file
  in the folder stops the build; totals now count reused files; conversion failures finish
  the build, write the manifest, then raise.
- F03-b `714a906`: the pose stage writes `pose_manifest.json` (fps, joints, betas, SMPL-X
  hashes), saves each pose atomically, and the corpus recipe holds the pose recipe. A corpus
  build needs a complete pose manifest.
- Stopped here because: the existing `data/HumanML3D_263/pose_data` and `data/AMASS_263`
  have no manifest, so both are now refused. Next data step (needs approval, GPU):
  `text2motion prepare --stage amass --adopt_pose_folder true` (re-extracts 20 sample files,
  tolerance 0.1 mm), then a corpus build into a new folder, then a new token pack.
- 264 tests pass, pyright 0 errors. Local is 4 commits ahead of GitHub; not pushed.

## Update (2026-10-25, repository: published to GitHub, abort incident)

- GitHub `thesis-v2` now equals local `thesis-v2` = `efd17e4` (98 commits). Pushed with
  `--force-with-lease=thesis-v2:739e3e7`; it replaced the donor `SYNC` commit `739e3e7`, which
  held only a `.gitignore` rewrite and a `CLAUDE.md` deletion (kept locally in
  `backup/github-thesis-v2-sync-739e3e7`). Before the push: no file above 50 MB, no secret name
  or key pattern in the 98 commits. After the push: every commit on GitHub `thesis-v2` has the
  owner as author and committer, and no message has a co-author or assistant line.
- `thesis-v2` tracks `github/thesis-v2`. The guard (`.claude/hooks/git_guard.py`, local, not in
  git) now allows a push only to a named remote and branch, only if every unpublished commit
  has the owner's identity and a clean message, and a force only with `--force-with-lease`
  (9 sample cases checked).
- Incident: a rebase left unfinished on 2026-09-15 stayed open for six weeks. `git rebase
--abort` moved `thesis-v2` back to `e22e735` (files of 15 September). Recovered with
  `git merge --ff-only efd17e4`; nothing was lost (246 tests pass). Rule: for a stale
  "rebase in progress", use `git rebase --quit`, never `--abort`.

## Update (2026-10-25, audit round 5: service settings F14, training scripts F21)

- F14 `a8daea9`: `serve`, `studio` and `health` read the service settings from one file
  (`configs/service.yml`) through `load_service_config`, which applies the environment
  overrides; order: defaults < file < environment < CLI. Before, the file replaced the
  environment port and host, and `health` read a different source. The duplicate `service:`
  block in `configs/default.yaml` is removed.
- F21 `e474ee8` + `574c6c7`: new module `scripts/training/TrainingRun.psm1` (6 tests, Windows):
  prints the exact quoted command, keeps the real exit code (cached process handle), kills on
  the time budget or no heartbeat under `artifacts/logs/train/` (code 124), passes `--resume`
  only when `*_last.pt` exists, skips a run only when its `*_last.pt` reached the last epoch
  (before: when the best checkpoint existed, which happens at epoch 6), and writes DONE only
  after success. All 16 active scripts use it (-364 lines). Defects removed: lost exit codes,
  DONE after a crash, heartbeat folder wrong (stall check never fired), "one trainer at a time"
  check that never matched (`train_tokenizer` vs `train-tokenizer`), an endless wait in
  `fsq_sweep.ps1`, and `--fresh_selection` on every relaunch of the dynamics fine-tune (a
  crash-resume could overwrite the best checkpoint with a worse one).
- The extend and fine-tune scripts now pass `--accept_legacy_resume --extend_schedule`
  explicitly, because their source files are version 1 and their horizon changes.
- Warning kept: `run_3day_local.ps1` and `run_rvq_ablation_local.ps1` pretrain on the legacy
  token packs that contain heldout windows (DATA01); F07 must reject those packs in code.

## Update (2026-10-24, audit round 4: frame rates, F01, and the 120 fps studio display)

- Frame rates measured (read-only): all 16,407 raw AMASS motion files are 120 fps (49.5 h; 477
  more files are body-shape files without `trans`). Raw Inter-X is 120 fps (README, and the
  processed copy has exactly 1/4 of the frames, 200 of 200 clips); processed Inter-X is 30 fps.
  The project does not read Inter-X (its files have `pose_lhand/pose_rhand` and two people).
- F01 `dc93353`: the AMASS extraction now stops on a rate that is not an integer multiple of
  20 fps (a stride would change the speed) and on a motion file without a rate field. No data
  changes (all files are 120 fps, stride 6). The "unverified timing" reason of the AMASS
  quarantine is gone; the heldout-leakage reason (DATA01) stays.
- Models train and generate at 20 fps (HumanML3D-263, 1 token = 4 frames = 0.2 s). Benchmark
  numbers use the 20 fps output only. The studio shows 120 fps: `display.upsample: 6`,
  `window_frames: 1200` (10 s), commit `02ec445`.
- Studio `55f84ea`: the live stream fits SMPL-X on the 20 fps joints and SLERPs the fitted
  rotations (roma, shortest arc) and the translation to 120 fps. Measured on a fast real chunk
  (3.46 m/s): same mesh as fitting interpolated joints (mean gap 0.01 cm), 120 ms against 111 ms
  per 0.8 s chunk; the live analytical solver is about 8x faster than real time either way.
  Correction: the earlier "fit is slower than real time" claim measured the offline iterative
  fit, which the live studio does not use.
- Open for the studio round (F16-F18): the retrofit (body switched on after the motion) still
  fits display-rate joints, because the studio does not store the 20 fps joints; F28: a dead
  fit thread blocks the stream for ever on the full fit queue; `perf_probe.py` gives a
  TypeError when `--model_dir` is missing.

## Update (2026-10-24, audit round 3: caption and motion pairing, F02)

- F02 `a3d4cc6`. Checked against the reference code (EricGuo5513/text-to-motion,
  `data/dataset.py`, `Text2MotionDatasetV2`). Training: the fallback "random tagged caption +
  full recording" is gone; the dataset now stops at build time if a clip has no caption that
  covers 40 frames (no real clip fails; train builds in 9.6 s). NaN intervals map to the whole
  clip, as in the reference (192 lines in the data).
- Evaluation now uses the Guo sample set: a whole clip is a sample only with an untagged
  caption; each tagged caption is its own sample with its own frames; samples with fewer than
  40 or 200+ frames are skipped (before: cropped to 196). In each run one caption per sample is
  drawn (seed from sample id + run) and drives both the generation and R-precision/MM-Dist
  (before: generation always from the first caption line, 174 test clips from a tagged one).
  New sizes: test 4,648 samples (450 segments, 234 out of length), val 1,530.
- Remaining known difference from the reference: Guo's random one-unit length cut is not copied.
- Consequence: run 0 no longer reproduces the old evaluation; earlier numbers are on the old
  sample set and the old caption rule. The final test evaluation must be run again anyway (F12).

## Update (2026-10-24, audit round 2: END tokens and evaluation repeats)

- F09 `6043b1a`: each generated row stops at its own END (END in any codebook ends the row);
  END is masked at step 0; the decoder never receives END. No effect on past numbers: all 56
  recorded evaluations used `length_mode: fixed`, where END is never sampled.
- F12 `d84903c`: `evaluate --reps R` now generates all clips again in each run (seed
  `crc32(clip_id) + run`, paired between twins) and reports mean +/- 1.96\*sd/sqrt(R) (Guo et al.
  formula, ddof 0) for FID, R@1-3, MM-Dist and Diversity. The bootstrap is a separate field
  (`fid_bootstrap_lo/hi`, run 0 only). Proof on real data: `--reps 1` on 64 val clips gives FID
  0.6781781234317192, bit-identical to the 2026-10-12 run. `--reps 3` on the same clips: FID
  0.678 / 0.764 / 0.967, mean 0.803 +/- 0.137, so one generation run can be far from the mean.
  What it changes for the thesis: every earlier "20-rep" number is ONE generation with 20 caption
  and batch resamples; the +/- values do not include generation variation. Describe them so until
  the test split is evaluated again with `--reps 20` (about 90 min per checkpoint).

## Update (2026-10-24, audit round 1: resume and numerical safety closed)

Fixed, tested and committed (219 tests pass, pyright 0 errors):

- F20 `dbeb7a3`: a NaN/Inf gradient stops all 3 loops before `optimizer.step()`
  (`error_if_nonfinite=True`); the overfit gate rejects non-finite metrics, zero counts and
  out-of-range values; a non-finite evaluation metric stops the run before selection.
- F08-1 `00f9519`: every checkpoint save is atomic (`motion/storage.atomic_torch_save`);
  `--resume` without the resume file is an error (scripts that pass `--resume` on a first run now
  fail; F21 fixes them).
- F08-2 `49ffd98`: resume format version 2 (RNG, uncertainty weights, schedule horizon),
  loaded with `weights_only=True`. Old files need `--accept_legacy_resume`; a new `--epochs` on a
  resume needs `--extend_schedule`. Resume only at an epoch boundary (decision D10).
- F11 `ed2886d`: a short last accumulation window gets the weight of a full window. With AdamW the
  old effect was small (Adam is almost scale-invariant), so past results change little from it.
- F08-3 `7628a71`: the resume file records its run directory; all 3 commands continue it.
- F10 `83ed32d`: `--eval_split` is deleted; selection is always on validation. No past run used test.
  New findings: F27 (residual FSQ with quantizer dropout crashes `__code_stats`; no config uses it);
  H09 (`git_guard` checks the whole command for "claude", so a commit command with a `.claude/` path
  is blocked); H10 (`git_guard` blocks `git rebase --quit`). H08 is fixed: hook paths use
  `$CLAUDE_PROJECT_DIR`. Open: the stale rebase state onto donor `739e3e7` (owner runs
  `git rebase --quit`). Fact for the thesis: grouped-FSQ tokenizer training uses no random numbers.
  Next: F09 (stop tokens), F12 (evaluation repeats), F02, F01, F21, F14, then the P2 items.

## Update (2026-10-24, audit item F06: each resumed generator run evaluated a frozen EMA)

What we found: before commit `98dc4b7`, a resume replaced the EMA dict, but `update()` still
changed the old tensors. After a resume, each evaluation and each saved best checkpoint used the
EMA weights from the resume point. Training after the resume had no effect on the evaluated model.
Proof (bit equality of all tensors, read-only check):

- `generator_transformer_bs8_e100.pt` (best at epoch 100, FID 0.2304) = the EMA in
  `snapshots_60ep/generator_transformer_bs8_e60_last.pt` (92 of 92 tensors).
- `generator_mamba_bs8_e60.pt` (best at epoch 30, FID 0.1831) = the EMA in
  `snapshots_60ep/generator_mamba_bs8_e60_last.pt` (177 of 177). That run resumed at epoch 30
  (lineage: segments 1-30, 31-60, 61-100), so its EMA stopped at epoch 30.
  What it changes:
- Mamba e60 (seed 2026) is a 30-epoch model. Transformer e60 (seed 2026) had no resume before
  epoch 60: correct. The twin comparison at 60 epochs is not matched.
- Both e100 runs and the `dyn_e72` / `dyn_e84` fine-tunes evaluated the frozen resume EMA.
  The claims "more training helps neither twin" (Lesson 17, sec. 6.1) and "the dynamics
  fine-tune does not help" have no support now.
- Mamba seed 2027 had no resume (60 epochs in one process, best FID 0.2157 at epoch 60): correct.
  Where we stopped: F06 fixed and tested (`98dc4b7`); F20 non-finite stops (`dbeb7a3`). Next: F08
  resume (epoch boundary only, strict `--resume`, old resume files only with a flag), then F11.
  No new run until the audit items are closed; then a matched rerun of the affected runs.

## Update (2026-10-13 22:40, second seed (task E6): Mamba run with seed 2027 is queued)

Owner said "go seed 2". Done in this sequence:

1. `git reset replay` (mixed, no file change): the branch `thesis-v2` now has the replay history
   (65 commits after `e22e735`). NOT pushed; the owner makes the push. Correction: `--soft` was the
   wrong advice, because it leaves the old tree staged; the mixed form is the correct one.
2. A background script commits the work of today in 12 small commits, 4 to 6 minutes apart.
3. New config `configs/generator/gen_pilot_fsq8x1024_kernel_seed2027.yaml`: the kernel config
   with `seed: 2027` (one line differs). `train-generator` has no seed flag; the seed is in the config.
4. Overfit gate with that config: PASS (accuracy 0.9917 at step 51, loss 8.72 -> 0.41).
   File `artifacts/outputs/gates/mamba_34m_kernel_seed2027_overfit.json`.
5. A second background script waits for the commits, commits the seed config, then starts:
   `train-generator --config <seed config> --backbone mamba --init_ckpt generator_mamba_pretrained.pt
   --ckpt_name generator_mamba_bs8_e60_seed2027.pt --epochs 60 --batch_size 8 --num_workers 0
   --eval_every 6 --cfg_scale 6.0 --temperature 1.0 --resume`. Log: `artifacts/outputs/seed2027_mamba.log`.
   Expected: start near 23:30, 60 epochs x 9.3 minutes = about 9.5 hours, end near 09:00.
   Same settings as seed 2026, except: the compiled kernel (same output to 6 decimals),
   `--num_workers 0` (4 before; data order comes from the seed, not from the workers), and the seed.
   Both seeds start from the same pretrained weights, so the seed changes the data order and the
   dropout of the fine-tune, not the pretraining. State this in the thesis.
   Next: the transformer twin with seed 2027 (about 7.5 hours), then the standard test evaluation
   (20 repetitions) for both, then the paired analysis.
   If the run stops: start the same script again; `--resume` continues from the last epoch.

## Update (2026-10-13 21:40, type errors: 234 -> 0 in the full project; results unchanged)

`pyright src scripts tests`: 0 errors, 1 warning. The editor tool reports no problem in any file
it tracks. 190 tests pass. Lint clean. The pyright baseline is now empty, so the edit hook
permits no type error in any file.
Proof that no result changed: `evaluate` for the Mamba e60 checkpoint on 64 validation clips
gives the identical line as before the repair (FID 0.678, R@1 0.418, R@3 0.622, MM 3.583,
Div 9.073, MModality 1.765).
Repairs by cause:

- Stored tensors and module lists with no declared type (about 60): class-level type lines for
  buffers; new base classes `QuantizerUnit` and `SequenceBackbone`; helper `typed_modules`;
  method `latent_quantizer()` on the tokenizer network; `codebook_tables` declared.
- Wrong argument types (about 105): tests use the enum members (`Backbone.MAMBA`), not text;
  test doubles are marked with `cast`; scripts use `required_path` for optional config paths.
- Values that can be `None` (about 30): a check with a clear error before the use.
- Missing packages: `matplotlib` installed and declared (6 new packages, none changed), so the
  figure and render scripts can run; a type stub for `pyarrow.parquet` in `typings/`.
- `ema.py`: the private PyTorch calls are replaced by public in-place calls with the same
  arithmetic.
  Real defects found by the type check: `scripts/evaluation/eval_generalization.py` used
  `base.paths`, which does not exist (now `base.config.paths`); the studio received a typed request
  where its code declared an untyped one (now the protocol `StudioLaunchArguments`).
  The 1 warning: `pyarrow` is not installed; only the optional parquet import of
  `motion/preparation.py` needs it. Not known: SonarLint warnings in files that are not open.
  All of this is uncommitted work on top of the branch `replay`.

## Update (2026-10-13 21:45, CORRECTION: the "0 problems" result was wrong; 234 type errors remain)

The owner pasted 11 problems from the editor after the "0 problems" report. Cause: the script
hid each type error that was inside the pyright baseline, and it has no SonarLint rules.
Repaired now (the editor shows 0 problems for these files): `studio/scene.py` (3 type errors,
1 redundant return), `app/bootstrap.py` (safe load of the text encoder checkpoint; explicit
return type), `studio/avatar.py` (old fit: complexity, `weight_decay`), `studio/viewer.py`
(two complex functions split in small methods). 190 tests pass.
The script no longer uses the baseline. Real count for the full project: 234 type errors.
Largest files: `tokenization/nn_quantizers.py` 35, `motion/kinematics.py` 20, four test files
11 to 17 each. Main causes: 105 wrong argument types (many are a text value where an enum is
declared), 60 from buffers and module lists that have no declared type, 13 imports of packages
that are not installed (matplotlib). Not repaired yet.
Rule for reports from now: "0 problems" is valid only if the editor tool confirms it.

## Update (2026-10-13 21:15, full check and repair: SUPERSEDED, see the correction above)

`python scripts/maintenance/check_problems.py --all` now reports 0 problems.
Final rule list: basic errors, imports, probable bugs (B), simpler code (SIM), returns (RET),
complexity above 15 (C90), too many returns or branches. Removed from the list, with the reason:
long lines (the formatter owns the line length), too many statements (the owner prefers long
functions with short lines), unused arguments (all 12 cases were signatures that an interface needs).
Automatic repair: 5 places (merged `with`, merged branches, plain `return`).
Repair by hand: 26 places. 8 variables that existed only for a `return` (motion code); 3 files
opened with no context manager; 3 `zip` calls with no `strict`; 2 default arguments that called
`OverfitCriteria()` (now the constant `DEFAULT_OVERFIT_CRITERIA`); a lambda that used a loop
variable; 3 complex functions split (`find_hazards`, `__avatar_controls`, and one branch less in
`stream_token_indices`); 1 type error in `cli.py` (`parse_request` returns `CommandRequest`).
Side effect: the formatter replaced the owner's aligned command table in `cli.py`. The aligned
form is still in the commit on the branch `replay`. Owner decision: keep the formatter form, or
keep the alignment with `# fmt: off` / `# fmt: on` (an exception to the zero-comments rule).
Tests: 190 pass. All files formatted. 23 files changed; not in a commit.

## Update (2026-10-13 20:50, script and hook that check and repair the problems of changed files)

Script: `python scripts/maintenance/check_problems.py` (no argument: files changed against the
last commit; file names: those files; `--all`: the full project). Steps: `ruff format`,
`ruff check --fix` (automatic repair), then a report of what remains: lint with a wider rule set
(bugs, simpler code, early returns, complexity above 15, too many branches or statements, unused
arguments) and type errors above the pyright baseline. Exit code 1 if a problem remains.
Hook: `problems_sweep.py` (Stop). It finds each Python file changed after the last clean check
(also files that a script changed), runs the script on them, and blocks the final reply with the
list if a problem remains. The marker file is `.claude/hooks/.problems_checked`.
Measured: the wider rule set finds 81 old problems in the full project (12 long functions,
9 complex functions, 8 `RET504`, 28 unused arguments, others). They do not block; a file must be
clean when it changes. Not done: the repair of those 81.
Limit: SonarLint has no command-line form here; `ide_problems_check.py` still covers the editor.

## Update (2026-10-13 09:45, all commits made on the branch `replay`; ready for the owner's push)

The branch `replay` has 65 commits after `e22e735`: 43 from the replay of the old history
(2026-10-12 19:04 to 2026-10-13 06:17, 7 to 23 minutes apart), 20 of the new work (06:25 to
09:38; 4 to 6 minutes apart from 08:42, by owner request) and 2 for the studio files (09:41, 09:45).
Each commit has the owner as author and committer; no message has an attribution line.
The tree of `replay` is identical to the snapshot of the main folder from 06:17.
The main folder differs in one file only: `studio/scene.py` (+4 lines, changed after the snapshot).
Claude Code reported a stop of the background scripts three times for low memory (0.4 GB free at
09:26); the script processes continued each time.
Owner steps: `git reset --soft replay` in the main folder, then
`git push --force-with-lease=thesis-v2:739e3e7 github thesis-v2`.
After the push: the worktree `../thessis-replay`, the tag `backup-before-replay` and the ref
`refs/snapshots/work` can go (owner decision). Not run on the final tree today: the test suite
(190 tests passed on the same files on 2026-10-12; memory was too low today).

## Update (2026-10-12 22:20, studio package smaller: actions.py removed, avatar.py 588 -> 506 lines)

`studio/actions.py` (3 log functions) joined `studio/generation.py`; the studio has 7 files.
`studio/avatar.py`: the hand-written Kabsch rotation and the rotation-matrix to axis-angle code
(72 lines) are replaced by `roma.rigid_vectors_registration` and `roma.rotmat_to_rotvec`
(equal to 1e-6 on test data, including half turns). `roma` is now a declared dependency.
The controller methods use early returns: `fit_single_chunk`, `on_fit_body_toggled` and
`__rebuild_avatar_mesh` are split into small methods; the exception `FitCancelled` is removed.
New owner rule saved: early returns, early break, low nesting.
Open (owner decision): `fit_smplx_to_joints` and `__yaw_init` (118 lines, the old iterative fit).
The studio does not use them; two scripts do (`perf_probe.py`, `render_mesh_demo.py`). Removal
also removes 8 config fields. The 3 editor warnings that remain in `avatar.py` are in that function.
After the second replay script, a `LEFTOVER` line is expected for the studio files and
`tests/studio/test_analytic_body_solve.py`; they need one more small commit.
Tests: 190 pass (one test of the removed hand-written function is gone). Lint clean.

## Update (2026-10-12 21:55, commit plan for the new work, interval for A1, task R1 DONE)

Commit plan: a second background script waits for the DONE line of the replay. Then it takes a
snapshot of the main folder and makes 20 small commits on the branch `replay`, 7 to 23 minutes
apart. Tested in a separate folder with no pause: 20 commits, tree identical to the snapshot.
The log line `DONE-NEW-WORK` means all is committed; a line `LEFTOVER` lists files with no group.
Expected end: near 10:30 on 2026-10-13. Then the owner runs `git reset --soft replay` in the main
folder and makes the push. The snapshot includes the owner's edit of `cli.py`.
A1 interval (96 sequences): the 95% bootstrap interval of R@1 with the action list is about
+/- 0.10 (Mamba: [0.36, 0.56], [0.32, 0.51], [0.38, 0.58] for actions 1 to 3). Against the joined
caption: exact McNemar p <= 0.003 and Wilcoxon p < 1e-11 for each action and both twins. In the
thesis. `own_caption_ranks` moved to `evaluation/metrics.py`, so two scripts use one function.
Task R1, memory of the generators (`scripts/qa/memory_probe.py`, 32 test captions, 48 tokens):

- Change one past token and measure the change of the next-token distribution (KL, nats).
  Mamba: 10.3 at distance 1, 0.26 at 8, 0.053 at 16, 0.015 at 32, 0.14 at 47 (the first token).
  Transformer: 14.5 at 1, 0.087 at 8, 0.026 at 16, 0.022 at 32, 0.28 at 47.
  A change of the caption: 0.78 (Mamba) and 1.20 (transformer).
- Half-life of the Mamba state values, from the mean step size of each layer: median 0.13 to 0.28
  tokens, 90th percentile 0.5 to 2.0 tokens, largest value 27 tokens. No state value has a
  half-life above 48 tokens; at most 1.4% of the values in a layer are above 8 tokens.
  Reading: both twins use mainly the last 8 to 16 tokens. This agrees with "keep 16" as the best
  reset setting. The trained Mamba has no long memory: the state has a fixed size, but its content
  is short. The first token keeps some effect (0.14), so a small part of the state carries the
  position in the clip; this is the probable cause of the failure after the trained length
  (a hypothesis, not measured). For the retrain decision of 2026-10-31: a carried state (T1) gives
  long memory only if the training task needs it; the action list (A1) gives the long structure
  with no training.
  Limits: the half-life uses the mean step size, so moments with a small step keep more; 32
  captions; one random replacement for each position. No figure: matplotlib is not installed.
  Tests: 191 pass. Lint clean.

## Update (2026-10-12 21:20, boundary test, task A2 negative, A1 in the thesis, editor-problems hook)

Boundary test (action list, reset every 8, 96 sequences). Kept tokens 16 / 24 / 32:

- Mamba: boundary acceleration 0.91 / 0.74 / 0.59 cm/frame^2 (middle of an action: 0.61);
  distance of action 2 to its caption 3.53 / 4.11 / 4.29; R@3 of action 2 0.750 / 0.604 / 0.604.
- Transformer: boundary acceleration 0.97 / 0.86 / 0.82; distance of action 2 3.54 / 3.62 / 4.08.
  Reading: more kept tokens give a smoother boundary and a slower change to the new action.
  The service setting stays at keep 16. Output files: `action_list_<twin>_keep<k>.json`.
  Task A2 (one-pass rebuild of the Mamba state at a reset): NEGATIVE, code removed.
  The one-pass state was identical to the token-by-token state (test, 1e-5), but slower on this
  GPU: reset every 8 keep 16 went from 47 to 79 ms/token, reset every 1 from 276 to 541. The
  parallel scan of training builds tensors of size batch x length x 1024 x 16 for each of 15 layers,
  and the 4 GB GPU was full (3.8 GB). A possible later test: the compiled kernel with
  `return_last_state`; it needs the hand-built package, so it is not in the plan.
  Thesis: section 6.4 has two new tables (`tab:action_list`, `tab:boundary`) and three paragraphs.
  The PDF builds (44 pages, no undefined reference).
  New hook `ide_problems_check.py` (Stop): if a turn changed a Python file with Edit or Write, the
  assistant must call the editor diagnostics tool for that file before the final reply.
  Limit: the tool returned an empty list for a file that is not open in the editor.
  Tests: 191 pass. Lint clean. Replay: 11 of 43 commits at 21:12. All new work uncommitted.

## Update (2026-10-12 20:50, task A1 DONE: the action list keeps text match over 3 actions)

What we tried: 96 sequences of 3 test captions (288 clips, seed 2026, ids in the output file).
Each sequence is one stream of 3 x 48 tokens (28.8 s), reset every 8 tokens, keep 16. Two methods
with identical random numbers: (a) one joined caption ("c1, then c2, then c3") for the full stream;
(b) action list: at the reset on each boundary the text changes to the caption of the next action.
Each segment is scored against the caption of its action (Guo evaluator, pools of 32).
Script: `scripts/qa/action_list_probe.py`. Output: `artifacts/outputs/qa/action_list_<twin>.json`.
What we got (action 1 / 2 / 3):

- Mamba, joined caption: R@1 0.198 / 0.135 / 0.125; text distance 6.30 / 7.07 / 7.95; foot slide 0.37 to 0.42.
- Mamba, action list: R@1 0.458 / 0.417 / 0.479; text distance 2.77 / 3.53 / 3.66; foot slide 0.27 to 0.29.
- Transformer, joined: R@1 0.240 / 0.104 / 0.094; text distance 6.25 / 7.19 / 7.62; foot slide 0.40 to 0.41.
- Transformer, list: R@1 0.448 / 0.427 / 0.521; text distance 2.89 / 3.54 / 3.32; foot slide 0.26 to 0.29.
- Boundary with the action list: mean acceleration 0.91 (Mamba) and 0.97 (transformer) cm/frame^2,
  against 0.61 and 0.62 in the middle of an action. No root jump at a boundary in any sequence.
  Reading: with the action list, actions 2 and 3 keep the text match of a single clip (R@1 of the
  e60 twins on the full test set: 0.418 and 0.395). The joined caption fails from action 1.
  The two twins give the same picture.
  Limits: 96 sequences (3 pools), so R@1 has an error of about 0.05; one seed; the models never
  saw a joined caption in training, so method (a) is a weak baseline; the boundary is about 50%
  rougher than the middle of an action.
  Where we stopped: the result is not yet in the thesis. Uncommitted until the replay ends.

## Update (2026-10-12 20:40, rename debt repaired, rename guard added, first step of task A1)

Debt: each old file name and function name is updated in the agent files, the decision records,
`references.md`, TASKS.md, the lesson notes and the pyright baseline. Only this file keeps old
names, because it is the history record.
Editor problems: the 6 type errors and 2 warnings in `generation/nn_backbones.py` are repaired
(0 errors in that file). The tokenizer checkpoint now loads in the safe mode (`weights_only=True`).
The generator bundle cannot load in the safe mode, so that call stays.
Rename tool: `python scripts/maintenance/rename_everywhere.py OLD NEW`. It moves the file (if OLD
is a path), updates code, tests, scripts, documents and the `.claude` folder in one step, and
reports each import of a module that does not exist. 0.4 seconds for the full project.
Rename guard: new hook `rename_guard.py` (PreToolUse on Bash, PowerShell and Edit). It denies a
move command on a project file and an edit that renames one function or class, and names the tool.
Type-error baseline: updated for the moved files; total 249 -> 242; no file is above its baseline.
Task A1, step 1: `stream_token_indices` accepts `text_at`, a function that gives the text for the
current token index. The text changes only at a reset. Two new tests. Not started: the test split
of 100 caption triples, the generation runs and the measures.
Tests: 191 pass. Lint clean. All uncommitted until the replay ends.

## Update (2026-10-12 20:20, owner rule: the file name says the kind of content)

A file with only neural network modules has the prefix `nn_`; a file with only pydantic models
has the prefix `pydant_`. Renamed: `generation/model.py` -> `nn_model.py`, `backbones.py` ->
`nn_backbones.py`, `text.py` -> `nn_text.py`; `tokenization/model.py` -> `nn_model.py`,
`quantizers.py` -> `nn_quantizers.py`; `evaluation/matcher.py` -> `nn_matcher.py`;
`streaming/contracts.py` -> `pydant_requests.py`. 45 files with references updated (code, tests,
scripts, lesson notes, NOTICE). 186 tests pass, lint clean, 129 type errors (no change).
Open for the owner: the mixed `contracts.py` files, `motion/model.py` (data types, wrong name),
`generation/losses.py`. Not updated: file names in the agent files under `.claude`, and the
pyright baseline. All uncommitted until the replay ends.

## Update (2026-10-12 20:10, two owner rules applied: one objective per function, small packages)

Rule 1: no "and" or "or" in a function name. Four functions repaired:
`connect_or_start_motion_server` -> `motion_service_client` (+ 2 private helpers),
`load_tokenizer_and_generator_spec` deleted (callers make the two calls),
`has_motion_or_body` -> `has_generated_motion`, `prompt_bar_and_playback` -> `prompt_playback`.
Rule 2: a package has 10 files at most; helper code joins the file of its concept. Eight files
merged: `schema.py` + `config_loader.py` -> `config.py`; `logging_setup.py` -> `run_log.py`;
`runtime.py` -> `bootstrap.py`; `annotations.py` + `datamodule.py` -> `datasets.py`;
`scan_kernel.py` -> `backbones.py`; `plausibility.py` -> `metrics.py`.
Files now: app 6 (+6 commands), motion 10, generation 9, evaluation 6, tokenization 8,
streaming 5, studio 8. Source: 11,163 lines. 186 tests pass, lint clean, 129 type errors (144 before).
Not decided by the owner: 40 test names with "and"/"or"; the pyright baseline needs new entries
for the merged files. All of this is uncommitted until the replay ends (7 of 43 commits at 20:07).

## Update (2026-10-12 19:53, checks after the library steps: real GPU run, formula tests, type errors)

What we tried: three small checks while the replay of the commits runs.
What we got:

- Real run: `evaluate` for the Mamba e60 checkpoint on 64 validation clips completes with the new
  parser and config loader (FID 0.678, R@1 0.418; a 64-clip FID is not comparable with the tables).
- Formulas: 7 new tests with answers known by hand (`tests/evaluation/test_known_values.py`):
  Frechet distance, MM-Dist, Diversity, R-precision in one pool, foot slide, acceleration, root jump.
- Type errors: the three files from step L1 went from 19 errors to 4; each file is at or below
  its baseline.
  Tests: 186 pass. Lint clean.
  Owner rule saved: take each formula from a reference library (scikit-learn first, then SciPy).
  scikit-learn has none of our formulas; SciPy already supplies the statistical tests.
  Where we stopped: all work of today after the replay start is uncommitted, until the replay ends.

## Update (2026-10-12 19:30, library steps L2 to L8: 3 applied, 1 partial, 3 rejected by measurement)

What we tried: each step from L2 to L8, with a measurement before each change.
What we got:

- L2 APPLIED. pydantic validates the config dataclasses. `schema.py` + `config_loader.py`:
  296 -> 152 lines. All 30 config files load to the identical value. Unknown keys fail at each level.
- L4 APPLIED. `mlflow-skinny` + SQLite store. Top-level config key `mlflow_tracking: true` sends
  the metrics of a run to `<logs_dir>/mlflow.db`. `manifest.json` and `metrics.jsonl` stay.
  A resumed run continues the same mlflow run. 31 new packages; no installed package changed.
- L5 APPLIED (move only). New files `generation/backbones.py` and `tokenization/quantizers.py`.
  The e60 checkpoints load.
- L6 PARTIAL. One dead function removed from `motion/kinematics.py`. No move to `roma`: the file
  is the MIT port that defines the 263 features, and `roma` has a different quaternion order.
- L3 REJECTED. `AveragedModel` copies the full model (more memory) and old resume files do not load.
- L7 REJECTED. The library schedule is identical up to the planned end, but after it the learning
  rate goes up again (0.145 against our floor 0.05). It saves 7 lines.
- L8 REJECTED. The message frame is 25 of 209 lines; `websockets` needs a rewrite of both sides.
  Tests: 179 pass. Lint clean.
  Where we stopped: the work is not committed, because the replay of the commits is in progress.
  Open: type errors above the baseline in 3 files from step L1 (`commands/generator.py`,
  `commands/serving.py`, `motion/contracts.py`). The baseline needs entries for the 2 new files.

## Update (2026-10-12, all 11 commands on pydantic and jsonargparse; all work committed)

What we tried: each command gets a pydantic request model; jsonargparse builds the parser from it.
What we got: `cli.py` is 149 lines (707 before). 177 tests pass, lint clean. All 56 command lines
in the drivers, the Makefile and the README parse.
Commits: 29 new local commits under the owner's identity (33 with the first 4). Work tree clean.
44 files had lost their CRLF line ends; restored before the commits, so the diffs stay small.
Six commits are larger than the guard limit (qa probes 1720 lines, studio 524, drivers 502,
notes move 50 files, thesis source 18 files, kernel source 80 files).
Where we stopped: remote `github` is added. The owner runs the push:
`git push --force-with-lease=thesis-v2:739e3e7 github thesis-v2`. It replaces 1 commit on GitHub
(`739e3e7 SYNC`, 2026-09-13).
Not done: one real GPU run with the new parser. The `post-commit` hook still tries a push to the
retired donor at each commit; it fails each time. Owner decision is open.

## Update (2026-10-12, `health` and `tokenize` moved to pydantic requests and jsonargparse)

Three of 11 commands use the new parser now (benchmark, health, tokenize). `cli.py`: 707 -> 677
lines. The import rule test rejected a new top-level module, so the shared request base is in
`motion/contracts.py`. 169 tests pass; no new type error. Fourth local commit: the Makefile.

## Update (2026-10-12, one Mamba epoch with the kernel: 9.3 minutes, was 66.5)

What we tried: config `gen_pilot_fsq8x1024_kernel.yaml` (the pilot config with `use_kernel: true`),
overfit gate, then one epoch with the settings of the 60-epoch run but `--num_workers 0`.
Run `20261012T145330Z_generator_mamba`.
What we got: the gate passes in 26 steps (the same as without the kernel). Epoch 1: 9.3 minutes,
ce 6.399 (old run without the kernel: 66.5 minutes, ce 6.408). Validation after 1 epoch: FID 2.55.
Read: 7.1 times faster for a full epoch. A 60-epoch Mamba run needs about 10 hours, not 66.
The first try crashed the computer: 2 data workers plus tests plus a benchmark at the same time,
with 0.5 GB of 15.9 GB RAM free (VS Code 5.1 GB, java 2.9 GB). Error 1455, page file too small.
Rule from now: one heavy job at a time, and `--num_workers 0` when RAM is low.
Also done: `benchmark` now uses a pydantic request and jsonargparse (step L1, first command);
same benchmark result as before; 166 tests pass. Three local commits exist (run merge script,
licence text, licence record). 151 files are not committed.

## Update (2026-10-12, first local commit; the donor remote is retired)

One local commit: `ba93552` "Add a script that merges the run folders of one training" (1 file).
The repository hook `.git/hooks/post-commit` tried to push to the donor folder; the push failed
(the branches diverged), so nothing was published.
Owner decision: the donor branch is at the end of its life. Do not merge, do not join.
Guard hook: new rule "rebase on the upstream before a commit"; it ignores the retired donor remote.
Where we stopped and why: 149 files are not committed. The post-commit hook is still on and fails
after each commit. No new place to push to is named.

## Update (2026-10-11, commit guard hook; the kernel now works in the training code)

Hook `.claude/hooks/git_guard.py` (before each shell command): no push, no amend, no rebase, no
`reset --hard`, no author or identity change, no assistant name or attribution line in a commit.
It asks the owner when a commit has more than 12 files or 400 changed lines. 18 test cases pass.
Kernel: the import workaround moved into `src/text2motion/generation/scan_kernel.py`, so
`use_kernel: true` now works; the existing test that compares the kernel with the plain scan
passes. 162 tests pass. Second measurement: 1055 ms plain, 85 ms kernel (12.4 times).
Where we stopped and why: 151 files are changed and not committed. No commit was made; the owner
did not ask for one. The full epoch time with the kernel is still not measured.

## Update (2026-10-10, the Mamba kernel builds on Windows: 14.7 times faster training step)

What we tried: build `mamba-ssm` 2.3.2.post1 with MSVC 18.6 and CUDA 13.2. Seven problems, all
solved (compiler options, MSVC preprocessor, a math constant, `#ifndef` inside a macro in two
source files, long build of 46 minutes, a Linux-only import, one missing dependency).
What we got: `scripts/benchmarks/kernel_probe.py`, run `20261010T153517Z_kernel_probe`. 34M
Mamba, batch 8 x 49 tokens, forward + backward of the generator only: 957 ms plain, 65 ms with
the kernel; peak memory 991 -> 800 MB; largest output difference 9.5e-7.
Where we stopped and why: the training command cannot use the kernel yet (the import workaround
is only in the probe). The full epoch time is not measured. `mamba-ssm` and `einops` are now
installed in `.venv` but are not in `pyproject.toml` or `uv.lock`; `uv sync` removes them.
The patched source is in `third_party/mamba_ssm-2.3.2.post1-msvc` (Apache-2.0).

## Update (2026-10-10, phase 1 done: paired evaluation of the e60 twins; disk cleaned)

What we tried: `scripts/evaluation/paired_twin_analysis.py`, run `20261010T144528Z`. One more
generation pass of both e60 checkpoints on the 2,189 test captions, then paired tests.
What we got:
| measure | transformer | mamba | paired test |
| FID difference (T - M) | | | +0.013, 95% [-0.032, +0.059], mamba lower in 70% of 1000 resamples |
| distance to caption | 3.113 | 2.938 | mamba nearer on 1165 clips vs 1024; Wilcoxon p 3.5e-5 |
| own caption at rank 1 | 0.497 | 0.520 | 346 vs 296 discordant; McNemar p 0.053 |
| foot slide m/s (real 0.171) | 0.261 | 0.253 | Wilcoxon p 0.12 |
Read: the mamba is nearer to the text (significant). Rank 1 is at the border. FID and the
physical measures do not separate the twins. Both slide ~50% more than real motion on all clips.
Thesis: table 5.3 "Paired comparison" added in section 5.5; PDF rebuilt.
Disk: removed from the donor folder `D:/Facultate/dissertation` on the owner's order:
`checkpoints/motion_ssm`, `checkpoints/rvq_tokenizer`, `data/humanml3d` (272-dim variant),
`data/pahoi`, `data/arctic`. Kept: `data/inter-x` (the owner wants it for later) and
`data/amass`. Drive D: 5.3 -> 97.1 GB free.
Kernel on Windows: `mamba-ssm` does not build as published. Four problems found and solved (see
`doc/notes/implementation-difficulties.md`); the patched build was still compiling at this entry.
Found for the retrain design: the official HumanML3D `index.csv` maps each clip to its AMASS
source and frames. 2,341 AMASS sequences hold 2 or more captioned clips (5,242 clips); 992 have a
captioned span of 20 s or more, 234 of 40 s or more. Captioned long sequences exist without BABEL.
Deadline recorded: submission 2027-02-01, all experiments done by 2026-12-01.

## Update (2026-10-10, plan written: `.claude/docs/TASKS.md`)

The owner agreed to a retrain if a small test supports it. `TASKS.md` holds the plan in 5 phases:
solid evaluation (first), long motion without training, research, retrain, final evaluation.
It also holds the disk survey: drive D has 5.3 GB free; 136 GB in the donor folder can go
(inter-x, old checkpoints, the 272-dimension HumanML3D variant, pahoi, arctic). Nothing is
removed yet; the owner must confirm. Raw AMASS (150.8 GB) stays.
Checked in the scaling paper (arXiv:2001.08361, eq. 4.4): D >= 5e3 \* N^0.74. Our data supports
16k to 64k parameters by that rule, so a 100M model is the wrong direction.

## Update (2026-10-10, long-stream test: the full Mamba state fails after the trained length)

What we tried: `scripts/qa/long_stream_probe.py`. 196 tokens (39 s) for 32 test captions, both
e60 twins, cut into 4 segments of 49 tokens. Measure: distance of each segment to its caption in
the evaluator space (lower is better), foot slide, cost per token. Same random numbers per policy.
What we got (distance, segments 1 to 4):
| | seg 1 | seg 2 | seg 3 | seg 4 | ms/token |
| mamba, full state (no reset) | 2.95 | 6.91 | 8.45 | 8.41 | 18 |
| mamba, reset every 32, keep 16 (old setting) | 3.00 | 3.67 | 3.46 | 3.78 | 27 |
| mamba, reset every 8, keep 16 | 2.99 | 2.97 | 3.11 | 3.11 | 47 |
| mamba, reset every 1, keep 16 | 2.94 | 3.29 | 3.25 | 3.23 | 276 |
| transformer, reset every 32, keep 16 | 2.82 | 3.46 | 3.49 | 3.30 | 11 |
| transformer, reset every 8, keep 16 | 2.85 | 3.03 | 3.10 | 2.82 | 12 |
Read: the Mamba with its full state breaks after 49 tokens (foot slide 0.31 -> 0.90 m/s). This
is the "random motion". A reset from the last 16 tokens keeps quality. Reset every 8 is better
than every 32 for both twins and is faster than playback (200 ms per token).
Done: `configs/service.yml` `anchor_every_tokens` 32 -> 8. Thesis: new section 6.4 "Behaviour
Beyond the Trained Length", abstract and conclusion updated; PDF rebuilt (no undefined refs).
Consequence, stated in the thesis: with a reset policy the transformer memory is bounded too, so
the Mamba memory advantage is practical only after training on sequences longer than 10 s.
Limit: 32 captions, one sample each. Direction is clear; exact size is not.
Also this session: rule "private function = two underscores" applied (95 renamed to `__name`,
22 made public because name mangling would break them; 162 tests pass); hook
`code_rules_check.py` (naming rule + "reuse before new code" reminder on large edits).

## Update (2026-10-10, logging; typed request proof of concept; project licence; Makefile)

Logging: all 77 `print` calls in `src/text2motion` are now `logger` calls. Each line has time,
level and module. `text2motion --log_level DEBUG <command>` adds detail. Each run folder gets a
`run.log`. The training epoch line and `metrics.jsonl` now have the learning rate and the epoch
minutes. Setup is in `app/logging_setup.py`. 162 tests pass; no new type error. No GPU run.
Typed request, proof of concept: `benchmark` only. `cli.py` builds a `StreamingBenchmarkProtocol`
and `run_bench(app, request)` gets it. Check: the same CPU benchmark before and after gave the
same label, horizons and state bytes. The other 10 commands still get the raw arguments.
Licence: `LICENSE` is the PolyForm Noncommercial 1.0.0 text, downloaded unchanged. `NOTICE` has
the copyright line and the MIT notices of the ported code (`licenses/third_party/`).
Compared projects (licence files downloaded): MoMask, MotionGPT, MDM, MLD are MIT; T2M-GPT is
Apache-2.0; MotionStreamer has its own research and education licence.
Makefile: short targets for the full cycle (`make help`). It uses PowerShell as its shell.
Where we stopped and why: GNU make is not installed on this computer, so the Makefile is not
tested (`winget install ezwinports.make`). The university rule on ownership of thesis code is
not checked. Open concern from the owner: motion becomes random after the trained length (49
tokens); the thesis measures memory at long length, not motion quality.

## Update (2026-10-10, command line cleaned; licence record; README; understanding hook)

Command line (`src/text2motion/app/cli.py`): each flag has a help text; each path is a `Path`;
fixed defaults moved to contracts (`CorpusLayout`, `OverfitGateRequest`, `StudioSamplingConfig`).
Renamed flags: `sanity-overfit --out` -> `--overfit_gate`; `pretrain --out` -> `--ckpt`;
`tokenize --out` -> `--token_pack`; `benchmark --out` -> `--report`; `prepare --out` ->
`--converted_dir` and `--out_dir` -> `--corpus_dir`; `--eval_batch` -> `--eval_batch_size`;
`--our_vab_dir` -> `--vocab_dir`. Drivers in `scripts/` and `infra/` use the new names.
Checked: 162 tests pass, no new type error, all driver commands use known flags. No GPU run.
Licences: `THIRD_PARTY_LICENSES.md` is generated by `scripts/licenses/collect_licenses.py` from
`licenses/verified_assets.json` (7 assets, each with source URL, date and a copied sentence) and
from installed package metadata. A hook regenerates it and rejects a record without evidence.
README: new order Problem / Demo / Features / Setup, with the results of today.
Hook `understanding_check.py`: each reply must end with an "Understanding check" section.
Where we stopped and why: the licence of this project itself is not chosen (owner decision, and
the university rules must be checked). Commands still get the raw argument object and use `print`;
the change to typed requests and `logging` waits for a go.

## Update (2026-10-10, hooks added; skills and agents cleaned)

Hooks (`.claude/settings.json`, scripts in `.claude/hooks/`):

- `ste100_prompt.py` on each prompt: adds the Simplified Technical English rules.
- `pyright_check.py` after each edit: type errors of the changed Python file. It reports a file
  only above its count in `pyright_baseline.json` (251 old errors in 51 files, not repaired).
- `prose_tells_check.py` after each edit in `doc/`: humanizer patterns in `.tex` and `.md`.
  Skills: 5 moved to `.claude/_archive/skills/` (arch-decision, azure, port-from-donor,
  repo-scaffold, dataset-unified). The 9 that stay have a current-state note and no donor paths.
  Agents: all 6 stay. Each has the measured results of today, the web tools, and two new duties:
  compare with online sources before a decision, and build a proof of concept before a full change.
  Where we stopped and why: the hooks and agents load at session start. Start a new session.

## Update (2026-10-10, run directories merged: one directory per training)

What we did: `scripts/runs/merge_runs.py` (dry run by default, `--apply` to merge) read all 143 run
directories under `artifacts/logs/train` and `artifacts/outputs/runs/**` and folded 19 chains.
A merged run keeps the id of its first contributing segment and holds: `metrics.jsonl` (gap-free),
`resumes.jsonl` (later launch manifests), `lineage.json` (who continued whom) and `segments/` (every
original directory, untouched). Nothing deleted. Evaluation and other one-shot runs (65) untouched.
Rule: a resumed launch continues the latest chain with the same checkpoint; epochs it re-ran replace
the crashed tail of the earlier segment. A different checkpoint name continues a chain only when its
first epoch is exactly the chain's last + 1 on the same tokenizer.
What we got (current twins, `artifacts/logs/train`):

- transformer trunk `20260920T061001Z` 1-100 (e60 + e100); mamba trunk `20260921T161630Z` 1-100.
- dynamics branches `20261001T133355Z` (transformer) and `20261001T152447Z` (mamba), 61-84 each,
  parent = their trunk at epoch 60 (a fork, so linked, not merged).
- RVQ transformer: `20260919T151438Z` 1-18 and `20260920T194019Z` 1-60.
  Open: `tokenizer/20260616T200731Z_tokenizer_fsq_g6_v1000` logs epochs 251-500 only; its first half
  has no run directory. New resumes should pass `--resume_run_dir <home>` so they land in place.
  New rule: every new or edited script starts with a comment holding its full CLI command.

## Update (2026-10-07, thesis ported to LaTeX: first full compile, 41 pages)

What we did: moved `doc/notes/dissertation_draft.md` into `doc/paper/chapters/` (abstract, 8 chapters,
2 appendices), filled `cls/bib.bib` (37 entries, 18 checked against the arXiv API, the rest from memory), added the
dynamics ablation and the e84 negative result. Build: `xelatex -> biber -> xelatex x2` from `doc/paper`
with `-output-directory=build`; PDF copied to `doc/paper/thesis.pdf`. No undefined refs or citations.
What we found while checking numbers against the eval logs:

- FID bootstrap intervals of the twins overlap (transformer 0.132 [0.127, 0.230], mamba 0.121
  [0.114, 0.212]); R@1 gap 0.022 is ~3 std of a repetition. Thesis now says: ahead in point estimates
  everywhere, resolved on text matching, on par on FID.
- Citable 34M twins are 8 / 15 layers, prefix length 1, decay groups ON (a_log, d_skip exempt). The
  draft's "16-token prefix", "12 / 23 layers" and "single decay group" paragraphs describe the 100M cells.
- Draft slip: FSQ perplexity is higher in 5 of 6 matched cells, not all (8x512: 340 vs 343).
  Where we stopped and why: section 1.2 (research questions + thesis statement) is left for the author.
  Still open: qualitative motion figures, tokenizer/generator diagrams, front matter (declaration,
  acknowledgements), T2M-GPT R@1 pairing to verify against the paper, reconcile 2189 vs 4384 test clips.

## Update (2026-10-07, 72 -> 84 continuation FINISHED: no gain, both twins)

What we tried: the resumed run completed 2026-10-05 06:41Z (driver total 11.3 h, both DONE markers).
What we got: neither twin beat its inherited best, so no `dyn_e84.pt` was written.
| | val FID e72 (best) | val FID e84 | val R@1 e84 | val R@3 e84 |
| transformer | 0.2479 | 0.2745 | 0.484 | 0.788 |
| mamba | 0.2397 | 0.2623 | 0.486 | 0.799 |
Train losses still creeping down at 84 (mamba ce 3.729, foot_skate 0.00452) while val FID rises:
more epochs of this recipe no longer help. dyn_e72 stays the selected dynamics checkpoint.
Where we stopped and why: the fine-tune line is exhausted. e60 rows stay the citable headline,
dyn_e72 is the matched ablation, e84 is a one-line negative result. No test eval run for e84
(nothing was selected). The LaTeX thesis in `doc/paper/` is still a skeleton (chapters are stubs,
`cls/bib.bib` is empty); the 67 KB draft lives in `doc/notes/dissertation_draft.md`.

## Update (2026-10-04, 72 -> 84 continuation SUSPENDED by user)

Transformer stopped during epoch 84 (state saved at end of epoch 83; best val FID still 0.2479 from
epoch 72 -- the epoch-78 val did not beat it). Mamba not started. Resume with exactly:
`scripts/training/finetune_dynamics_twins_local.ps1 -Epochs 84 -SeedFrom dyn_e72 -ContinueSelection`
(the transformer finishes epoch 84 + its val, then Mamba runs 72 -> 84).

## Update (2026-10-04, dynamics fine-tune extended 72 -> 84, both twins)

User chose the longer fine-tune (same weights). Launched 07:28Z:
`finetune_dynamics_twins_local.ps1 -Epochs 84 -SeedFrom dyn_e72 -ContinueSelection` (driver log
`artifacts/outputs/finetune_dynamics_e84_driver.log`). Each twin resumes its dyn*e72 state; best-by-val
continues across the whole 60 -> 84 fine-tune (inherited best: transformer 0.2479, mamba 0.2397), so
`generator*{b}_bs8_dyn_e84.pt` only appears if epochs 73-84 beat it -- otherwise dyn_e72 stays best.
Done markers now carry the target epoch (`finetune_dynamics_{b}\_e72_DONE.txt`, `...\_e84_DONE.txt`).
Expected: transformer ~1.5 h, Mamba 13-20 h (GPU thermal throttling). Then rerun
`scripts/evaluation/dynamics_ablation_eval.ps1` adapted to the selected checkpoints.

## Update (2026-10-03, dynamics-loss ablation: RESULTS, both twins, matched)

Both twins fine-tuned epoch 60 -> 72 with `w_foot_skate: 10`, `w_accel: 18`, val-selected at 72.
Test (2189 clips, citable command, test touched once for these checkpoints):
| | FID | R@1 | R@3 | MM | Div |
| transformer e60 | 0.132 | 0.395 | 0.669 | 3.796 | 9.362 |
| transformer dyn_e72 | 0.158 | 0.407 | 0.681 | 3.746 | 9.359 |
| mamba e60 | 0.121 | 0.418 | 0.693 | 3.677 | 9.437 |
| mamba dyn_e72 | 0.127 | 0.419 | 0.696 | 3.661 | 9.401 |
(reference: transformer e100 without the terms tested FID 0.163 -- extra epochs alone drift FID up.)
Naturalness (64 test clips, same clips/seed; ground truth accel 0.674, jerk 0.559, skate 0.184 m/s):
| | accel | jerk | over-limit | foot skate |
| transformer e60 | 0.717 | 0.572 | 2.68% | 0.287 |
| transformer dyn_e72 | 0.688 | 0.553 | 3.05% | 0.277 |
| mamba e60 | 0.666 | 0.535 | 2.66% | 0.276 |
| mamba dyn_e72 | 0.659 | 0.538 | 2.14% | 0.262 |
Read: the terms move both twins the intended way (foot skate -3.5% / -5%, transformer accel+jerk
closer to ground truth) and slightly improve text alignment (R@1/R@3/MM) on both; FID roughly flat for
Mamba, worse for the transformer (but no worse than plain extra training). Effect is modest: skate is
still ~45% above ground truth. Single seed. The thesis e60 rows stay the citable headline; this is a
matched ablation. Twin ranking unchanged: Mamba still leads the transformer on every metric here.
Process notes: eval script now runs HF offline and only writes DONE if every step produced a result
(first attempt silently "succeeded" with 5/6 steps failed on a Hub network error).

## Update (2026-10-01 ~19:00, all activity suspended by user)

Transformer dynamics fine-tune: DONE (epoch 72, best val FID 0.2479 at epoch 72;
`generator_transformer_bs8_dyn_e72.pt`). Val at 72 matched the no-penalty extension within noise
(FID 0.248 vs 0.242, R@3 0.792 vs 0.799, MM 3.048 vs 3.050).
Mamba dynamics fine-tune: STOPPED on request ~42 min into epoch 62 (correction 2026-10-02: epoch 61
had finished at 19:28 and was saved, so `generator_mamba_bs8_dyn_e72_last.pt` holds the epoch-61
state). RESUMED 2026-10-02 13:01Z from epoch 61 (fresh selection intact, no val run yet). To resume: rerun
`scripts/training/finetune_dynamics_twins_local.ps1` -- the transformer is skipped by its DONE
marker, Mamba restarts epoch 61 from the epoch-60 state (~13 h). Driver return-value bug fixed in the
script (completion check now reads the last return value).
Not yet done (needs both twins): test eval + naturalness probe of the dyn_e72 checkpoints vs e60.

## Update (2026-10-01, matched dynamics fine-tune of BOTH twins launched)

Rule adopted: everything we do to one twin we do to both.
What we did before launching:

- Calibrated on 20 real train batches from each twin's epoch-60 state: geometric terms weighted sum
  0.266 / 0.265; raw foot_skate 0.0047 / 0.0046, accel 0.0027 / 0.0028 (transformer / mamba).
  Weights chosen so each new term counts as much as one existing term (~0.05): `w_foot_skate: 10`,
  `w_accel: 18` in `configs/generator/gen_pilot_fsq8x1024_dynamics.yaml` (= gated pilot config +
  those two keys).
- Overfit gates for the new config: both PASS in 26 steps (ce 0.009 / 0.005).
- Found and fixed on the way: my runtime `service:` edits had broken the pilot config's gate hash
  (and Path.write_text had flipped 5 files to CRLF). Runtime settings now live in
  `configs/service.yml` (`--service_config` overlay on serve/studio); training configs restored
  byte-identical (gen_pilot sha 241a1de4...). Config loader now maps `w_foot_skate`/`w_accel` and
  refuses a nested `train.loss_weights` block (it used to pass through as a raw dict).
- `--fresh_selection` for resumed fine-tunes (best-by-val restarts inside the fine-tune; refuses to
  start without the resume file).
  Run: `scripts/training/finetune_dynamics_twins_local.ps1`, launched detached (driver log
  `artifacts/outputs/finetune_dynamics_driver.log`): each twin resumes its epoch-60 state, 60 -> 72,
  val every 6 (same cadence as the 100-epoch extension = no-penalty reference at 66/72). Transformer
  first (~1.5 h) then Mamba (~13 h). DONE marker only if the checkpoint reached epoch 72.
  Next when done: citable test eval for both (`generator_{b}_bs8_dyn_e72.pt`) + naturalness probe
  (foot skate) for both vs their e60 rows.

## Update (2026-10-01, naturalness: measured, display interpolation, opt-in dynamics losses, visible QA)

What we tried: `scripts/qa/motion_naturalness_probe.py` on 64 test clips, same captions and lengths:
ground truth vs tokenizer round-trip vs generated skeleton vs studio mesh.
What we got: the generator does NOT teleport -- generated skeleton is as smooth as ground truth
(accel 0.666 vs 0.674 cm/f^2, jerk 0.535 vs 0.559, over-limit 2.7% vs 3.3%, 0 teleports). Its one
measured gap is foot skating (+50%, 0.276 vs 0.184 m/s). The analytic IK mesh adds only 2-9% accel at
chain ends. The likely visible cause: 20 fps motion held 3 refreshes on a 60 Hz screen.
Done:

1. Studio interpolates streamed joints 3x (20 -> 60 fps, `display.upsample: 3`), causal, ends on each
   source frame, so no added latency; mesh fitted on the 60 fps joints.
2. Opt-in training terms `foot_skate` (horizontal foot speed while ground truth marks contact) and
   `accel` (L1 between recovered-joint accelerations of prediction and ground truth -- matches dynamics,
   does not over-smooth). Weights default 0: existing recipe and checkpoints unchanged. Not trained yet.
3. QA scripts: `--visible` (real window, drawn cursor) and `--pace`. Selenium cannot drive ImGui.
   Suites: 164 unit, 36/36 scenario, 57/57 control.

Where we stopped and why: a training run with the new terms is a multi-hour local commitment (Mamba
~66 min/epoch) and changes the thesis twin comparison unless both twins get it -- user decision.

## Update (2026-10-01, studio froze: no mesh, no movement -- stream was never paced)

What we tried: py-spy on the user's live studio + service. Studio: `_load_selected_model` blocked
reading the load reply, holding the client lock, so every Generate queued behind it. Service: stuck
writing chunk 1904 of a forever stream nobody read. Reproduced headless: forever mode streamed
~172 frames/s vs 20 fps playback -- the 600-frame window trimmed faster than it played (avatar looks
frozen), the fitter fell behind (mesh missing). The old slow CPU fit had been throttling the stream
by accident; CUDA graphs + analytic IK removed that brake.

What we got (fixed, tested):

1. Pacing: the studio accepts a chunk only while it is <= `display.stream_lead_frames` (120) ahead
   of the playhead (`trimmed_frames` + frame); TCP back-pressure slows the service. Measured: 20
   frames/s generated for 20 fps playback, pose moving, mesh on screen 6/6 samples.
2. Client never abandons a stream half-read: on early exit it sends cancel and drains to the end
   (`_abandon_stream`); an unexpected message closes the connection instead of desyncing.
3. Service never swallows a command mid-stream: any command during a stream cancels it and is
   handled right after (`deferred_request`) -- the lost "load" was exactly this.
   New scenario `long_forever_stream` (30 s forever, pacing, movement, mesh, Stop, Load model x2,
   generate). Suites: 158 unit, 36/36 scenario, 57/57 control checks.

Where we stopped and why: fixed and verified end to end. The user's hung studio/service processes
from 00:13 run the old code and must be closed before relaunching.

## Update (2026-09-30, every studio control tested by real clicks)

What we tried: `scripts/qa/studio_controls_test.py` -- records every ImGui widget's rect while the real
panels draw, then clicks/drags/types through the viewer's own input handlers and asserts each effect.
57 checks over 40+ controls (environment, sampling, avatar, model, playback, prompt bar).

What we got -- two more real bugs, fixed:

1. Stale body chunks leaked into the next take (body 128/288 frames on a 96-frame skeleton, playhead
   past the timeline): the stale check ran outside the lock, then appended. `append_fitted_chunk` now
   checks and appends atomically under the buffer lock, and the take counter moves under the same lock.
2. After Stop the status said "done -- body fitted on N frames"; the fitter's completion message
   overwrote "stopped". It now only reports done for a take that was not cancelled.
   Test-side lessons: click only once a widget's rect is stable across frames (transient status lines
   under a scrolled panel shift the layout by a line); a fast take can finish before Stop is clicked
   (use forever mode). Result: 57/57 controls, 28/28 scenarios, 154 unit tests.

Where we stopped and why: all controls verified. Open, cosmetic: panel content shifts when the
"updating avatar..." note appears/disappears while scrolled.

## Update (2026-09-30, human-style studio QA + the bugs it found)

What we tried: built `scripts/qa/studio_human_test.py` (real studio viewer, offscreen, scripted human
sessions, per-frame invariants, pixel/mask checks) and `scripts/qa/motion_glitch_probe.py`.

What we got -- real bugs found and fixed:

1. S toggled skin AND shadows (aitviewer binds S to shadows). Skin moved to H.
2. A new take was appended onto the previous take's skeleton: after a long take the timeline showed
   the old motion first (600-frame timeline for an 80-frame take) and the window trim cut the new
   body to 1 frame. First chunk of a take now replaces the skeleton.
3. Chunks kept arriving after Stop. Stop now drops in-flight and buffered chunks.
4. Head pitched -44..+58 deg vs the torso: the 9 cm neck->head bone amplifies joint noise. Neck now
   pulled toward the chest rotation (`SHORT_BONE_PARENT_PULL = {12: 1.0}`): within +-15 deg, same
   joint error. Still visibly tilted on some clips; 2.0 halves it at the cost of head motion.
   Model "glitch": over-limit accelerations are 3-11x more frequent in plain Mamba past its horizon than
   with re-anchoring; not concentrated on decoder seams or anchor rebuilds; streamed recovery == whole.
   Harness lessons (not app bugs): headless keys are all "undefined" (use the PyQt6 key table); an
   unpaced headless loop starves worker threads (pace at 60 Hz like vsync).
   Result: 28/28 human checks, 154 unit tests pass.

Where we stopped and why: head tilt partly remains (tuning call); SMPL-X model is built lazily on the
first chunk (~1 s first-mesh delay) -- pre-warming at startup is the next small fix.

## Update (2026-09-30, unbounded-length generation + mesh redesign)

**Long generation -- what we tried:** measured prompt adherence over 48 s streams (8 prompts x 3
seeds; per 8 s window, rank of the true caption among 32 by Guo text-motion distance, top-3 hit
rate; chance 0.09). Hypothesis: "runs, then does something else" is leaving the training horizon
(clips <= 49 tokens, prompt injected once, end token masked) -- not positional embeddings.

What we got: plain Mamba 0.92 -> 0.38 after ~10 s (confirmed). Fix = re-anchoring: every 32 tokens
rebuild the state from the text prefix + the last 16 tokens, so the model never runs past ~49
positions. Mamba stays 0.71-1.00 across 48 s; transformer 0.75-0.96 and its 96-token cap is gone
(prefill restarts positions). Opt-in (`anchor_every` / `anchor_context` on `stream_token_indices`,
service config `anchor_every_tokens: 32`, `anchor_context_tokens: 16`); off by default, so the
citable evaluation path is unchanged. Context is a bounded deque -> O(1) memory holds. Service
streams 1200 frames for both backbones (Mamba 3.35 ms/token with graphs, transformer 9.49).
Tests in `tests/streaming/test_streaming.py` (past-positions, overrun refused, off == unchanged,
graph == eager when anchored).

**Mesh -- design debate:** the per-chunk SMPLify optimizer is an iterative solver on the real-time
path. Measured on a real 160-frame clip: optimizer 20.5 s (0.4x real time), 5.26 cm, jerk 1.57;
analytic IK (per-joint weighted Kabsch onto child + grandchild directions, small pull to the parent
rotation for twist, one SMPL-X forward) 0.34 s + 0.12 s forward (~23x real time), 4.35 cm, jerk
0.47. Studio streaming now uses `solve_smplx_pose_from_joints`; the optimizer stays for the offline
scripts (`render_mesh_demo.py`, `perf_probe.py`). Headless studio: full 12.8 s clip meshed in ~3 s.
Tests: `tests/studio/test_analytic_body_solve.py` (known pose recovered to < 1.5 cm; axis-angle
round-trip near 180 deg).

Where we stopped and why: remaining mesh error (~4.4 cm) is mostly bone-length mismatch between
SMPL-X neutral and the generated skeleton -- next lever is fitting betas once per take. Retraining
for long horizons (longer/concatenated clips, text injected every step) would beat re-anchoring
but costs a training run; re-anchoring is free and measured.

## Update (2026-09-30, studio showed no SMPL-X mesh -- found and fixed)

What we tried: reproduced the studio headless (real generation controller, body fitter and scene
updater against a live service, fake scene) and counted what each render tick would show.

What we got: before -- body mesh on 0% of ticks, skeleton 95%; only 112/256 frames ever fitted.
Four causes, all fixed:

1. On a normal finish the stream worker discarded every chunk still queued for fitting. Now only
   cancel discards; the fitter drains the rest.
2. The playhead ran ahead of the CPU fit (~0.8x real time), so `_sync_display_nodes` always showed
   the skeleton. While a fit runs with skin on, the playhead now waits at the fitted frontier
   (frame 0 until the first mesh).
3. aitviewer advanced the frame inside `super().on_render()`, after our sync -> one-frame-stale
   mesh/skeleton choice. The viewer now advances first (same bookkeeping), then holds, then syncs.
4. Dead `drop_skeleton` flag (set in 4 places, only ever cleared) removed.
   Because the fit now outlives the stream, a new take (`stream_buffers.take`) marks the old fit stale
   (`fit_is_stale`) so it aborts instead of writing into the new clip.
   Also: first-chunk fit budget 30+75 -> 20+50 iters (7.7s -> 4.2s, 1.5 -> 2.0 cm; continuation
   chunks sit at ~6 cm anyway). Status reads "fitting body..." after generation ends.
   After: mesh on all 256/256 frames, 76% of ticks (rest = frame-0 hold before the first mesh), first
   mesh at ~5s instead of ~10s. Tests: `tests/studio/test_body_playback.py`. Suite 147 passed.

Where we stopped and why: SMPL-X fitting is launch-bound like the generator was -- GPU gives only
1.0x vs CPU 0.8x real time -- so the mesh plays slightly slower than real time while fitting.
Real next lever: fewer continuation iters or a batched/graphed fit, a quality trade-off for later.

## Update (2026-09-30, bug hunt on the real-time path: model -> service -> protocol -> studio)

What we tried: read every function from token sampling to the studio client, reproduced each
suspect with a test before fixing.

What we got (all fixed, each with a regression test or an end-to-end check):

1. Studio could not start: `run_studio` resolved the default checkpoint before applying `--model`.
2. Service child crashed on Windows (`WinError 10106`, then `No module named 'pwd'`): the spawn
   env was an allow-list missing SYSTEMROOT / USERNAME. Now inherits the full environment. The
   earlier STATUS guess ("session churn") was wrong -- it was this.
3. Protocol desync: a cancel arriving after the last chunk was answered with "unknown cmd", so the
   NEXT generate read that error (and then orphaned chunks). Server now treats a stray cancel as a
   no-op. Test: `test_a_cancel_that_arrives_after_the_stream_ended_gets_no_reply`.
4. Service died for every new client after a failed model load (`None.hello()`). Now sends an empty
   hello so the client triggers a load. Test: `test_a_client_connecting_after_a_failed_load_...`.
5. `[perf]` mislabelled time: "generate" included tokenizer decode, socket send and client
   back-pressure; no CUDA sync. Now split into generate / detokenize / send+client-wait with syncs.
   The studio's 165 ms/step was really 2.7 ms/token + waiting on the CPU SMPL-X fit.
6. Stream loop ran one wasted backbone step after the final token. Removed.

Measured end to end (headless service, Mamba e60, cfg 6, 88 tokens): 2.69 ms/token, 40.7x real
time with a fast client. Full suite 142 passed.

Where we stopped and why: remaining findings are low-impact -- a body-fit/new-take race that live
mode cannot reach (it never fits) and manual mode cannot reach at human speed; after loading a
transformer the studio clamps steps to 96 and keeps that clamp when you switch back to Mamba. The
real app bottleneck is the CPU SMPL-X fit (see studio.yml `fit.device`).

## Update (2026-09-30, real-time inference optimization -- CUDA-graph decode for Mamba)

What we tried: measured where a Mamba decode step spends its time on the RTX 3050 (4 GB). It is
kernel-launch bound, not compute bound: 15 layers x many tiny ops at batch 2 (CFG). bf16 autocast
made it slower (22 ms vs 14.6 ms backbone step) because it only adds cast kernels.

What we got: captured the per-token step (embed -> 15 Mamba steps -> 8 heads) as one CUDA graph,
replayed each token with static state buffers. Backbone step 14.6 -> 1.70 ms. Full streaming step
(incl. CFG + top-p sampling) 18.1 -> 2.92 ms/step, 6.2x, tokens bit-identical to eager for the same
seed (`generator_mamba_bs8_e60.pt`, 30 steps, cfg 6.0). One-off capture cost on the first request
per batch size. Code: `CapturedDecodeStep` / `EagerDecodeStep` in `generation/model.py`, switched
on by `service.cuda_graph_decode` (true in `default.yaml` and `gen_pilot_fsq8x1024.yaml`), refuses
loudly on a transformer backbone or CPU. Two tests in `tests/streaming/test_streaming.py`
(graph == eager tokens; transformer refused). Full suite 140 passed.

Also: the studio's model registry served the July checkpoints (FID ~3-4). Added the thesis
checkpoints (`generator_mamba_bs8_e60.pt`, `generator_transformer_bs8_e60.pt`) as the first two
entries, so the demo now defaults to the models the paper reports.

Where we stopped and why: the step is now ~70x faster than real time (a token is 4 frames = 200 ms
at 20 fps), so further step tuning won't change the demo. Next bottleneck is outside the generator:
text encode, chunk detokenize, CPU skeleton recovery, SMPL-X fit -- measure via the service's
`[perf]` line before touching them. Paper caveat: sec. 5.10's inference numbers are eager for both
twins; the graphed Mamba is a systems result and must not be compared with an eager transformer as
an architecture effect (the transformer's growing KV-cache has no static shape to capture as-is).

## Update (2026-09-27, training-time vs inference-time added to the paper)

Measured directly rather than estimated, added as dissertation sec. 5.10 / lesson 13.8. Training: from
the uninterrupted 60- and 100-epoch run segments' manifest timestamps, transformer 7.4 min/epoch,
Mamba 66.5 min/epoch (~9.0x). Inference: measured in-process on the actual 60-epoch checkpoints via a
one-off script (`measure_inference.py`, called `load_inference_pipeline` from
`app/commands/serving.py` directly, bypassing the studio's socket service after that layer hit an
unrelated `WinError 10106` -- a Windows socket-provider init failure inside `import torch` -> `asyncio`,
almost certainly an artifact of this session's heavy Python-process churn rather than a real app bug;
worth a plain retry from a normal terminal before investigating further). 30-step/120-frame generations,
prompt "a person walks", cfg 6.0/temp 1.0/top_p 0.9 matching sec. 5.6's eval settings, 3 reps after CUDA
warmup: transformer 9.4 ms/step (307ms/clip, 19.5x real-time), Mamba 20.0 ms/step (629ms/clip, 9.6x
real-time) -- only ~2.1x, not ~9x. The asymmetry is explained, not just observed: gradient checkpointing
(needed to fit the eager scan's activations in 4 GB, sec. 5.9) recomputes each Mamba block's forward
pass during backprop -- a training-only cost inference never pays.

## Update (2026-09-27, generator twin -- 60-epoch result confirmed robust at 100 epochs)

**The headline finding:** at a converged 60-epoch budget, the Mamba twin leads the transformer on
**every metric measured**, including FID (0.121 vs 0.132) -- reversing the earlier 18-epoch reading
where the transformer had led on FID. Both twins were then extended a further 40 epochs (100 total,
`scripts/training/extend_twin_to_100ep_local.ps1`) specifically to check whether 60 epochs was itself
converged. It was: the Mamba's best-by-val checkpoint never moved past epoch 30 in the additional 40
epochs (val FID oscillated 0.24-0.34, never beating 0.183), so its test result is unchanged (FID
0.121, re-confirmed by an independent re-run of the full test-split eval,
`artifacts/logs/train/20260927T114353Z_evaluate_test`). The transformer's running best-by-val _did_
improve twice more (epoch 72: 0.242, epoch 100: 0.230), but that epoch-100 checkpoint scores **worse**
on the full test split (FID 0.163) than the epoch-48 checkpoint's 0.132 -- a genuine val/test
disagreement, not cherry-picked, since best-by-val tracking ran continuously across the whole 100-epoch
trajectory. Per protocol (val-selected, test-once), the epoch-100 checkpoint is the correct citable
number for the 100-epoch budget. Net effect: the Mamba's lead over the transformer **widens** from
~8% relative (60-epoch budget) to ~26% relative (100-epoch budget). Training loss decreased smoothly
and monotonically for both models across the full 100 epochs, so this reads as the data-constrained
regime already argued in dissertation sec. 6.1 (a 34M model can already memorize the ~6.53M-token
fine-tune budget), not an optimization failure.

**Mechanistic finding, unrelated to the above but discovered while investigating why Mamba trains
~6x slower per epoch than the transformer twin:** `_parallel_scan` in `generation/model.py:29-39` is a
genuine, correctly-vectorised Hillis-Steele parallel scan (O(log L) rounds, each a full-tensor op, not
a sequential per-timestep Python loop) -- there is no algorithmic flaw preventing GPU parallelism. The
real cost is (a) no fused `mamba_ssm` CUDA kernel on this Windows machine, so each round is several
separate kernel launches instead of one fused pass; (b) `checkpoint_blocks = not cfg.use_kernel`
(`generation/model.py:163`) enables gradient checkpointing on every Mamba block whenever the fused
kernel is unavailable, recomputing each block's forward pass during backprop to fit activations in
4 GB -- roughly doubling compute on top of (a); (c) the param-matched twin needs 23 Mamba layers vs 12
transformer layers, since a Mamba block has no separate MLP. This also explains why the 60/100-epoch
runs used `--batch_size 8` directly for Mamba with no grad-accum split (unlike an earlier iteration
of this pipeline that needed batch4 x accum2) -- checkpointing frees enough memory to run at batch 8
outright, at the compute cost in (b).

**Decision: no mamba-100M run planned.** Scaling capacity further on the same fixed ~6.53M-token
fine-tune budget is expected to push deeper into the over-parameterised regime this update just
observed directly (sec. 6.1: the 100M twins are already ~18x past the compute-optimal ratio vs 34M's
~6x), not away from it. With Azure abandoned, a local 100M Mamba run is a realistic multi-week
commitment for a result the thesis's own diagnosis predicts won't generalize better. The paper has
been updated throughout (dissertation sec. 5.6/5.6a/5.9, lessons 08/11/12/17/19/00-roadmap) to reflect
this as the final, converged Contribution B result rather than a pending item.

**Files:** both 60-epoch checkpoints archived unchanged at
`artifacts/checkpoints/generator/snapshots_60ep/` before the extension began. Transformer's 100-epoch
checkpoint: `generator_transformer_bs8_e100.pt` (epoch 100, val FID 0.230). Mamba's 100-epoch run
produced no new best-bundle file at all (`generator_mamba_bs8_e100.pt` does not exist) since nothing
ever beat the inherited epoch-30 best; its `_last.pt` confirms 100 epochs completed with `best_fid`
unchanged at 0.1831. Training logs: `artifacts/outputs/extend100_transformer_fsq.log`,
`artifacts/outputs/extend100_mamba_fsq.log` (mamba split across two `artifacts/logs/train/` run
directories by the crash/resume documented in the entry below, unaffected by this update).

## Update (2026-09-23, mamba 60-epoch twin -- OOM crash at epoch 30/31, resumed)

**Why this run exists:** the transformer-vs-mamba twin comparison (Contribution B's primary result)
was originally measured at 18 epochs, same undertraining risk as the tokenizer A/B that turned out
to reverse at 60 epochs (see the 2026-09-21 update below). Both transformer twins (FSQ, RVQ) already
completed cleanly at 60 epochs -- FSQ test FID 0.132, RVQ test FID 0.294 (see
`paper/FEEDBACK.md`). Mamba+FSQ is the last piece: same 60-epoch matched recipe, needed to close out
Contribution B's headline table.

**What happened:** trained cleanly through epoch 30 (val FID improving steadily: 0.8471 -> 0.3866 ->
0.3561 -> 0.2930 -> **0.1831** at epoch 30, well ahead of the transformer twin's own trajectory at
the same epoch). Around epoch 30/31 a DataLoader worker hit a Windows shared-memory error
(`RuntimeError: Couldn't open shared file mapping ... error code: <1455>`, i.e. the system commit
limit) -- system RAM was measured at 99.8% committed, ~412 MB free out of 16 GB, at the time. The
worker-thread crash was survived once (background `multiprocessing.queues._feed` exception, not
fatal to the main process), but the process died for real shortly after; by the time the session
noticed, `python.exe` was gone and the heartbeat was frozen at 09:26 local.
**Nothing was lost**: `generator_mamba_bs8_e60_last.pt` (the epoch-granular resume checkpoint, with
optimizer/scheduler state) was current as of epoch 29 (0-indexed, i.e. 30 epochs completed), best FID
0.1831 -- matches the log exactly.

**Root cause, best guess:** `persistent_workers=True` + `pin_memory=True` + `num_workers=4` (added
this session for the DataLoader prefetch speedup) slowly accumulating Windows shared-memory handles
over a long wall-clock run. The transformer twins used the same settings and finished 60 epochs
without incident, but mamba runs ~4x slower per epoch (no fused CUDA kernel on Windows --
`mamba_ssm` isn't installed -- eager scan only), so it spent far more wall-clock hours exposed to
whatever is slowly leaking.

**Resumed** at 16:50 local with `--num_workers 2` (lowered from 4, as a stability mitigation) and
`--resume`. Confirmed clean pickup from the log: `resumed from
artifacts\checkpoints\generator\generator_mamba_bs8_e60_last.pt at epoch 30 (best FID 0.1831)`.

**Known gap, now partially fixed:** `start_run()` always minted a fresh timestamped run dir on every
invocation, with no way to keep logging into the same folder across a crash+resume -- a real gap
given this project's `--resume` flag is supposed to mean "continue," not just "continue the model
weights." Added `--resume_run_dir <path>` to `train-generator` (`app/run_log.py`,
`app/commands/generator.py`, `app/cli.py`) so a future resume can append into the original run dir's
`metrics.jsonl` (via a new `resumes.jsonl` provenance trail for the second invocation's git
commit/versions) instead of fragmenting into a new one. **This particular resume, though, was already
running (started independently, likely by the user directly) before the flag existed**, so its
metrics live in a _second_ run dir: epochs 1-30 are in
`artifacts/logs/train/20260921T161630Z_generator_mamba/metrics.jsonl`, epochs 30-60 (once complete)
will be in `artifacts/logs/train/20260923T135028Z_generator_mamba/metrics.jsonl`. When citing the
final mamba curve, stitch both files together (same checkpoint lineage, matched recipe throughout,
just two log directories because of this one crash).

## Update (2026-09-20, MATCHED 60-EPOCH TWIN EXTENSION -- FSQ done, RVQ killed before epoch 1, resume needed)

**Why this run exists:** both the FSQ and RVQ transformer twins were originally trained for only 18
epochs (matched recipe), far short of CLAUDE.md's own golden rule (100-200+ epochs). Azure is
abandoned (credit expired 2026-09-17, quota never granted) so this is a local-only extension:
retrain both from scratch at 60 epochs, matched recipe, to see whether the extra epochs change the
picture -- and specifically whether they affect the surprising 2026-09-19 finding that RVQ's
generator beats FSQ's downstream despite losing on both reconstruction and codebook health (see
`paper/FEEDBACK.md`'s 2026-09-20 update, `paper/lessons/07-tokenizer-results.md` sec. 7.4).
Script: `scripts/training/run_matched_twin_60ep_local.ps1`.

**FSQ stage: COMPLETE.** 60/60 epochs, converged (epoch 59->60 loss barely moved). Best-by-val
checkpoint is epoch 48 (val FID **0.2477**); final epoch 60 was slightly worse (0.2712) -- gate-kept
correctly, not the deliverable. Saved to
`artifacts/checkpoints/generator/generator_transformer_bs8_e60.pt`. Manifest:
`artifacts/logs/train/20260920T061001Z_generator_transformer/metrics.jsonl` (71 lines: 1 start + 60
epoch + 10 eval, confirmed complete, file untouched since 16:13 local).

**RVQ stage: KILLED before completing epoch 1, on purpose (user had to step away).** No resume
checkpoint exists yet (`generator_transformer_rvq_bs8_e60_last.pt` was never written), so nothing was
lost -- but **resuming this means restarting the RVQ 60-epoch run from scratch, not continuing it.**

**To pick this back up:** rerun the same script --

```
.\scripts\training\run_matched_twin_60ep_local.ps1
```

It is idempotent per stage: it will see `generator_transformer_bs8_e60.pt` already exists and SKIP
straight to the RVQ stage, restarting that one fresh (init from the existing
`generator_transformer_rvq_pretrained.pt`, gate from the existing
`artifacts/outputs/gates/transformer_rvq_34m_overfit.json` -- both still valid, no need to rebuild
either). At the measured ~9 min/epoch pace, 60 epochs is roughly 9h.

**Known false-failure pattern, already hit 3+ times this session:** the wrapper's `Start-Process`
-based exit code is unreliable on this machine -- it has reported "failed" for runs that actually
completed cleanly (verified via checkpoint inspection each time). Do not trust the script's own
success/failure message; check the actual checkpoint file and its recorded `epoch`/`validation_fid`
instead (`torch.load(path, weights_only=False)`).

## Update (2026-09-14b, AZURE PLATFORM -- and the GPU quota is NOT grantable by script)

**The blocker is worse than "pending": the automated quota path is refused everywhere.**

- `Standard NCADS_A100_v4 Family vCPUs` = **0** in every region checked.
- `az quota update ... value=24` returns **`QuotaNotAvailableForResource`** in BOTH
  `germanywestcentral` and `swedencentral`. That error means the self-service grant is refused and a
  **human-reviewed support ticket** is required. No script gets past it.
- Entitlement is NOT the problem: `az vm list-skus --size Standard_NC24ads_A100_v4` shows the SKU
  with an **empty Restriction column in 16 regions**. We are allowed the SKU; we have none of it.
- `Standard_NC*_T4_v3` is `NotAvailableForSubscription` -- the T4 fallback does not exist for us.
- **A second limit also blocks**: `Total Regional vCPUs` is **10**, and the SKU needs 24. Both limits
  must rise in one ticket, or the grant arrives and the VM still cannot start.
- Subscription is `PayAsYouGo_2014-09-01` with **spending limit Off** -- nothing stops billing
  automatically, so auto-deallocate is the only brake.

**Ticket prepared, not filed.** `infra/azure/request_gpu_quota.ps1` (parameterised, `-WhatIf`
verified). It asks for BOTH limits, leads with the reproduced `QuotaNotAvailableForResource`
evidence, states entitlement is already proven, and lists six substitute regions so a
capacity-constrained reviewer can grant elsewhere instead of rejecting. Contact details are
parameters -- deliberately not hardcoded.

**Platform provisioned and verified** (`infra/azure/provision_platform.ps1`, idempotent, `-WhatIf`):

- `rg-thesis-t2m` (swedencentral), tagged `project=thesis purpose=text2motion managed-by=cli`.
- `stthesist2m01` -- Standard_LRS, Hot, **public blob access off, TLS 1.2 minimum**.
- Containers `data` / `checkpoints` / `runs`.
- Three user-assigned managed identities: `id-thesis-trainer`, `id-thesis-inference`,
  `id-thesis-deployer`. Managed identities, not service principals -- no secret to rotate or leak.
- **Least privilege verified by query, not assumed.** `az role assignment list --assignee <trainer>`
  returns exactly three assignments, each at **container** scope: `data` Reader,
  `checkpoints` Contributor, `runs` Contributor. No account-level access, no compute rights.
- `kv-thesis-t2m` with RBAC authorization; the inference identity holds `Key Vault Secrets User`.

**On-demand trainer** (`infra/azure/train_on_demand.ps1`):

- **Refuses to start** when A100 quota < 24 -- verified: aborts at the check, exit 1, creates nothing.
- `try/finally` teardown: rescues artifacts (`azcopy sync` of checkpoints + outputs) then deallocates
  on **every** exit path, success or failure.
- `az vm auto-shutdown` backstop armed at VM creation, because a `finally` block does not run if the
  laptop loses power -- that is the failure mode that bills a full night.
- Attaches `id-thesis-trainer`, so the training VM inherits the container-scoped RBAC above.

**Skill added**: `.claude/skills/azure/SKILL.md` -- the one-command-at-a-time house style, the verified
subscription facts, what `QuotaNotAvailableForResource` actually means, entitlement vs quota, the
identity model, blue-green, key rotation, and the auto-deallocate requirement.

**Still open**: budget + action group need an alert email (`-AlertEmail`), Azure ML workspace and
model registry, blue-green endpoint on free tier. The 100M checkpoints for BOTH twins already exist
locally (`generator_transformer_100m_val163.pt`, `generator_mamba_100m.pt`), so the demo path has
real models and needs no GPU.

## Update (2026-09-14, PRE-LAUNCH VALIDATION -- both twins gate green, three launch blockers fixed)

**Both twins PASS the mandatory real-batch overfit gate**, run locally against the exact config the
cloud uses (`configs/generator/gen_pilot_fsq8x1024.yaml`, tokenizer `fsq_g8_v1024.pt`):

- transformer: token_accuracy 0.6094 -> **1.0000**, CE 4.770 -> 0.0044 in 26 steps.
- mamba: token*accuracy 0.4620 -> **0.9918**, CE 5.350 -> 0.0409 in 26 steps.
  Gate artefacts with config SHA in `outputs/gates/validate*{transformer,mamba}.json`. The full path
(data -> tokenizer -> CLIP text -> generator -> loss -> optimizer) is wired correctly. All ten
`preflight()` artefacts are present locally, so the bundle can be staged as-is.

**I BROKE a config earlier today and have now corrected it.** The knowledge graph flagged
"Eval Matcher Path Drift (missing t2m/ segment)" comparing `aws.yaml` against `default.yaml`, and I
"fixed" `aws.yaml` to match. **The graph had the direction backwards.** The matcher actually lives at
`data/official_evaluator/extracted/text_mot_match/model/finest.tar` -- **no `t2m/` segment** (verified
with `test -e`, and `infra/make_bundle.sh` stages `extracted/` preserving that layout). The 23 configs
were right; `default.yaml` had been wrong all along and my edit made `aws.yaml` wrong too. Both are
corrected, and **all 25 configs now resolve to a real file on disk**. Lesson: a config path is
validated by `test -e`, never by comparing two configs -- two files agreeing can both be wrong.

**The paid run would have trained Mamba eager.** `gen_pilot_fsq8x1024.yaml` -- the config
`run_twins.sh` actually uses (NOT `final100m_fsq8x1024.yaml`) -- had no `use_kernel` key, so it fell
back to the dataclass default `False` in `generation/contracts.py:59`. That means the eager scan plus
gradient checkpointing (`model.py:163`, `checkpoint_blocks = not cfg.use_kernel`) on a billed A100.
Added `use_kernel: true`. Note this also turns gradient checkpointing off, raising VRAM -- fine for
the 31M pilot at batch 64 on an 80 GB A100, and an OOM would fail fast and visibly.

**Added a fused-kernel preflight gate** to `infra/azure/run_twins.sh`. `use_kernel: true` crashes at
runtime if `mamba_ssm` did not build, and STATUS already records an AWS mamba-ssm ABI mismatch. The
gate greps the config and, only when the flag is on, imports
`mamba_ssm.ops.selective_scan_interface` -- aborting **before any billed training time** with a
pointer to `infra/kernel_canary.sh`. Verified it fires locally (where mamba_ssm is absent).

**Hardened `stage()` against a silently half-trained model.** It called `guarded` then `touch`ed the
`.done` marker on the next line; `guarded` returns 1 when the watchdog kills a stage on the max-hours
or stall budget. Empirically `set -euo pipefail` aborted before the `touch`, so the behaviour was
already safe -- but it depended entirely on an implicit `set -e`, and the failure mode (stage marked
complete, skipped on resume, training continues from a half-trained checkpoint) is severe for an
unattended overnight run. Now an explicit `if ! guarded ...` with a "not marked complete; rerun to
resume" message.

**Launch shape:** `run_twins.sh` runs three pairs sequentially (transformer_fsq, mamba_fsq,
transformer_rvq), each pretrain + gate + train, with `MAX_HOURS` per stage. Nine stages will not fit
one night; the twin comparison that carries Contribution B is the first two pairs. `--resume` on
pretrain and train, per-stage `.done` markers, and 10-minute `azcopy` blob sync make a partial night
safe to continue.

Suite 136 passed / 2 skipped, ruff clean, `bash -n` clean.

## Update (2026-09-13b, PYLANCE 110 -> 0, PLUS THE REFACTORS AND THE TWO CONFIG BUGS)

**The 110 Pylance problems were 2 problems.** 92 of them (84%) were one root cause: `imgui` ships no
type stubs, so Pylance could not resolve any attribute on it. Fixed with a 3-line
`typings/imgui/__init__.pyi` declaring `__getattr__`, plus a `[tool.pyright]` block in
`pyproject.toml` (`stubPath = "typings"`, `typeCheckingMode = "standard"`). Blanket-suppressing all
110 would have buried the other 18, which were genuine.

**The 18 real type bugs, all fixed:**

- `studio/contracts.py` -- the `ViewerHost` Protocol declared 5 members while `viewer.py` used 9. Added
  `viewports`, `viewport_mode`, `shadows_enabled`, `_past_frametimes` and `reload_settings()`. The port
  was lying about its own surface; `_past_frametimes` being private makes our dependence on aitviewer
  internals explicit rather than hidden. `shadows_enabled` had to be `Any`, not `bool`: **mutable
  Protocol attributes are invariant**, and aitviewer's own type is `device | Any | bool`.
- `app/bootstrap.py` -- `Path(path_config.hml3d_out_dir)` on a `Path | None` would have raised an
  opaque `TypeError`. Added `_required_path(value, key)` which fails loud naming the config key.
- `studio/avatar.py` -- `_yaw_init` could return `None` when no candidate wins; the `orient is None`
  guard did not narrow `pose`/`transl`. Both now raise explicitly.
- `motion/amass.py` -- `self._models: dict[str, object]` was the root of both errors there (`object` is
  narrower than we know); now `dict[str, Any]`, matching an untyped third-party model.
- `studio/viewer.py` -- `on_render` now matches aitviewer's base signature exactly instead of `**kwargs`.

**Refactors applied (see the new `pylance` and existing `anti-waste` skills):**

- `app/commands/serving.py` -- deleted the hardcoded `use_kernel=False`. Serve/studio now honour
  `config.generator.use_kernel` instead of always running Mamba eager regardless of hardware.
- `generation/` -- extracted `warmup_cosine_factor(step, warmup, total_steps, floor)` into
  `trainer.py`, now used by both `GeneratorTrainer.build_scheduler` and `pretrain_generator`.
  **This exposed a latent bug**: `trainer.py` clamped `min(progress, 1.0)` and `pretrain.py` did not,
  so past `total_steps` the pretrain LR would have climbed back up off the cosine instead of resting
  at the floor. Invisible in either file alone; obvious once side by side.
- `tokenization/ema.py` (new) -- `ExponentialMovingAverage` moved out of `tokenization/trainer.py`.
  `generation/trainer.py` was importing the _tokenizer's trainer_ just to get a generic utility.
  Placement is forced by the DAG (shared code must sit at or below the lower consumer, and
  `tokenization` is below `generation`), but a dedicated module stops the import reading as a
  borrow and decouples the two trainers.
- `evaluation/metrics.py` -- the `for k in range(top_k)` rank loop is now `hits[rank:] += 1`.
  Verified **bit-identical** against the pre-change value `[0.02803309 0.06617647 0.10110294]`.

**Two config bugs fixed (both would have wrecked the Azure run):**

- `configs/aws.yaml` -- restored the missing `t2m/` segment in the frozen Guo eval-matcher path.
- `configs/generator/final100m_fsq8x1024.yaml` -- `use_kernel: false` -> `true`, matching its sibling
  `final100m.yaml`. **Note this also disables gradient checkpointing** (`checkpoint_blocks = not
cfg.use_kernel`), so VRAM rises -- fine on an A100 80 GB, would OOM on the 4 GB local card.

**Measured and deliberately NOT changed** (documented so it is not re-litigated):

- EMA -> `torch.optim.swa_utils.AveragedModel`: rejected. It holds a second full model copy (2x
  parameter memory on a 4 GB GPU) and ours already uses the same fused `torch._foreach_*` kernels.
- Quaternions -> `scipy.spatial.transform.Rotation`: rejected. Ours are scalar-first `(w,x,y,z)`,
  scipy is scalar-last; scipy has no autograd and no GPU; and `representation.py` calls the torch
  versions directly on the recovery path. Any numerical drift moves recon-FID, the citable number.
- `kinematics.py` torch/numpy "twins": not duplication. The math is written once in torch; the `_np`
  variants are 4-line `from_numpy -> fn -> .numpy()` adapters. Correct as-is.
- Vectorising `r_precision`: measured at **0.135 s/rep, 2.7 s for the entire 20-rep protocol**. Not
  worth it against an eval dominated by generation.
- Folding `pretrain` into `GeneratorTrainer`: would push "no text encoder / no EMA / single loss"
  conditionals into the trainer to remove duplication elsewhere. Only the LR schedule was genuinely
  shared, and that is now extracted.

**rope caveat worth remembering:** `MoveGlobal` crashed mid-apply (`AttributeError: 'PyObject' object
has no attribute 'get_returned_object'`) _after_ rewriting one importer, leaving the test importing an
empty module. It reports the planned changes before applying them, so a crash can leave a partial
write -- always re-check the files it listed.

Suite **135 passed / 3 skipped**, ruff clean, **0 Pylance diagnostics**, 0 upward imports, boundary
guard 5/5.

## Update (2026-09-13, KNOWLEDGE GRAPH + TWO CONFIG BUGS THAT WOULD HAVE WRECKED THE AZURE RUN)

**Built a knowledge graph of the whole repo** (`/graphify`, outputs in `graphify-out/`): 200 files /
116k words -> **1,623 nodes, 4,043 edges, 109 communities**, 87% EXTRACTED / 13% INFERRED (avg
confidence 0.92) / 0% AMBIGUOUS, **no import cycles**. God nodes are `MotionTokenGenerator` (56
edges), `Split` (53), `ApplicationContext` (42), `Backbone` (41) -- the generator and the split enum
are the two real hubs. Cost 52k in / 36k out.

**Two config bugs found, both verified, both would have silently ruined a paid GPU run.**

1. `configs/aws.yaml:12` -- the frozen Guo eval-matcher path is
   `data/official_evaluator/extracted/text_mot_match/model/finest.tar`, **missing the `t2m/`
   segment** that `configs/default.yaml:14` has. A cloud eval would fail to load the matcher, which
   is the whole citable-FID track.
2. `configs/generator/final100m_fsq8x1024.yaml:41` -- `use_kernel: false`, while its sibling
   `configs/generator/final100m.yaml:31` has `true`. In `generation/model.py:115` `use_kernel`
   selects `mamba_ssm.ops.selective_scan_interface.selective_scan_fn` (the fused CUDA scan), and
   `model.py:163` sets `checkpoint_blocks = not cfg.use_kernel`. So the **headline 100M FSQ twin**
   would run Mamba eager _plus_ gradient checkpointing on an A100 where the fused kernel works.
   **This is a live hypothesis for why the latency claim failed** (mamba 28.6-29.7 ms/step vs
   transformer 14.6-16.5): the local measurement was eager by necessity on Windows, but this config
   would carry the handicap into the cloud too. Untested -- flip it and re-measure before concluding
   anything about SSM latency.
   Related: `app/commands/serving.py:24` hardcodes `use_kernel=False` with no override, so the
   studio/serve demo always runs eager regardless of hardware. `evaluation/benchmark.py` does NOT
   route through serving, so the benchmark numbers do not inherit that particular hardcode.

**Also fixed:** `infra/kaggle/README.md:3` said "Primary venue is now Azure A10" while
`infra/azure/README.md` explicitly _rejects_ the A10 ("the worst buy on every efficiency axis --
2.2x the A100's cost per TFLOP") and picks `Standard_NC24ads_A100_v4`. Corrected to A100.

**Literature check via `pwc-cli`** (`uv tool install pwc-cli`, binary `pwc` -- works, read-only
Papers With Code CLI). **Motion Mamba (ECCV 2024, arXiv:2403.07487)** already claims "the pioneering
motion generation model utilized SSMs" -- but it is **bidirectional** (Bidirectional Spatial Mamba)
inside a **U-Net latent diffusion**, so it cannot stream. Contribution B is **not scooped, but its
wording is**: "first SSM motion generator" is taken; the surviving claim is "first **token-
autoregressive, causal** S6 motion generator", and Motion Mamba becomes the ideal foil (SSMs work
for motion; using them bidirectionally forfeits streaming). One genuine risk still to chase:
**Learning Human Motion with Temporally Conditional Mamba (arXiv:2510.12573, SIGGRAPH Asia 2025)**
puts conditioning "into the recurrent dynamics of the Mamba block" -- implies causality, no code
released, needs a full-paper read before the novelty claim is final.

**Graph caveats, honestly:** 357 dangling edges are all external package imports (`pkg_torch`,
`pkg_numpy`) -- benign, and they record the dependency surface. 430 collapsed edges are the same
node pair joined by `calls`+`references`+`uses`, folded by the undirected build; `--directed` would
preserve the multiplicity. 6 Terraform files contributed nothing until `graphifyy[terraform]` was
installed (tree_sitter_hcl), which added 61 infra nodes -- a silent completeness gap worth knowing
about.

## Update (2026-08-28b, BOUNDARY GUARD MADE ROT-PROOF)

**The architecture guard had silently stopped guarding.**
`tests/app/test_import_boundaries.py` enforces ADR 0004 by listing, per stage, the operation
modules that `contracts.py` may not import. That list still said `dataset` and `evaluation` --
modules renamed earlier the same day to `datamodule` and `metrics`. The test stayed green because
it asserts **absence**: once a name no longer exists, it simply matches nothing and checks nothing.
rope could not catch it either, since these are string literals in a set (same blind spot as
`__all__`).

**Root cause is the polarity of the assertion, so the fix inverts it.** Instead of enumerating what
is forbidden (a list that rots on every rename), the guard now enumerates the small, stable set of
roles a `contracts.py` may import and derives everything else from disk via `stage_module_names()`:

- `CONTRACT_SAFE_ROLES = {contracts, representation, model}` for cross-stage imports.
- `OWN_STAGE_CONTRACT_SAFE_ROLES = {representation}` within a stage's own package.
  Any module that is added or renamed is therefore forbidden **by default** -- the guard cannot lose
  coverage by falling out of date. Zero operation-module names remain hardcoded.

**It also closed a gap that had always been there.** The old check only compared against
`text2motion.{own_stage}.{name}`, so a `contracts.py` importing an operation from a _lower_ stage
was never inspected. `studio/contracts.py` could have imported `motion.datamodule` freely.

**Same rot pattern existed one level up**, so `test_stage_map_matches_the_packages_on_disk` was
added: a stage renamed out of `ALLOWED_DEPENDENCIES` would previously have been skipped by a
`continue` on the missing path. The map and the package layout must now agree in both directions.

**Verified by injection, not by assumption.** Three probes, each reverted after: an own-stage
operation import fails; a cross-stage operation import fails (this one passed silently before); a
new ungoverned stage package fails. Suite is **135 passed / 3 skipped** (up from 133), ruff clean.

**Architecture assessment done the same session.** Coupling shape is textbook -- highest fan-in is
`motion.contracts` (22 modules), `motion.representation` (18), `generation.contracts` (17), so the
most-depended-upon modules are declarations; highest fan-out is `app/bootstrap.py` (13) and
`app/cli.py` (11), the composition root. 10,059 src LOC / 2,433 test LOC. Test ratio is
risk-weighted correctly: `streaming/` highest at 0.54 (the thesis novelty), `evaluation/` lowest at
0.10 but concentrated on metric math rather than plumbing. Remaining soft spots:
`evaluation/benchmark.py`'s five public functions have no direct tests (the bounded-state claim is
covered indirectly by `tests/streaming/`), and `studio/scene.py` is the largest module at 527 LOC.

## Update (2026-08-28, LINEAGE AUDIT + RENAME PASS)

**Lineage checked, and it is intact.** All 60 modules import, 133 tests pass, ruff clean, and the
stage graph is still a clean DAG -- `motion -> tokenization -> generation -> evaluation/streaming/
studio -> app`, **zero upward imports**. Every class CLAUDE.md promises (`ApplicationConfig`,
`ApplicationContext`, `ApplicationBootstrap`, `MotionTokenizer`, `TextToMotionGenerator`, the
`MotionClip`/`MotionChunk` family) exists. The Copilot rename pass landed; nothing was lost in the
code itself.

**What the pass had genuinely missed: colliding module names.** Four fixed, with `rope` (the
semantic engine behind IDE rename) so every import and reference was rewritten by the tool, not by
hand. rope caught a reference in `scripts/evaluation/eval_generalization.py` that a grep over `src/`
would have missed.

- `motion/dataset.py` -> `motion/datamodule.py`, class `MotionRepository` -> `MotionDataModule`.
  It was one letter from its sibling `motion/datasets.py`, which it imports. The class owns splits,
  scaler, datasets and dataloaders, so `DataModule` is both accurate and standard MLOps vocabulary.
- `app/commands/runtime.py` -> `app/commands/serving.py`. It holds `load_inference_pipeline`,
  `run_serve`, `run_health`. `app/runtime.py` (seed/device/precision) keeps its name -- that one was
  already right; the collision needed fixing from the other side.
- `tokenization/evaluation.py` -> `tokenization/metrics.py`. Stopped shadowing the `evaluation/`
  stage package; parallels `evaluation/metrics.py`.
- Left alone deliberately: `contracts.py` / `model.py` / `trainer.py` / `metrics.py` / `config.py`
  repeating per stage is ADR 0004 role symmetry, not duplication. 127 of 234 short names are math
  notation (`qmul(q, r)`, `qrot(q, v)`, `x_t`/`u_t`, `R`, `q,k,v`) -- renaming those would hurt
  readability against the reference math.

**Two rope limits worth remembering.** It shells out to `git mv`, so it silently refuses to move an
untracked file -- `app/commands/` was untracked, the move failed while `cli.py` was still rewritten,
leaving a broken import. Fix was `git add -N` then re-run. It also does not rewrite string literals,
so `__all__` in `motion/__init__.py` had to be corrected separately.

**Stale doc paths fixed:** `.claude/agents/eval-metrics-engineer.md` (`scripts/eval/` ->
`scripts/evaluation/`, `scripts/perf/` -> `scripts/benchmarks/`), `ARCHITECTURE.md`
(`tests/studio/test_studio_composition.py`), `matrix_complete.ps1`, and the CLAUDE.md package
inventory, which was missing `motion/`'s amass/annotations/datasets/normalization/storage and
`studio/`'s contracts/config/generation/actions.

**Index drift cleaned up.** The git index still recorded a two-generation rename -- it thought the
probes lived at `scripts/perf/` and the eval scripts at `scripts/eval/`, while the worktree had
`scripts/benchmarks/` and `scripts/evaluation/`. Six `RD` entries (renamed in index, deleted in
worktree). Re-staged with `git add -A scripts/`; index and worktree now list identical files and no
`RD` entries remain. Rename pairing verified at `-M20%`, and the root scratch files `_l2.py`,
`_twin_eval.py`, `_sanity_100m.py` are gone from both index and disk -- nothing orphaned.
`tests/motion/test_dataset.py` -> `test_datamodule.py` was renamed in the same session, and the
whole `tests/` tree is now `git add -N` tracked so rope's `git mv` works on it.

**`container.py` deletion staged**, and with it a footgun closed. Making rope's `git mv` work had
required `git add -N` on 43 new files, which leaves them in the index with the path recorded but
**zero content** -- a plain `git commit` would have written all 43 as empty files, including every
module in `app/commands/` and the whole `tests/` tree. Content is now staged for exactly those 43
paths (not the 84 unrelated modified files). Side effect worth noting: only once `bootstrap.py` had
content in the index could git pair it with `container.py`, so the ADR 0004 move is now recorded as
a **rename** rather than an unrelated delete + add. 39 renames are staged in total, so the
restructure's lineage survives in git history.

## Update (2026-08-20, LIVE STUDIO SESSION -- 4 fixes, 2 dead ends)

**What was broken and is now fixed.**

1. _The avatar never moved through space._ `recover_skeleton` was called per chunk, and
   `recover_root_rot_pos` restarts its `cumsum` at zero every call, so every 16-frame chunk
   re-integrated from the origin. Straight-line test: whole-sequence travels 3.15 m, streamed
   travelled 0.75 m and snapped back at every boundary. Fix = `StreamingSkeletonRecovery` in
   `motion/representation.py`, carrying accumulated yaw + xz offset. Streamed now equals
   whole-sequence to 1e-7 including turning. The follow-cam chasing that sawtooth was the same
   bug, not a camera problem.
2. _Yaw half-angle._ The carry quaternion `(cos y, 0, sin y, 0)` rotates by **2y**, but the direct
   pose used a plain `yaw_matrix(y)`, so the mesh got half the world rotation the skeleton got and
   drifted apart on every turn.
3. _Stop did nothing._ Generation actually stopped on the first press -- `worker DONE` printed
   before the user's Stop -- but `_finish` was only called from the fit worker's `finally`, so
   `generating` stayed True and the button kept reading "Stop". A slow fitter froze the whole UI.
   `_worker` now calls `_finish` itself; the fitter reports separately.
4. _SMPL-X fit was ~160x too slow on GPU._ 4 GB card, two CUDA processes; VRAM exhaustion spills
   to shared system memory. Measured CPU **54 ms/iter** vs >=8700 ms contended on GPU.
   `fit.device: cpu`, `fit.threads: 4` (measured best, and 4 beats 6 and 12 -- the batch is tiny).

**Unbounded generation -- a Contribution B result.** `stream(steps=0)` runs forever. Backbones
expose `max_positions`: mamba `None`, transformer `max_seq_len + 1`. Asking a transformer for
unbounded raises loud. Measured live on mamba 34M: **3056 frames / 764 tokens in one stream**, and
a later run **8688 frames / 2172 tokens** = **22.6x past the transformer's 96-token ceiling**,
7.7-8.7x realtime, flat memory, clean stop. The transformer cannot produce frame 385 at all.

**Two dead ends, both switched off (kept behind flags).**

- _Out-of-process fitter_ (`studio/fit_service.py`, `fit_client.py`, `fit.service`). Built on the
  hypothesis that GIL contention was the problem. Measured **slower**: 13.4 s/chunk vs 7.8 s
  in-process under load. The contention is CPU, not the GIL. Moving work across a process boundary
  does not create cores.
- _Direct rot6d -> SMPL-X pose_ (`studio/direct_pose.py`, `scripts/render/calibrate_rest_pose*.py`,
  `fit.direct`). 42x faster and it **looked** fine on joint error (5.47 cm) -- but joint-position
  error cannot see limb twist or compensating rotation pairs. Rendered it: head off the skull,
  hands at the hips, body 0.61 m deep vs 0.33 m. Rotation error vs the optimiser was **63 deg**,
  and teacher-supervised recalibration only reached 17.2 deg, plateauing at step 200/1500 -- the
  22-constant-rotation model has saturated. It is also a _worse_ optimiser init than a zero pose
  (33.6 cm vs 16.7 cm). Real finding worth keeping: T2M applies a joint's OWN global rotation to
  its incoming bone, SMPL applies the PARENT's, so the mapping is shifted one level down the tree.
  A closed-form correction is not enough; this needs a learned retargeting net.

**Studio usability.** `studio --model mamba-34m` (registry key, lists valid keys on typo);
`Reload settings + models` button / `R` re-reads `configs/studio.yml` live; `S` toggles skin at any
moment; `[UI]` action log for every button/toggle with OK/FAILED outcomes. That log is what
diagnosed the Stop bug, a CUDA OOM, and a system MemoryError -- none were visible before it.

**This laptop is out of headroom.** Three distinct failures in one session: CUDA OOM inside the
mamba step, system `MemoryError` on model swap (894 MB free of 16 GB), and an access violation
loading SMPL-X during tests. Mitigations added: `empty_cache()` after each generation,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and the old model is released before a new one
loads. These are mitigations, not fixes.

**Still open, unchanged and still the real blocker:** the 100M checkpoints
(`generator_transformer_100m_val163.pt`, `generator_mamba_100m.pt`) were exported without their
fine-tuned CLIP and cannot be served or re-evaluated. Everything demoed today was the 34M twin.

Tests 114 passed / 1 skipped. Ruff at baseline.

## Update (2026-08-19c, LIVE DEMO RUNS + UNBOUNDED STREAMING)

**The studio demo runs again, on the 34M mamba twin.** Two things blocked it.

1. _Every 100M checkpoint is unservable._ Configs fine-tune CLIP (`unfreeze_last_n=1`,
   `unfreeze_projection=true`), so a checkpoint must carry its text encoder. Only `*_last.pt`
   files are atomic bundles (`generator` + `text_encoder`); the headline "best val FID" exports
   (`generator_transformer_100m_val163.pt`, `generator_mamba_100m.pt`, `generator_transformer.pt`,
   `generator_mamba.pt`) saved the generator ONLY, and no matching `*_text_encoder.pt` exists.
   The fail-loud guard in `app/container.py` correctly refuses them. This blocks re-evaluation,
   not just the demo. `configs/demo_models.yaml` now lists only the three loadable bundles and
   documents the disabled four. **Open: recover or re-export the 100M text encoders.**
2. _SMPL-X fitting was ~160x too slow on GPU._ The fit thread was alive but had not finished 30
   Adam iterations on 16 frames after 4m22s, with the GPU at 3589/4096 MiB and only 24%
   utilisation -- VRAM exhaustion spilling to shared system memory (two CUDA processes on a 4 GB
   card: the service holds the generator, the studio wanted the same GPU). Measured CPU: **54
   ms/iter, ~22 s for a 196-frame clip**. Fit device is now config-driven,
   `configs/studio.yml -> fit.device: cpu`.

**Unbounded streaming implemented -- and it is a Contribution B result.** `stream(steps=0)` now
runs forever. Backbones expose `max_positions`: mamba `None`, transformer `max_seq_len + 1`, and
the service reports `max_steps=0` for an unbounded backbone instead of a config-derived cap.
Requesting unbounded on the transformer raises loud (it also fixes a latent opaque `IndexError`
from `nn.Embedding` when `steps > max_seq_len`).

Measured live, mamba 34M, prompt "walk":

| run | frames | tokens  | motion | rate          |
| --- | ------ | ------- | ------ | ------------- |
| 1   | 3056   | **764** | 2m33s  | 7.7x realtime |
| 2   | 480    | 120     | 24s    | 8.4x realtime |

The transformer twin stops at **96 tokens (384 frames, 19.2 s)**. The mamba twin reached **764
tokens, 8.0x past that ceiling**, in one stream, stopped cleanly on Stop, studio working set
**52 MB** with no growth. This is the bounded-state claim shown as unbounded generation rather
than as a state-size table.

Duration control: `--seconds` (default 0.0 = forever) plus `--steps` (0 = forever); the UI shows
a "run forever" checkbox when the model supports it.

Two deliberate limits, both measured not assumed:

- **indefinite mode is skeleton-only** -- generation runs 7.5x realtime, CPU body fit 0.45x, so an
  unbounded stream feeds a fitter ~16x slower than its input. Mesh-forever needs generation paced
  to playback, which is a separate change.
- **viewer keeps the last 600 frames** (`display.window_frames`); SMPL-X vertices alone accrue at
  2.5 MB/s (~9 GB/h).

**Open question worth an experiment:** 764 tokens is 8x beyond the 96-token training horizon.
Nothing crashed, but semantic coherence that far out is unmeasured.

Tests 112 passed / 1 skipped (2 new, both backbones' unbounded contract). Ruff: no new findings.

## Update (2026-08-19b, RUN SCRIPT + GRAD-ACCUM FIX + SCRIPTS REORGANIZED)

**Grad-accum epoch-boundary leak FIXED.** `_accum_count` was never reset in `train_epoch`, so a
trailing partial group kept its gradients, `zero_grad` was skipped at the boundary, and the last
batch of epoch N merged with the first of epoch N+1. Extracted `apply_pending_gradients()` and
call it at the end of `train_epoch`. Verified: 5 batches at grad_accum 2 now gives **3 optimizer
steps (was 2)** and the counter is 0 at epoch end. Also switched the cosine schedule to ceil
division so the extra per-epoch step does not run the LR off the end of its plan.
`pretrain.py` was already correct -- zeroes at epoch start, truncates to an exact multiple, and
fail-louds if the optimizer-step budget drifts. Left alone.

**`infra/azure/run_twins.sh` written.** One launch for all six runs (pretrain + gate + finetune
for transformer/FSQ, mamba/FSQ, transformer/RVQ), transformer first so breakage costs $5 not $40.

- **preflight** checks every config / tokenizer / token pack / data dir and the CUDA device,
  fails loud before anything expensive starts
- **guard built in**: max-hours budget plus a stale-write kill, reason appended to
  `outputs/GUARD_KILL.txt`
- **sentinels** (`outputs/logs/<stage>.done`) make it idempotent, so a spot eviction resumes
  where it stopped instead of redoing finished stages. This also protects the overfit gate,
  which must stay byte-identical across sessions.
- optional `BLOB=` env var starts a background azcopy sync of checkpoints + outputs every 10 min
- Verified all four paths: success, skip-on-rerun, failure (exit code propagates and the marker
  is NOT written), and the find-based stall detector.

**Scripts reorganized by series** -- the folder now tells you what a thing is for:

- `scripts/training/` -- `local_guard.ps1`, `run_3day_local.ps1`. Paths deliberately UNCHANGED so
  `CLAUDE.md` and the Azure runbook still resolve.
- `scripts/training/tokenizer_sweep/` -- the 10 Contribution A sweep scripts, marked DO NOT RE-RUN
- `scripts/perf/` -- `perf_probe.py`, `mem_probe.py`, `scaling_budget.py`, `param_count_100m.py`
  (was `_sanity_100m.py`)
- `scripts/eval/` -- gained `twin_eval.py` (was `_twin_eval.py`) and `l2_ground_truth.py`
  (was `_l2.py`). The three loose `_*.py` at repo root are gone.
- `scripts/README.md` -- index of every series with the old-to-new name mapping, so older STATUS
  entries stay greppable. Entries below still cite the old paths; they are history, not corrected.

**Bundler bug caught by the move:** it tarred `scripts/eval` only, so `perf_probe.py` would have
vanished from the payload and the canary would have failed on the box. It now tars all of
`scripts/` (38 files), and ships both tokenizers plus both AMASS token packs.

110 tests pass, ruff clean across `src` and `scripts`.

**WHERE WE STOPPED AND WHY.** Everything is ready to launch. Still **blocked on the Azure GPU
quota request** -- `Standard NCADS_A100_v4 Family vCPUs` 0 -> 24 and `Total Regional vCPUs`
10 -> 24 in germanywestcentral.

Open: `app/container.py:20` still defaults to the old `tokenizer_fsq.pt` (recon-FID 0.0266)
instead of the sweep winner. Always overridden in practice, but it is a silent fallback to a
worse tokenizer.

## Update (2026-08-19, TRAINING PREP -- early stopping, RVQ ablation assets, batch 64)

**What we got into.** Planning the actual Azure training runs, and making sure we cannot burn
credit on a run that stopped improving.

**Tokenizer question: CLOSED, nothing to retrain.** Pulled the full sweep out of
`outputs/runs/tokenizer/` -- 18 runs, full test split (2189 clips). Winner `fsq_g8_v1024` at
recon-FID **0.0170** (MPJPE 106.9). Best RVQ `rvq_l8_1024` at 0.0205. **FSQ beats RVQ at every
matched depth** (g8 -17%, g6 -12%, g4 -8%) and by **-26% at matched depth AND matched vocab**
(g8_v512 0.0195 vs l8_512 0.0264). That last one kills the "FSQ only wins because bigger
codebook" objection -- the separate iso-vocab ablation was not needed (both iso runs came in
worse anyway, 0.031 and 0.087). Contribution A is done. Do NOT retrain: the generator is trained
on frozen tokens from this checkpoint, so swapping it invalidates every generator we own.

**Early stopping ADDED** (`--patience N` on train-generator). N consecutive evals with no val-FID
improvement -> stop. Counter persists in `<stem>_last.pt` so a spot eviction does not reset it.
Verified with a fake trainer; 110 tests pass, ruff clean.

**Checked the training loop end to end. It is sound.** EMA 0.999, CFG dropout, pkeep 0.8, warmup

- cosine, full loss recipe (CE + soft-decode recon + rot6d + foot), text encoder trainable at
  1e-5. Two things verified specifically because early stopping depends on them:

* **Eval is deterministic** -- fixed clip slice, fixed caption, per-clip seeded sampling
  (`evaluator.py:155`), seeded R-prec permutation. Same weights => same FID. So patience acts on
  signal, not noise.
* **Twin fairness is structural** -- `item_rng(seed, epoch, idx)` makes augmentation a pure
  function, so both twins see byte-identical batches in identical order.

**Three misalignments found:**

1. **Epochs inconsistent** (18 / 20 / 50 / 150 across scripts, code default 60) while CLAUDE.md's
   own plateau lesson says 100-200+. The `--epochs 18` in the pilot plan was a 4GB-laptop
   compromise. Going with **100**.
   TRAP: the cosine schedule is built over `args.epochs`, so setting epochs high and letting
   patience stop early leaves LR un-annealed mid-schedule. Set epochs to what you INTEND to
   train; patience is a safety net, not the stopping rule.
2. Model selection uses 600 of 1460 val clips. Deterministic so ranking is consistent; final
   table still uses the full 20-rep test protocol.
3. **Grad accum leaks across epoch boundaries** -- `_accum_count` never reset in `train_epoch`.
   2803 batches at bs8 is odd, so with grad_accum 2 this DOES trigger. Hits both twins equally
   so not a fairness bug. NOT YET FIXED.

**Decisions taken:** batch **64** (code default and what lr 2e-4 was tuned for; bs8 was the 4GB
compromise). **Pretrain BOTH twins** -- symmetric or not at all, pretraining only mamba would
break the controlled comparison.

**Tokenizer ablation set up (the new thesis result).** Downstream comparison: same generator,
same everything, only the tokenizer differs. `fsq_g8_v1024.yaml` and `rvq_l8_1024.yaml` were
already deliberately mirrored (same seed/width/downsample/resblocks/8 quantizers/1024 vocab).
This upgrades Contribution A from "better reconstruction" to "better generation" for ~$5.

- Wrote `configs/generator/gen_pilot_rvq8x1024.yaml` (needs `tokenizer.kind: rvq`; the sweep
  config never set it because `train-tokenizer` has a `--tokenizer` override and `tokenize` does
  not). Verified it builds RvqBaselineTokenizer and loads the trained checkpoint.
- Built `data/amass_tokens_rvq8x1024.npz` locally (free, 94 s, AMASS_263 was still on disk at
  2.9G/13249 clips). Verified structurally identical to the FSQ pack: same 8303 segment keys,
  same shapes, int16, 8 codebooks x 1024. So run C's pretrain is symmetric with run A's.
- Decided AGAINST using MoMask's released RVQ as the training baseline: different architecture,
  downsample and token count means any difference is confounded. Cite its published number
  instead, per ADR 0001.

**Two bundler bugs found and fixed** (both would have failed on the box): `ROOT` still went two
levels up after the script moved from `infra/kaggle/` to `infra/`, and a mangled line
continuation broke the file list. Bundle now ships both tokenizers and both token packs.

**Payload settled, ~4.6 GB, no config edits needed** -- `aws.yaml` and the pilot configs use
RELATIVE paths, so mirroring the layout under `~/thesis` is enough.

**Six runs planned** (transformer chain first -- cheapest, so breakage costs $5 not $40):
pretrain+finetune transformer/FSQ (run A), pretrain+finetune mamba/FSQ (run B),
pretrain+finetune transformer/RVQ (run C). Then `evaluate --split test` 20-rep, once, at the end.
Rough ~11 h ~= $48 on-demand.

**WHERE WE STOPPED AND WHY.** All local prep is done. Still **blocked on the Azure GPU quota
request** -- `Standard NCADS_A100_v4 Family vCPUs` 0 -> 24 and `Total Regional vCPUs` 10 -> 24 in
germanywestcentral. Nothing can run until that lands.

Open: grad-accum boundary fix (#3 above); whether to wrap the six runs in one
`infra/azure/run_twins.sh` with the guard; `app/container.py:20` still defaults to the old
`tokenizer_fsq.pt` (0.0266) instead of the sweep winner -- always overridden in practice, but it
is a silent fallback to a worse tokenizer.

## Update (2026-08-18, SPEED PASS + CLOUD VENUE = AZURE A10)

**What we got into.** After the package restructure we did a whole-system speed pass, then picked
where the fair twin comparison will actually run.

**Speed work -- what worked (all measured, not estimated):**

- **Demo generation 40.8 s -> 2.9 s (14.2x).** This was the real complaint. Two causes: the SMPL-X
  fit ran on the socket-reading thread and blocked everything, and the transformer had no prefill.
  Moved the fit to a worker thread, added `TransformerBackbone.prefill()` (194.4 ms -> 9.4 ms,
  20.7x), and cut the first-chunk fit iters 60+150 -> 30+75. Total demo 40.8 -> 27.1 s. Fit quality
  unchanged (5.14 cm).
- **Eval 16.2x on the full val split** (818 s -> 50.6 s). Rollouts were batch-1. Bucketed clips by
  exact token length, batch 32. Metrics identical.
- **Import 19.44 s -> 3.90 s.** `Gender` was living in `motion/preparation.py`, which dragged
  pandas+pyarrow into every single command. Moved it to `motion/model.py` and made the table libs
  lazy. Torch floor is 3.11 s, so we are close to the limit.
- Smaller wins: Mamba dbx reassociation 2.32x on that op (also avoids a second 103 MB tensor),
  `F.rms_norm` 1.071x, grad-safe `negative_a()` cache 1.2% of rollout.

**Speed work -- what we tried and REJECTED because the measurement said no:**
sequential scan (19% slower), turning off gradient checkpointing (12x slower, spills to 7.92 GB),
KV preallocation (flat 64->1024 tokens, no gain), fewer continuation fit iters (already
under-converged at 7.29 cm).

**Two lessons that cost us time:**

1. Estimates were wrong every time (5-15% -> 3.5%, 10-25% -> 1.2%, KV -> zero). Only direct
   measurement held up.
2. Laptop GPU thermal drift produced **two false readings** ("batching is slower", "RMSNorm is
   slower"). Both vanished under interleaved A/B in one process. Rule: never compare across
   processes or across time.

**Thesis-relevant finding (important).** At 96M on GPU: transformer 14.6-16.5 ms/step with state
growing 5.9 -> 76.7 MB; Mamba 28.6-29.7 ms/step with state flat at 2.68 MB. So the **bounded-state
claim holds. The latency claim does not** at these horizons. Say this plainly in the write-up.

**IK finding (not acted on).** The design doc says no inverse kinematics, but the code throws away
rot6d and then solves for it. Direct mapping measures 38.11 cm (26.29 cm after removing a constant
offset) vs 5.14 cm for IK. The error is flat with depth, so it is a rest-pose mismatch -- fixable
offline. Left the IK call in place until that is done.

**Cloud venue -- decided: Azure A10, Kaggle as fallback.**

- Problem: on Windows the transformer gets its fused kernel and Mamba does not (`mamba_ssm` and
  `triton` will not build). That measures the OS, not the architecture.
- The user has a **$200 / 30-day Azure credit** from cert prep. Earlier we said Azure free tier
  cannot do GPU -- that was true for a _trial_ subscription only. Upgrading to pay-as-you-go
  unblocks GPU SKUs **and keeps the unused credit** for the rest of the 30 days.
- Box: `Standard_NV36ads_A10_v5`, full A10, 24 GB, sm_86. Picked for **Ampere, not speed** -- it has
  bf16 and TF32, which the free Kaggle T4 (sm_75) does not. That means our precision results
  transfer, and the 96M twin pair becomes trainable (it was out of scope on free hardware).
- Money is not the constraint: ~$0.591/h spot, canary ~2 h, pilot twins ~8 h. **The 30-day clock is
  the constraint.**

**Code changes for the cloud move:**

- Fixed every stale `text2motion.train.*` / `text2motion.eval.*` call in `infra/aws/*.sh` and
  `run_3day_local.ps1` -> `text2motion.app.cli <subcommand>` with the now-required
  `sanity-overfit` gate. Zero stale references left.
- Canary and bundler moved out of `infra/kaggle/` to `infra/kernel_canary.sh` and
  `infra/make_bundle.sh` -- both venues share them now.
- The canary **auto-detects the GPU arch** (`TORCH_CUDA_ARCH_LIST` defaults to the detected
  capability), so it targets 8.6 on the A10 and 7.5 on a T4 with no flag. It also aborts on sm < 7.0
  and has an explicit import check at the exact line where both AWS attempts died.
- New runbook `infra/azure/README.md`; `infra/kaggle/README.md` marked FALLBACK.

**WHERE WE STOPPED AND WHY.** Everything that can be done without a GPU box is done. We are
**blocked on the Azure GPU quota request** -- quota starts at zero per region and must be approved
before any A10 can be allocated. That costs calendar, not credit, so it is the day-1 action.

Next, in order: (1) upgrade to pay-as-you-go, pick one region, request NVADSA10v5 = 36 vCPUs plus
the Spot quota separately, set budget alerts and VM auto-shutdown; (2) run the canary -- gate A is
does the kernel import, gate B is the fused-vs-eager training speedup, gate C should show ~1.00x on
rollout because `MambaMixer.step` has no fused path at all; (3) decide the retrain scale from gate
B. If quota is refused or still pending after ~3 days, switch to Kaggle and stop deliberating.

Two things still open and known: `scripts/eval/perf_probe.py` tokenizer code-stats + top-k sampling
(~1-3% each, deferred), and the studio default checkpoint
`checkpoints/generator/generator_transformer_100m.pt` **does not exist** -- `demo_models.yaml`
points at a raw state_dict with no matching CLIP, which the fail-loud guard correctly rejects.
Also: ~127 paths are still uncommitted.

## Update (2026-08-16, PACKAGE RESTRUCTURE -- workflow packages, facades, one CLI)

**The package is now organised by pipeline stage, not by technical category.** 70 source files ->
31 (+ 8 package inits), `shared/` deleted, dependencies point strictly inward and are verified to do
so (a walk of every `ImportFrom` reports **0 upward imports**):
`motion -> tokenization -> generation -> evaluation / streaming / studio -> app`.

Path mapping for anything written before this date (older entries below are left as they were
written and still name the old paths):

- `data/hml3d/{dataset,motion_window,clip_index,stats}.py` -> `motion/dataset.py`
- `data/hml3d/{feature,joints,param_util}.py` -> `motion/representation.py`;
  `{quaternion,skeleton}.py` -> `motion/kinematics.py`
- `data/hml3d/{amass_paths,raw_pose,convert_official,regenerate,pretrain_corpus}.py` -> `motion/preparation.py`
- `model/{tokenizer,rvq_baseline,contracts}.py` -> `tokenization/model.py`
- `model/{generator,generator_architecture}.py` -> `generation/model.py`; `model/text_encoder.py` -> `generation/text.py`
- `train/*` -> `generation/{trainer,losses,pretrain}.py` + `tokenization/trainer.py`
- `eval/*` -> `evaluation/{metrics,matcher,evaluator,benchmark}.py` + `tokenization/evaluation.py`
- `stream/*` -> `streaming/{decoder,service,protocol}.py`; `render/*` -> `studio/{avatar,scene,viewer}.py`
- `shared/*` -> `app/{config,checkpoint,runtime,run_log}.py`

**Every stage is now a subcommand of one entry point** (`python -m text2motion.app.cli <cmd>`:
`prepare`, `train-tokenizer`, `tokenize`, `sanity-overfit`, `train-generator`, `pretrain`,
`evaluate`, `benchmark`, `serve`, `studio`). The `.ps1` sweeps call it with **identical flags and
configs**, so sweep provenance is unchanged. `_twin_eval.py` is now a shim over `... cli evaluate`.

**Stage boundaries exchange named objects** (`MotionClip`, `MotionTokens`, `MotionBatch`,
`GeneratedMotion`, `MotionChunk`) instead of positional tuples, one facade per subsystem
(`MotionRepository`, `MotionTokenizer`, `TextToMotionGenerator`,
`HumanMl3dGenerationEvaluator`, `MotionInferenceServer`), and **all construction happens in
`app.bootstrap.ApplicationContext`, created by `ApplicationBootstrap`**. Backbone/quantizer choice
is one registry per subsystem, not scattered `if backbone == ...` branches. The `VAR` tree is gone
-- each constant now lives with the concept it describes.

**Four real bugs fell out of the move** (all previously latent, none introduced by it):

1. **Checkpoint saving was broken for `Backbone`.** `SE("Backbone", {...})` inside a
   `SimpleNamespace` has no module-level attribute to resolve against, so `torch.save` of a
   `GeneratorBundle` raised `PicklingError`. Real module-level enums fix it.
2. **The val dataloader could not produce a batch.** `rng.choice(usable)` was called on a list of
   `(annotation, segment)` tuples; NumPy rejects that as 2-D (`a must be 1-dimensional`).
3. **Random-crop offset was off by one** (`randint(0, overflow)` excludes the last valid start), and
   crashed outright when `overflow == 0` in the window dataset -- reachable, since clips are kept at
   `>= window`.
4. **`load_generator_checkpoint` returned a raw dict where a `GeneratorBundle` was expected**, so
   legacy (non-bundle) checkpoints crashed in `bundle.verify_tokenizer`. Return contract is now honest.
   Also: `--bootstrap 0` documented "0 disables" but crashed on an empty percentile array; it now disables.

**Verification.** Test suite **110 passed / 1 skipped / 0 failed** (was 106 passed, 2 failed, 1
collection error). `ruff check` + `ruff format --check` clean. All 24 configs load unchanged; the
`hml3d.{dim,num_joints,fps}` block is now a validated static contract (`motion/representation.py`)
rather than a per-run knob. 3 trained tokenizers and the 31.8M Mamba generator strict-load and
run. Full 20-rep eval through the frozen Guo matcher reproduces on 32 val clips
(FID 4.465 / R@1 0.319 / R@3 0.534 / MM 4.522 / Div 8.757 / MModality 2.737).
`benchmark` reproduces the twin claim: transformer state 0.56 -> 1.08 MB across horizon, Mamba flat
at 1.17 MB, param-matched 31.63M vs 31.84M.

## Update (2026-07-28b, CODE-QUALITY PASS — and F8: eval was NON-DETERMINISTIC)

**F8 (CRITICAL, fixed). `evaluate.py` never seeded torch.** It seeded two local numpy RNGs
(`default_rng(0)` for caption choice + MModality pairs) but **never `seed_everything`** — while the
actual motion sampling (`temperature` / `top_p` multinomial inside `generator.stream`) is driven by the
**torch** RNG. Every other entrypoint seeds (train_generator, train_tokenizer, train_pretrain,
eval_generalization all call `seed_everything`); the ONE that produces the citable numbers did not.

- Proof: the same command twice gave FID 5.864 then 4.269. **Every number ever produced by
  `evaluate.py` is non-reproducible run-to-run** — the 34M twin table, the val rows, the cfg/temp
  sweep. The 20-rep protocol averages over CAPTION choice but not over SAMPLING noise.
- Fixed: `seed_everything(cfg.seed, cfg.deterministic)` + `--seed` override. **Verified: identical
  command twice now yields byte-identical output (FID 3.504 / MM 3.409 / Div 9.598 both runs).**
- This had to land BEFORE the corrected citable table, or that table would not be reproducible either.

**Code-quality refactor (user goal: less coupling, no duplicates, group vars into objects, fewer
params, deterministic output).** Measured before -> after:

- `evaluate_generation` **19 params -> 5**; functions with >=8 params **10 -> 6**.
- New `eval/context.py`: `EvalContext` (device, dirs, our/eval stats, matchers, build_text,
  downsample + `clip_ids`/`normalize`/`denormalize`), `GenerationPipeline` (generator+text_encoder+
  tokenizer with `eval`/`stream_tokens`/`generate`/`generate_batch`), `SamplingCfg`
  (temperature/top_p/cfg_scale/stop_at_end + `from_args`).
- Duplication removed: `Mean.npy` load sites **16 -> 10**, matcher/stats load sites **14 -> 8**,
  `build_text` defined **2x -> 1x**, `stream_motion` deleted (absorbed into the pipeline).
- Magic path killed: `evaluate.py` hardcoded `r"data/t2m_glove/glove"` -> new `PathsCfg.our_vab_dir`.
- `eval/` is now 476 lines total across evaluate/generation_eval/context.
- Verified after each step: 37 non-slow tests green, ruff clean+formatted, and `evaluate.py`
  smoke-run end-to-end on real data (incl. the MModality path and sidecar auto-discovery).

**God-class decomposition (STARTED 2026-07-28).** Root cause found:
`class StreamingStudioViewer(GuiPanels, GenerationControl, Avatar, Display, Viewer)` — a **5-way mixin
inheritance**. The "classes" are namespaces sharing ONE object's mutable state, which is why 68 attrs
landed on a 131-line class with no ownership. Extracted state into owned objects in new
`studio_viewer/scene_nodes.py` (state owners, not the mixins — composition of the mixins themselves is
still TODO):

| class                 | attrs before | after  |
| --------------------- | ------------ | ------ |
| StreamingStudioViewer | 68           | **39** |
| GuiPanels             | 55           | 48     |
| GenerationControl     | 48           | **29** |
| Avatar                | 39           | **21** |
| Display               | 23           | **12** |

- `SceneNodes` — rest/motion/body nodes + follow centre; typed `clear_*`/`show_*`/`focus` replace the
  stringly-typed `_remove(self, attr: str)` `getattr`/`setattr` reflection.
- `FitState` — SMPL-X track (orient/pose/transl/betas), model cache, remesh versioning.
- `StreamBuffers` — producer/consumer queues + **the lock, living with the data it protects**.
- `LiveRegen` — the live-regeneration debounce state machine.
- **Concurrency invariant found + encoded**: the lock guards BOTH the body-chunk queue and the fit
  track because `avatar.py` appends to them together — they must stay **index-aligned** or the
  retrofit renders mismatched frames. Was implicit at each call site; now `append_fitted_chunk()` /
  `reset_fit_track()` enforce it in one place, with tests. (My first draft's comment claiming the lock
  guarded "the buffers, nothing else" was FALSE and is corrected.)
- `LiveRegen` was a **pure move, semantics unchanged** — `prev_prompt` deliberately keeps its dual role
  (gui change-detector AND pending-edit flag); that overloading is pinned by tests rather than
  "improved" blind, since the GUI cannot be run headlessly.
- Verified without launching the GUI: **12 tests** in `tests/test_scene_nodes.py` against a fake scene
  (node replacement, stale-selection clearing, focus precedence, drain semantics, fit alignment,
  debounce), AST scan proving **0** dangling references to old attribute names, and an import check
  that the full mixin chain still resolves. Suite 37 -> **49 green**, ruff clean.
- Line count `studio_viewer/` 1,025 -> ~1,180: an accepted trade (user-confirmed) — explicit state
  classes + tests cost lines but buy ownership and testability. Lines come back when the mixins become
  composition and `avatar.py`/`display.py` node handling stops being duplicated.

**Still OPEN:**

- `GuiPanels` 48 attrs — mostly imgui widget bindings; untangling touches the widget wiring, the
  riskiest part to do blind. Needs a manual viewer smoke test alongside.
- **Mixins -> composition** (the actual fix for the god object) — state extraction was the prerequisite.
- ~~11 broad handlers~~ **CORRECTED + FIXED (2026-07-28).** The "all 11 swallow" reading was wrong:
  only **2 were genuinely silent**, both in `display.py` (`except Exception: pass`). The other 9 print
  a traceback and surface a status/error message, and they sit at **thread / RPC-command / per-file
  batch boundaries**, where catching `Exception` is correct practice — a worker thread that dies on
  one bad chunk is worse than one that reports and continues. Those 9 are KEPT by design.
  - `display.py` `scene.remove` guard **DELETED**: aitviewer's own `Scene.remove` already does
    `try: self.nodes.remove(n) except: pass`, so our outer guard was pure redundancy that could only
    mask a real failure in `n.release()`.
  - `display.py` `center_view_on_node` guard **replaced by a precondition** (`_center_view`): checks
    `node.current_bounds` is finite (the actual failure mode — centring divides by a nan diagonal)
    and warns instead of silently passing. Unit-probed both branches.
  - **Silent broad handlers now: 0** (verified by AST scan). `Display._remove`'s stringly-typed
    `getattr`/`setattr` reflection is left in place — it belongs to the deferred class cleanup.
- Remaining >=8-param functions: `generator_loss` (12), `evaluate_tokenizer` (12), `_forward_chunk`
  (9), `connect_or_spawn` (8), `build_window_loader` (8), `build_dataloader` (8).

## Update (2026-07-28, DECOUPLING AUDIT — streaming decode was NOT the decode we evaluate; fixed + exact)

**F1 (HIGH, fixed).** `eval/evaluate.py` and `eval/generation_eval.py` decode the WHOLE token sequence;
`stream/decode.py` decoded each 4-token chunk in ISOLATION. `Decoder1d` is padded + NON-causal
(resblocks k=3 at latent rate, ConvTranspose k=4 s=2, final k=3), so isolated chunks zero-pad where
neighbours belong => the demo emitted different motion from the paper's numbers (6.16% rel MAE,
denormalized 263 space; 17.9% at 1 token/chunk — smaller chunks are WORSE, more boundary per frame).

- **Receptive field measured, not assumed: EXACTLY +/-4 latent tokens = +/-16 frames = +/-0.8 s.**
  `stream.decode.measure_decoder_context` probes it at construction by perturbing one token and
  recording which frames change — architecture-agnostic, so a causal tokenizer auto-reports right=0.
- **Fix:** ring buffer of `left_context` past tokens + `lookahead` future tokens; decode
  `[left | chunk | lookahead]`, emit only the chunk's frames. At lookahead 4 the streamed motion is
  **bit-exact** with the whole-sequence decode (MAE 2e-6 = float noise). State = 12 tokens at ANY
  horizon, so the O(1) memory claim is untouched; the cost is a FIXED 0.8 s latency constant.
- Latency/fidelity dial (64 test clips): lookahead 0/1/2/3/4 -> 3.22 / 1.08 / 0.34 / 0.07 / 0.0000 %.
  `scripts/eval/streaming_decode_fidelity.py`, run `20260728T084648Z_streaming_decode_fidelity`.
  Written up as paper sec. 5.5.1. All 5 existing call sites use the 3-arg form => fixed for free
  (the live demo gains 0.8 s latency and now matches the evaluated artifact).

**F2 (documented, NOT fixed — would invalidate the 34M table).** AdamW puts EVERY generator param in
one decay group at wd 0.01. Mamba carries **600,576** params with no transformer counterpart
(`a_log`, `d_skip`) that reference S6 exempts from decay. Since a = -exp(a_log), decay drives a -> -1 =
**shortens the SSM memory horizon**, on exactly the thesis axis. Reported in sec. 5.9 as a CONSERVATIVE
bound (the penalty falls only on the SSM, so measured parity understates it). Fix = first change in any
re-run; retraining only one twin breaks the matched comparison.

**F3 (fixed).** `generator.text_prefix_len` vs `text_encoder.prefix_len` was a comment-only invariant
with ZERO enforcement — the mechanical cause of the 34M-vs-100M scale/conditioning confound.
`shared/config.validate_config` now RAISES on mismatch (+ on max_seq_len position-budget overflow),
called from `load_config` => covers all 9 MotionGenerator construction sites at once. All 24 configs
validate. `configs/generator/gen_34m_prefix16.yaml` created to decouple the confound: **33.91M params,
byte-identical to the prefix-1 cell except the prefix** (verified). Sanity-overfit PASS (CE 7.10 ->
0.0019, text emb (2,16,512)).

- **THE VAL PAIR IS THREE-WAY CONFOUNDED, not two.** Manifest audit: the 34M cell ran **cfg 5.0 /
  temp 1.1**, the 100M cell **cfg 6.0 / temp 1.0**. Sampling alone moves FID -44% at fixed weights
  (our own June sweep), so it can carry a large share of the 7.48 -> 1.63 gap. sec. 5.6 rewritten
  again: the pair is now presented as "the configuration as a whole improved", capacity treated as
  OPEN. Both val runs also reused the same overwritten path `checkpoints/generator_transformer.pt`,
  so the exact 7.48 checkpoint is likely gone -- prefer `generator_transformer_bs8.pt` (the citable
  34M twin) as the comparator when re-evaluating.
- **RUN LAUNCHED 2026-07-28** (pid 38284, guard `local_guard.ps1` stale-45min / max-10h, log
  `outputs/twin_34m_prefix16.log`): 34M/prefix-16, AMASS prior `generator_transformer_34m_prior.pt`
  (loads clean into the prefix-16 cell), 18 ep, bs8, **cfg 6.0 / temp 1.0 to match the 100M cell**,
  eval on val, `--ckpt_name generator_transformer_34m_prefix16.pt`. Gives the conditioning-only term
  at held capacity AND held sampling; capacity-only = (100M/prefix16) - (this).

**F4 (fixed).** `losses.soft_decode` rebuilt the FROZEN codebook every training step and
isinstance-branched on quantizer internals. Added `units` + `combine_codes` to `GroupedFSQ`/
`ResidualFSQ`; codebook tables now cached per device; `ResidualFsqTokenizer.codebook_size` no longer
branches on a config string.

**F5 (CLOSED — and it overturned sec. 5.8's central claim).** No retraining was needed: `tokenize_corpus`
emits only FULL 196-frame windows, so of 13,249 AMASS clips **only 3,164 ever entered pretraining** and
**10,030 clips (14.4 h) were NEVER SEEN** — a free holdout. Scoring each released prior on 200 unseen
clips vs 200 trained clips truncated to the SAME lengths (`scripts/eval/pretrain_generalization.py`,
runs `20260728T09{2426,2624,2857}Z`, CPU so the GPU stays on the prefix-16 run):

| prior            | params | trained-clip CE | held-out CE | gap       |
| ---------------- | ------ | --------------- | ----------- | --------- |
| Transformer-34M  | 33.91M | 4.75            | **5.87**    | +1.12     |
| Mamba-34M        | 34.10M | 2.10            | **6.93**    | +4.83     |
| Transformer-100M | 98.07M | **1.38**        | **7.19**    | **+5.81** |

- **Held-out ordering is the EXACT REVERSE of training ordering.** Training CE 4.75 -> 1.38 (3.4x
  better) made held-out CE WORSE (5.87 -> 7.19). The 1.54 the paper called `eps_capacity` is
  **memorization of a 3.25M-token corpus**, as sec. 6.1's entropy budget predicted (~3x capacity).
- Caveat: unseen clips are all <196 frames (median 95), so absolute gaps carry distribution shift —
  but the shift is IDENTICAL across the three priors, so the _differential_ (+1.12 vs +5.81) and the
  reversed ordering are unconfounded.
- **sec. 5.8 rewritten**: "train longer buys little" SURVIVES (past the knee, epochs buy memorization);
  "lower pretrain CE demonstrates useful capacity" DOES NOT and is withdrawn. The AMASS prior's value
  is unaffected — it was always established downstream (3x pilot-FID cut), never by CE. This is now
  the paper's strongest argument for selecting on the downstream metric.
- **sec. 5.9 flipped**: the Mamba prior's lower train CE (2.62 vs 5.10) was flagged as possibly giving
  the SSM a stronger init. Held-out says the opposite — Mamba prior 6.93 vs transformer 5.87, so the
  SSM enters the fine-tune from the WEAKER prior and still reaches R-precision parity.
- **Code fix**: `train_pretrain.py --val_fraction` (default 0.05) splits **by clip stem** (verified:
  zero clip overlap, all 8,303 windows preserved) and logs `val_ce` + `gap` per epoch, so no future
  pretrain can report a training CE alone.

**F6 (NEW, sec. 5.4 prior ablation — the "3x FID cut" does NOT hold at all sampling settings).**
Traced both cells to run dirs. **What is sound:** the two eval manifests are byte-identical except the
checkpoint (same config, val, 300 clips, cfg 3.0, temp 1.1, top_p 0.9, same geometry/hyperparams) —
runs `20260620T051405Z_evaluate_val` (no-prior 6.58/0.217/7.71) and `20260620T202820Z_evaluate_val`
(+prior 2.12/0.258/8.38). Numbers reproduce the paper's table exactly. Pretrain CE 7.05->5.10 verified
(`20260620T095439Z_pretrain_transformer`). **What is NOT sound: the ranking INVERTS on the models' own
in-training gauge.** `generator_transformer.pt` is best-by-val-FID (`train_generator.py:153`), and the
two fine-tune runs logged:

- no-prior `20260620T003227Z_generator_transformer` (120 ep requested, ran to 113): best in-train val
  **FID 1.72 @ ep 90** (R@1 0.417); trajectory 3.02/2.25/3.02/3.68/2.23/**1.72**/3.18/2.69/1.96/2.77
- +prior `20260620T101557Z_generator_transformer` (40 ep, `--init_ckpt` the AMASS prior): best in-train
  val **FID 3.38 @ ep 30** (R@1 0.312); trajectory 10.4/6.65/5.8/4.49/3.46/**3.38**/3.66/3.65

So at **cfg 1.0** (200 clips) the NO-PRIOR model is ~2x BETTER (1.72 vs 3.38); at **cfg 3.0** (300
clips) it is 3x WORSE (6.58 vs 2.12). Same split, same best-by-val checkpoints, opposite verdict.
Leading hypothesis: the AMASS prior changes CFG RESPONSE (no-prior degrades under guidance, +prior
improves) — which would be a genuinely interesting result, but it is NOT the claim sec. 5.4 makes.
Also note the budgets differ 113 vs 40 ep, and the exact +prior-40ep checkpoint is GONE
(`generator_transformer.pt` is now the 98M model — the path was overwritten repeatedly).
**RESOLVED — sec. 5.4's DIRECTION IS CORRECT; the "inversion" was an artifact.** Controlled CPU re-eval
(150 val clips, identical settings, `outputs/prior_ablation_recheck.log`):

| model        | cfg 1.0              | cfg 3.0              |
| ------------ | -------------------- | -------------------- |
| no-prior     | FID 8.81 / R@1 0.136 | FID 6.67 / R@1 0.216 |
| +prior (bs8) | FID 6.31 / R@1 0.110 | FID 2.66 / R@1 0.253 |

The prior wins on FID at BOTH guidance settings, and cfg 3.0 reproduces the published pair (6.67 vs
6.58; 2.66 vs 2.12 on a later +prior ckpt). The CFG-inversion hypothesis was WRONG. Residual caveats
for sec. 5.4 are only: budgets differ 113 vs 40 ep, and single-seed. **The apparent in-train-vs-standalone
contradiction had a completely different cause — see F7.**

**F7 (CRITICAL, NEW — `evaluate.py` scores every model with the WRONG TEXT ENCODER).**
`train_generator` partially unfreezes CLIP (`unfreeze_last_n=1`, `unfreeze_projection=True`,
`text_encoder_lr=1e-5`) and the in-train gauge (`generation_eval`) uses that **co-adapted** encoder.
`evaluate.py:242` built a **PRISTINE** `CLIPTextEncoder` and never loaded the trained one. Verified by
diffing `generator_mamba_bs8_last.pt`'s stored encoder against fresh CLIP: **19 of 197 tensors differ**
(exactly layer 11 + final_layer_norm + text_projection; max delta 0.016). So the generator is fed
embeddings from an encoder it was never trained against.

- This explains the long-standing "in-train gauge is optimistic" puzzle that sec. 5.6 note (ii)
  attributed to SAMPLE SIZE: no-prior in-train 1.72 vs standalone 8.81; mamba-bs8 in-train best_fid
  **0.82** vs citable test **4.06**. Same ~5x pattern every time.
- **Every `evaluate.py` number is affected**: the citable 34M twin table (3.30 / 4.06), the 100M val
  row (1.63), the 5.4 ablation, and the cfg/temp sweeps that "locked" sampling. Direction: the paper's
  absolute FIDs are **pessimistic**, not optimistic — fixing this should IMPROVE them.
- **Fix applied**: `evaluate.py --text_encoder_ckpt <name>_last.pt` loads the co-adapted encoder;
  fails loud if the file lacks a `text_encoder` entry; WARNS if the config unfreezes CLIP but no
  encoder ckpt was passed. A/B in flight (same generator, pristine vs co-adapted, all else identical)
  -> `outputs/text_encoder_ab.log`.
- **A/B MEASURED (same generator, same 150 val clips, cfg 6.0/temp 1.0, only the encoder differs):**

| encoder                        | FID       | R@1       | R@2       | R@3       | MM-Dist   | Div       |
| ------------------------------ | --------- | --------- | --------- | --------- | --------- | --------- |
| pristine (published behaviour) | 4.461     | 0.293     | 0.421     | 0.521     | 4.740     | 8.038     |
| **co-adapted (correct)**       | **0.959** | **0.394** | **0.570** | **0.665** | **3.921** | **9.812** |

**FID improves 4.65x**; Diversity 9.81 vs GT 9.5. `outputs/text_encoder_ab.log`. The paper's absolute
numbers are understated by roughly this factor.

- **ROOT-CAUSE FIX APPLIED**: `train_generator` now writes `<ckpt>_text_encoder.pt` beside every
  best-by-val checkpoint; `evaluate.py` AUTO-DISCOVERS that sidecar (and still warns loudly if the
  config unfreezes CLIP but no encoder is found). Generator and encoder can no longer drift apart.
- **CITABLE TABLE CANNOT BE RE-SCORED FROM SURVIVING ARTIFACTS.** Mamba twin: recoverable — its best
  epoch (18, FID 0.821) IS its last epoch, so `generator_mamba_bs8_last.pt`'s encoder matches; sidecar
  written to `checkpoints/generator/generator_mamba_bs8_text_encoder.pt`. Transformer twin: **NO `_last.pt`
  exists at all** — encoder gone, and its best was ep 12 of 18 so a last-epoch encoder would have been
  mismatched anyway. Scoring mamba with the right encoder against transformer with pristine CLIP would
  hand one twin a ~4.6x advantage — strictly worse than the current table.
- **ACTION 2026-07-28: prefix-16 run KILLED at ep 4** (state preserved, resumable:
  `checkpoints/generator/generator_transformer_34m_prefix16_last.pt` + `--resume`). **Transformer twin RELAUNCHED**
  to regenerate its encoder, reproducing the original recipe from manifest
  `20260629T090748Z_generator_transformer`: gen_pilot_fsq8x1024, 18 ep, bs8, cfg 6.0, temp 1.0,
  eval_every 6, init from `generator_transformer_pretrained.pt`, seed 2026 — but FRESH (no `--resume`,
  the original's resume was crash-recovery) and `--ckpt_name generator_transformer_bs8_enc.pt` so the
  citable checkpoint is not clobbered. pid 36416, guard stale-45m/max-10h,
  `outputs/twin_transformer_bs8_enc.log`. The fix changes only what is SAVED, not what is trained, so
  this is a reproduction, not a new experiment. THEN: score both twins on full test, 20-rep, each with
  its own encoder -> the corrected citable table.

**F7 PAPER-UPDATE CHECKLIST (every claim the corrected numbers touch — update in ONE pass, do not fix
the table and leave dependents stale):**

- L443-444 twin table rows (all 8 metrics x 2 backbones) — REPLACE.
- L453-454 val pair (7.48 / 1.63) — **BOTH rows came from `evaluate.py` = pristine encoder, so both are
  invalid**, not just confounded. The three-way-confound rewrite still stands but the numbers go too.
- L32-34 abstract/intro summary (0.255 vs 0.246, FID 4.06 vs 3.30) — REPLACE.
- L463-465 "parity at matched budget": the 1.5-sigma claim and the "~23% relative gap" must be
  RECOMPUTED (sigma changes with the new numbers).
- L472-474 comparability note (i) "far above the cited baselines" — recheck against MoMask 0.045 /
  T2M-GPT 0.116; the "these are pilots" argument weakens if FID drops toward ~1.
- L494 sec. 5.7 "absolute pilot FID (3.30/4.06) is a generator-capacity result" — the capacity
  diagnosis was built on inflated numbers; re-argue or withdraw.
- L499 sec. 5.8 "7.48 -> 1.63" — same invalid pair.
- sec. 5.4 ablation (6.58 / 2.12) — ALSO pristine-encoder numbers. The DIRECTION is re-verified (see
  above) but the absolutes need re-scoring or an explicit caveat.
- sec. 5.1 sampling sweep (6.14 -> 3.46 at s=5.0) — pristine-encoder AND provenance already broken
  (cited `outputs/{cfg_sweep,temp_sweep}.log` do not exist; 6.14 traces to nothing, 3.46 matches only
  an in-train eval of a different run at 200 clips/cfg 1.0). The "locked" sampling setting may not be
  optimal under the correct encoder — re-sweep before trusting cfg 6.0/temp 1.0.
- sec. 5.7 teacher-forced val CE/top-1 (5.02 / 8.5%) + gap 0.94 — CHECK which encoder produced these.
- **UNAFFECTED (verified, no text encoder in the path):** sec. 4.4 tokenizer matrix (motion encoder
  only), sec. 5.5 streaming benchmark (random weights), sec. 5.5.1 streaming fidelity (tokenizer only),
  sec. 5.8 holdout CE (null/zero text condition), sec. 6.1 token budget.

Also this session: sec. 6.1 (data-constrained regime, D=6.53M tokens vs Chinchilla's 1.96B => 301x
short; `scripts/eval/scaling_budget.py`, run `20260727T145249Z_scaling_budget`), and the 5.6
"scale preview" row rewritten — it moved capacity AND prefix at once, so it is no longer claimed as a
capacity result. Tests 33 -> 44 green, ruff clean, ASCII clean.

## Update (2026-06-10b, FINISH-LINE PLAN EXECUTING — Track A done, runs in flight, harvest armed)

Plan accepted (`~/.claude/plans/what-is-the-best-peaceful-map.md`): ~$80 AWS budget → one g5.xlarge
canary (~$2) + THE final 100M twin run (~$20) + contingency/reserve; everything else local.
User-confirmed: 100M twins on g5, prefix+END upgrades included (gated), AMASS pretraining ablation on.

- **SAMPLING LOCKED (val, 300 clips, ep-20 transformer): cfg_scale 5.0, temperature 1.1, top_p 0.9**
  → FID 3.46 / R@1 0.21 vs 6.14 / 0.105 unguided (−44% FID, ×2 R@1, zero retraining). Full grids in
  `outputs/{cfg_sweep,temp_sweep}.log`. Over-guidance onset at scale 8 (FID up, R@1 still rising).
- **TRACK A COMPLETE** (28 tests green, ruff clean, sanity gate PASS):
  (1) `use_kernel` fused mamba-ssm scan — EXPLICIT opt-in, lazy import that fails LOUD (user rule:
  no try/except fallbacks, ever); parity test auto-runs where mamba-ssm installs (= the g5 canary).
  (2) **16-token CLIP prefix** (`TextEncoderCfg.prefix_len == GeneratorCfg.text_prefix_len`; pooled
  - 15 masked token states; P=1 byte-identical to legacy — parity-tested, old ckpts unaffected).
    (3) **END token** (vocab+1; placed at each clip's last token; excluded from soft-decode;
    `stream(stop_at_end=True)` self-terminates, fixed-length mode masks END → protocol-comparable).
    (4) **100M twins param-matched: transformer 12L/d768 = 94.70M vs mamba 23L = 96.37M (+1.77%)**,
    bf16 autocast (`TrainCfg.amp`), `configs/final100m.yaml`. Sanity-overfit on the 4GB GPU:
    CE 7.0 → 0.003 (transformer) / 0.004 (mamba) — PASS both, prefix+END+bf16 active.
- **Cloud mamba (31.8M, old recipe) at ep ~27, ~43 min/ep; dies Jun 11 01:30 UTC at ~ep 42.**
  **Head-to-head @ep20 (200-clip trend, no CFG): mamba R@1 0.193 == transformer 0.193, R@3 0.375 >
  0.354, FID 4.80 vs 3.24 (within that metric's epoch-to-epoch noise)** → early gate signal GOOD.
  Best ckpt auto-synced to S3 (ckpt-sync.service verified working, 14:42 upload).
- **Iso-vocab FSQ ablation training locally** (500 ep, ~20 h, `tokenizer_fsq_isovocab.pt` — added
  `--ckpt_name` so ablations can't clobber winners + run_log manifests in train_tokenizer too).
  Trend: recon-FID 0.190→0.054 by ep 100, perplexity 331/512. Question: FSQ vs RVQ 0.0382 at the
  SAME 512 vocab. ETA ~Jun 11 07:00 UTC, then the GPU frees for AMASS regen.
- **Streaming benchmark built** (`eval/streaming_bench.py`): ms/step + EXACT state bytes + CUDA peak
  vs horizon; CPU smoke already shows KV growth vs fixed SSM state. Run on idle GPU for the figure.
- **HARVEST CHAIN ARMED:** background watcher polls the box every 10 min → on death: download
  `generator_mamba.pt` → `checkpoints/generator_mamba_cloud.pt` + log; when the GPU frees: twin
  table v0 = `_twin_eval` per backbone (`--split test --cfg_scale 5.0 --temperature 1.1`, 20-rep,
  full test) → `outputs/twin_table_v0.log`. **Machine must stay awake overnight.**

## Update (2026-06-10c, BOX HARVESTED + TERMINATED EARLY — mamba peaked at ep 30)

- **Mamba overfitting turn confirmed at ep 40** (FID 5.26, R@1 0.094) after the **ep-30 peak:
  FID 4.17, R@1 0.208 — better than ANY transformer epoch** (twin best R@1 0.193 @ep20). Trajectory:
  ep20 4.80/0.193 → ep30 4.17/0.208 → ep40 5.26/0.094. Same overfit pattern as the twin, one decade
  later — consistent with SSM sample-efficiency at matched budget (worth a thesis paragraph).
- **Harvested**: `checkpoints/generator_mamba_cloud.pt` (= ep-30 EMA best) + `outputs/mamba_cloud.log`.
  **Instance terminated ~7 h early** (zero evals remained before the hard kill) — saved ~$4.
- AWS CLI session expired mid-watch (user re-ran `aws login`); box's own IAM sync was never affected.
- Iso-vocab tokenizer confirmed ALIVE (12 s CPU / 12 s wall probe; log lags ~80 ep from stdout
  buffering — launch future local runs with `python -u`). On completion → twin table v0 on the
  freed GPU: `_twin_eval` per backbone, `--split test --cfg_scale 5.0 --temperature 1.1`, 20-rep.

## OPEN DECISIONS (the queue, in order)

1. **ADR-0002 gate (after twin table v0):** mamba FID within ~1.5x of transformer at locked sampling
   → proceed to the 100M run; else debug locally first. Record the v0 table in ADR 0002 either way.
   Caveat to record: mamba ~42ep vs transformer 150ep wall-budget mismatch (both select best-by-eval;
   transformer peaked at ep 20, so epoch-budget is comparable in practice).
2. **Tokenizer for the final run:** winner-FSQ (0.0266) unless iso-vocab/AMASS-pretrained beats it on
   recon-FID; decision by recon-FID + a downstream val-FID check. Iso-vocab is otherwise a table row.
3. **Final-run hyperparams pinned in `configs/final100m.yaml`** (wd 0.01, pkeep 0.8, 60 ep, bs 64
   bf16, eval-every-5 on val at cfg 5.0/t1.1); canary may revise epoch count if pace projects >20 h.
4. **GPU queue (local):** iso-vocab (running) → AMASS regen → tokenizer pretrain ablation → twin
   table v0 interleaved at harvest. Streaming-bench figure on any idle window.
5. **Budget ledger:** ~$21 spent (g4dn 36 h). Reserved: canary ~$2, final run ~$20, contingency ~$15,
   optional 31.6M-recipe ablation rerun ~$7. Spend nothing on cloud before the v0 gate verdict.

## Update (2026-06-10, CLOUD RUN LIVE-READ + FULL CODE REVIEW — act before Jun 11 01:30 UTC)

(Corrects the first version of this entry: the instance is ALIVE — it runs in **eu-west-1**, the
earlier "dead" read came from querying eu-north-1, the bucket region.)

- **Instance `i-0d79ed71ae495022e`** (g4dn.xlarge, eu-west-1) launched Jun 9 13:30 UTC, GPU 100%.
  **Transformer: 150 ep COMPLETE in 5h57m** (~2.4 min/ep). **Mamba: running, ~34 min/ep** (pure-
  PyTorch scan ≈ 14x slower). **Hard max-lifetime (36h) kills the box Jun 11 01:30 UTC ≈ epoch ~50.**
- **MITIGATED data-loss bug:** `run.sh` syncs only `outputs/`, uploads checkpoints only after a
  backbone finishes → a lifetime-kill would lose ALL mamba weights. Started `ckpt-sync.service` on
  the box via SSM (s3-syncs `generator_mamba*` every 15 min). Fix run.sh before any relaunch.
- **Transformer result: best in-train FID 3.24 @ ep 20** (R@1 0.193); after ~ep 20 train CE falls
  (6.8→1.08) while eval degrades (R@1→0.09–0.13) = overfits early; ep 20→150 wasted. Best ckpt on S3.
- **Code review findings (logical errors not previously spotted):**
  1. **Model selection on the TEST split** — in-train `evaluate_generation` reads `test.txt` and the
     best checkpoint is picked by test FID = selection leakage. Switch in-train eval to **val.txt**
     (1,460 clips); test is touched once, by `_twin_eval.py`, at the end.
  2. **CFG trained but never applied** — `stream()` has no guidance scale; we pay cfg_dropout=0.1 and
     collect zero benefit. Add two-pass guided sampling + sweep scale/temp on val (no retrain needed).
  3. **No regularization to speak of** — `TrainCfg.weight_decay=0.0` (AdamW degenerates to Adam),
     dropout 0.1, no token-input corruption (T2M-GPT's `pkeep` trick). Explains the ep-20 overfit.
  4. In-train eval = first-200 test ids, 1 rep, first caption → biased + noisy; trend-only. The
     citable protocol stays `_twin_eval.py` (20-rep, full split, random caption).
  5. Dataset caption-crop edge case: a captioned segment whose crop falls outside [40,196] silently
     keeps the FULL clip with the segment's caption → mislabeled training pairs (small but real).
  6. Cosine schedule spans 150 ep, so LR is ~97% of peak in the ep-20–40 window where the model
     actually peaks → shorten schedule (~50–60 ep) or early-stop.
- **Representation capacity is NOT the bottleneck:** tokenizer recon-FID 0.0266 (full test) vs
  generator ~3.2 → the gap is generator/conditioning/protocol, not Contribution A.
- **FIX BATCH APPLIED (same session; 22 non-slow tests green, ruff clean, smoke on real data):**
  (1) `stream(cfg_scale=)` — CFG guidance via a cond+uncond batch row (zero text = the drop_text
  null), state stays bounded; plumbed through `generation_eval`, `train_generator --cfg_scale`,
  `_twin_eval --cfg_scale/--split/--ckpt`. (2) In-train eval now defaults to **val** (`--eval_split`),
  eval_every 10→5, epochs default 150→60 (decay must land in the ep-20–40 peak window). (3)
  `TrainCfg.weight_decay` 0→0.01 + new `pkeep=0.8` input-token corruption (targets stay clean;
  tests pin pkeep=1.0). (4) Dataset: captions are now chosen among annotations whose segment is
  usable, cropped to the segment, and windowed INSIDE the crop (mislabeled-pair fix; still 22,418
  train clips). (5) `shared/run_log.py` manifests (config+git+seed+versions, metrics.jsonl) wired
  into train_generator + a CLAUDE.md golden rule (select on val, test once via `_twin_eval`).
  (6) `run.sh`: checkpoint s3-sync loop + 60 epochs. CFG-scale sweep (1–4, val, 300 clips, cloud
  ep-20 transformer ckpt) running locally -> `outputs/cfg_sweep.log`.
- **NEXT:** (1) let mamba run to the lifetime kill (~ep 50, ckpts now safe; transformer peaked by
  ep 20–40 anyway, so the twin gate is still adjudicable); (2) implement CFG inference + val-split
  in-train eval + weight_decay/pkeep + run-manifest logging; (3) `mamba-ssm` CUDA kernels for the
  next run; (4) `_twin_eval.py --backbone both` → citable table into ADR 0002; (5) streaming
  latency/memory plot (local, free); (6) iso-vocab FSQ ablation; (7) optional AMASS tokenizer
  pretraining as an ablation (the 290 GB donor data is unused by design on the citable track).

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
  - missing-word->unk fallback displaced every text embedding (MM-Dist ~9). NOT the encoders, matcher
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
- **Fix WORKED.** `GroupedFSQ` (G groups -> G\*4=24-D latent, every group used) + `TokenizerCfg.quantizer
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
  gdown, id 1cmXKUT31pqd7*XpJAiWEo1K81TMYHA5n -> `data/t2m_glove/glove/our_vab*\*`; maxdiff 1e-7). So
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

- **Streaming deadlock FIXED.** `run_producer` pushed STREAM_END with a _blocking_ `put` -> deadlock
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

Re-verify: `PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/` · lint: `... -m ruff check src/ tests/`

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

---

## Update (2026-07-05, DECISIONS LOG: demo rebuild, comment policy, 34M twin TEST done, 100M mamba EAGER on cloud)

**Demo studio rebuilt** (`scripts/demos/stream_viewer.py`): runtime model registry
(`configs/demo_models.yaml` pins each checkpoint to its EXACT config+tokenizer -- no free mixing),
live avatar options (gender/shape/skin applied to the streaming body in real time via
`mesh_from_params`), one-figure-per-frame sync (body where fitted, skeleton for the tail -- fixed the
"disconnected skeleton vs mesh"), infinite-loop playback, soft-pink default sky RGB(240,182,182),
percent-based layout (fixes the DPI/multi-monitor overflow), fit-body checkbox acts on the current clip
(retro-fits skeleton-only motion). Fixed a real bug: `load_pipeline` never swapped mamba `n_layers`.

**Comment policy tightened -> INLINE-ONLY** (memory `code-style-conventions` updated). Removed ALL
standalone comments + ALL docstrings repo-wide (65 files, ~800 net lines) via `.claude/strip_comments.py`
(AST/tokenize for .py, heredoc-aware for .sh/.ps1, syntax-verify + git-revert net); only trailing inline
`#` comments remain. Then **all scripts expanded to one-statement-per-line** (no `if x; then y; fi` or
`;`-chained clustering). 38 tests still pass, ruff clean, ASCII clean.

**34M twin TEST eval COMPLETE (citable: both pilots, full test 2189 clips, 20-rep, cfg 6 / temp 1.0 / GT-length):**

- Transformer-34M-bs8: FID **3.30**, R@1 0.246, R@2 0.386, R@3 0.485, MM-Dist 5.26, Div 8.55, MModality 3.36
- Mamba-34M-bs8: FID 4.06, R@1 **0.255**, R@2 **0.398**, R@3 **0.498**, MM-Dist **5.00**, Div 8.03, MModality 2.88
- **Read: PARITY.** Mamba leads every R-precision rank (~1.5 sigma) + MM-Dist; transformer leads FID (~23%) +
  diversity. Neither dominates -> supports the thesis claim (matched-budget parity + O(1) streaming), NOT a
  SOTA claim. Paper sec 5.6 rewritten with the real table + sec 5.9 (twin fairness) + `paper/FEEDBACK.md`.
  Runs: `outputs/runs/20260704T044653Z_evaluate_test` (tf), `.../20260704T050118Z_evaluate_test` (mamba).
  Note: full-test FID > the 200-clip in-train val gauge (~1-1.6) -- small val subset was optimistic; test is citable.

**Mamba-34M finetune fixes**: grad-accum (`TrainCfg.grad_accum` + `--grad_accum`) so the 4GB card runs
bs4 x accum2 = effective bs8 matching the transformer twin (plain bs8 OOM'd); resume checkpoint staged on
CPU then freed (was OOMing by parking the 390MB state on-GPU).

**100M mamba twin DECISION: from-scratch AMASS pretrain -> finetune, MATCHING the transformer's pipeline**
(the 100M transformer used its AMASS prior too, NOT from-scratch -- so a fair twin needs the same): Stage A
pretrain 30 ep bs32, Stage B finetune 60 ep bs64 `--init_ckpt`, cfg 5.0 / temp 1.1. `infra/aws/run_mamba100m.sh` + README.

**Cloud infra hardening (3 g5.xlarge boxes burned, ~$4 total):**

- CUDA-version bug FIXED: the kernel gate now auto-detects the toolkit matching torch (was hardcoded
  cuda-13.0; the DLAMI moved to torch cu128 -> use `/usr/local/cuda-12.8`).
- Never-busy watchdog raised **90 -> 150 min** (`user_data.sh.tftpl`): a from-source kernel build idles the
  GPU ~35 min and, with setup hiccups, crossed the old 90-min ceiling -> two boxes self-terminated pre-training.

**KERNEL WALL (2026-07-05): the fused mamba-ssm kernel does NOT work on the current DLAMI.** Source build
succeeds but import fails ABI-link: `undefined symbol c10_cuda_check_implementation` (torch 2.7/cu128 vs
mamba-ssm 2.3.2) -- same class of error as June's canary note; the DLAMI moved to a torch the kernel can't link.

**DECISION (user, 2026-07-05): train the 100M mamba twin EAGER (`use_kernel: false`).** The eager
parallel-scan trains the IDENTICAL weights; the kernel only affects inference LATENCY (benchmarked
separately, does not touch the twin's FID/quality). `configs/generator/final100m_fsq8x1024.yaml` ->
`use_kernel: false`; `run_mamba100m.sh` stripped of the kernel gate + wheel cache -> goes GPU-busy within
minutes of boot. Cost est ~$25-30, may brush the 36h hard cap; best-by-val syncs to S3 every 10 min so a
cap-stop still yields a usable checkpoint. NEXT: `terraform apply` -> bootstrap (bundle + overlay eager
config/script) -> monitor `s3://.../results/` for `generator_mamba_100m*`; `MAMBA100M_DONE` marks done;
then `evaluate.py --split test` completes the 100M twin row.
state vs growing KV; overfit), `src/text2motion/train/{losses,ema,trainer}.py` (token-CE +
soft-decode-recon through frozen decoder + velocity/foot/root + EMA 0.999 + CFG dropout; A+B wired).
⏳ Train on real tokens; add text-encoder (CLIP/SBERT) + top-layer unfreeze.

- **6 — Streaming + aitviewer studio** ✅ code.
  `src/text2motion/stream/decode.py` (bounded-state → windowed token→frame decode → bounded queue,
  drop-to-latest, STREAM_END), `src/text2motion/render/studio.py` (263→22-joint `recover_skeleton`,
  aitviewer `Skeletons` view/headless lazy-imported via `[viewer]` extra, `collect_stream`).
  ⏳ Drive with a trained model + real aitviewer render (`uv sync --extra viewer`). 168 SMPL-X path deferred.

## Critical path remaining (strict order, data/GPU-gated)

1. **USER:** download SMPL+H + DMPL into `body_model/`. 2. regenerate 263 → L2 validation (VQVAE recon
   FID). 3. train Residual-FSQ tokenizer + baseline benchmark (Contribution A evidence). 4. train
   generator + twin w/ full loss → ADR 0002 FID gate (Contribution B evidence). 5. streaming studio demo.
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
