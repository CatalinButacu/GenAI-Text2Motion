# TASKS -- planned work and proposals (written 2026-10-10)

STATUS.md records what is done. This file records what is planned and when.
Each task has an identifier, the question it answers, its cost, and its completion condition.
Rule for all phases: the citable protocol does not change (HumanML3D-263, the Guo evaluator,
2,189 test clips, 20 repetitions, CFG 6.0). New tests are added; they do not replace it.

## Epic template

Each big plan is an epic. An epic starts only after you approve its plan.
An epic does not start until all the epics that it depends on are defined.

- **ID and question:** the question that the epic answers.
- **Use cases:** each case the result must handle, including edge cases.
- **Depends on:** other epics.
- **Time box:** the last day. **Budget:** Copilot calls, GPU hours.
- **Debate:** the options, the measured or published evidence, and the choice.
- **Tasks:** ID, content, finish criterion.
- **Stops:** a plan change, a discovery, a failed criterion, a commit.
- **State:** planned, approved, running, paused or done.

## Epics

Epics: `.claude/docs/epics.csv`. Tasks: `.claude/docs/epic_tasks.csv`. One line each. The full A0 and A1 plan text is in `.claude/_archive/TASKS_epics_2026-11-04.md`.

## Timeline

| Phase | Content | Training necessary | When |
|---|---|---|---|
| 1 | Solid evaluation of the e60 twins | no | 2026-10-10 to 2026-10-14 |
| 2 | Long motion without training | no | 2026-10-14 to 2026-10-24 |
| 3 | Research and small tests before a retrain | no | 2026-10-14 to 2026-10-31 |
| 4 | Retrain of the two twins | yes | 2026-11-01 to 2026-11-21, only after a go from phase 3 |
| 5 | Final evaluation and thesis update | no | 2026-11-21 to 2026-12-01 |

Dates from the owner (2026-10-10): submission is 2027-02-01. All experiments and code end on
2026-12-01. December and January are for the paper only. One 60-epoch Mamba run takes
approximately 3 days, so phase 4 has time for two matched runs of the two twins, not more.
Decision date for the retrain design: 2026-10-31.

## Phase 1 -- Solid evaluation (concern 1: "is the Mamba result real?")

**E1, E2 and E3 are DONE (2026-10-10, run `20261010T144528Z_paired_twin_analysis`); see STATUS.md.**
Do this phase first. It needs no training and it makes each later comparison stronger.
Run E1, E2 and E3 in one evaluation run for each twin (approximately 30 minutes in total).

- **E1 Paired bootstrap on FID.** Question: in what share of resampled test sets is the Mamba
  FID lower? Use the paired bootstrap function in `src/text2motion/evaluation/metrics.py`.
  Done when: the thesis reports the interval of the FID difference.
- **E2 Paired test for each clip.** Question: for each of the 2,189 captions, which twin is
  nearer to its caption (evaluator distance), and which twin gets the caption at rank 1?
  Statistics: sign test or Wilcoxon test on the distance pairs; McNemar test on the rank-1 pairs.
  Done when: the thesis reports a p-value for text match.
- **E3 Physical measures on the full test set.** Question: do foot slide, acceleration and jerk
  agree with the evaluator? Today these are measured on 64 clips only. They do not use the
  evaluator. Done when: a table for 2,189 clips and both twins exists.
- **E4 A second judge.** Question: does a different text-motion model give the same order?
  Needs research first (task R2). Done when: one independent model scored both twins.
- **E5 Blind test with persons.** 20 pairs of clips, same caption, the person does not know the
  model. Done when: the count of preferences for each twin is in the thesis appendix.
- **E6 Second training seed.** The only test that measures variation between training runs.
  Cost: one full run for each twin. Do it in phase 4 if a retrain happens; if not, state the
  single seed as the first limit.

## Phase 2 -- Long motion without training

- **A1 DONE 2026-10-12** (`scripts/qa/action_list_probe.py`, numbers in STATUS.md). 96 sequences,
  not 100, so the pools of 32 are full; 48 tokens for each action, so a reset is on each boundary.
  Open after A1: a smoother boundary (the boundary is about 50% rougher than the middle).
- **A1 Action list (plan outside the model).** Split a prompt into actions. Keep a pointer to
  the current action. At each reset, give the generator the text of the current action and the
  last 16 tokens. Test design:
  1. Build a fixed long-form test split: 100 triples of test captions, seed 2026, saved by clip
     id (no caption text in the repository).
  2. Generate 3 x 49 tokens for each triple, both twins, two methods: (a) one joined caption for
     the full stream, (b) the action list.
  3. Measure, for each action segment: distance to the caption of that action, rank-1 and rank-3
     retrieval, foot slide; and at each action boundary: acceleration and jerk.
  Code change: `stream_token_indices` must accept a text condition for each reset.
  Done when: the table (a) against (b) exists for both twins.
- **A2 DONE 2026-10-12, NEGATIVE.** The one-pass rebuild is correct but slower on the 4 GB GPU
  (47 -> 79 ms/token at reset every 8, keep 16). Code removed. See STATUS.md.
- **A2 Fast state rebuild for the Mamba.** At a reset the Mamba reads the 16 kept tokens one by
  one. Use the parallel scan of training to rebuild the state in one pass. Measure ms per token
  before and after with `scripts/qa/long_stream_probe.py`.
  Done when: the Mamba cost at "reset every 8, keep 16" is measured again.

## Phase 3 -- Research and small tests before a retrain

- **R1 DONE 2026-10-12** (`scripts/qa/memory_probe.py`, numbers in STATUS.md). Both twins use
  mainly the last 8 to 16 tokens; no Mamba state value has a half-life above 48 tokens.
  Not done: the figure (matplotlib is not installed) and the text in the thesis.
- **R1 Effective memory of the trained Mamba.** Question: how many tokens back does the state
  keep information? Method: read the retention factor exp(delta * A) of each layer on test
  streams and convert it to a half-life in tokens; confirm with a perturbation test (change one
  past token, measure the change of the next prediction). Done when: one figure shows the
  half-life for each layer.
- **R2 Literature survey (research agent, online sources only).** Topics: real-time motion
  generation for games; training for long sequences (carried state, joined clips); the BABEL
  labels for AMASS (size, licence, label format); a second evaluator for task E4.
  Done when: a comparison table with a source for each row exists.
- **R3 Compiled scan kernel on Windows. BUILD DONE 2026-10-10.** The kernel builds after 7
  repairs (`doc/notes/implementation-difficulties.md`). One training step of the 34M Mamba:
  957 ms plain, 65 ms with the kernel (14.7 times faster), same output to 6 decimal places.
  Still to do, in this sequence:
  1. Move the import workaround from `scripts/benchmarks/kernel_probe.py` into the package, so
     `use_kernel: true` works in `train-generator`.
  2. DONE 2026-10-12: one full epoch with the kernel is 9.3 minutes (66.5 without it), with
     `--num_workers 0`. Epoch-1 loss 6.399 against 6.408.
  3. DONE 2026-10-12: the overfit gate passes with the kernel in 26 steps.
  If the epoch time is near the transformer's 7.4 minutes, a 60-epoch Mamba run needs hours and
  not 3 days. Then phase 4 has time for more runs and for a second seed (task E6).
  The kernel does not change the step path of streaming. A run with the kernel is a new run;
  do not mix its results with the e60 results.
- **R4 Model size.** The rule of Kaplan et al. (arXiv:2001.08361, equation 4.4) is
  D >= 5e3 * N^0.74 tokens. Our 6.53M to 18M tokens give 16k to 64k parameters by that rule; a
  34M model needs 1.9 billion tokens. The constants come from English text, so the number is not
  exact, but the direction is clear: the limit is data, not parameters. Do not move to 100M.
  Small test: train a 10M pair for the same budget and compare validation FID with the 34M pair.

## Phase 4 -- Retrain proposals (both twins, matched)

Start one of these only after phase 3 gives a go. Each applies to the two twins.

- **T1 Carried state.** Train on long AMASS sequences in consecutive windows and give the state
  of one window to the next (detached). The model learns a state that stays valid at any length.
  AMASS has 40.6 hours; 23 hours are in use. Limit: AMASS has no captions.
- **T2 Captioned long sequences.** First choice: the official HumanML3D `index.csv`. It maps each
  clip to its AMASS source and frames. Measured 2026-10-10: 2,341 AMASS sequences hold 2 or more
  captioned clips; 992 have a captioned span of 20 s or more. Place the captions on the AMASS
  timeline and train on the long sequence. Second choice: BABEL labels (after task R2 confirms
  size and licence). Third choice: captions for unlabeled AMASS parts from the nearest
  HumanML3D clip in the evaluator space (owner's idea; the labels have noise).
- **T5 Fewer and larger Mamba layers at the same width and parameter count** (owner's idea), for
  example 8 layers with expansion 4 in place of 15 layers with expansion 2. Width stays 512, so
  the shared parts of the twins stay identical. Small test first: step time and one short run.
- **T3 Causal tokenizer decoder.** Removes the 0.8 s lookahead. It needs a new tokenizer, so
  each generator must be trained again on the new tokens.
- **T4 Second seed** (task E6) for the final configuration.

## Phase 5 -- Final evaluation and thesis update

- Run the citable protocol one time for each final checkpoint, with E1 to E3.
- Run the long-stream test and the action-list test on the final checkpoints.
- Update chapters 5 and 6, the abstract and the conclusion table. Keep the e60 results as the
  baseline row.

## Disk space (measured 2026-10-10)

Drive D has 5.3 GB free. Drive C has 95.9 GB free. The project on drive C does not read the
donor folder `D:\Facultate\dissertation` (no path in the configs, no link).
Proposed removal, waiting for the owner's confirmation (nothing is removed yet):

| Folder on D | GB | Why it can go |
|---|---:|---|
| `data\inter-x` | 44.4 | two-person data, not used by this project |
| `checkpoints\motion_ssm` | 42.5 | models of the old project, not used |
| `data\humanml3d` | 31.9 | the 272-dimension variant, rejected (not convertible to 263) |
| `checkpoints\rvq_tokenizer` | 8.2 | tokenizer of the old project |
| `data\pahoi` | 5.1 | not used |
| `data\arctic` | 3.9 | not used; the SMPL-X models are in this project |
| Total | 136.0 | |

Keep: `data\amass` (150.8 GB, the raw source for the AMASS conversion and for BABEL),
`data\models_smplx_v1_1.zip`, `data\t2m_download`, and the git repository of the donor.
On drive C, `data\hml3d_official` (4.1 GB) is the download cache of the official release; the
converted data is in `data\HumanML3D_official`. It can go and can be downloaded again.

## Library migration (owner decision 2026-10-12: PyTorch utilities, pydantic, jsonargparse, mlflow)

One library for each step. Each step ends with green tests and one commit. No step changes a
model, so no step needs a retrain.

Measured in a small test on 2026-10-12 (jsonargparse 4.52.0, pydantic 2.14.1):
- jsonargparse shows a `Field(description=...)` as help text for a pydantic `BaseModel`. It does
  not for a pydantic dataclass. Our rule forbids docstrings, so requests must be `BaseModel`.
- jsonargparse accepts an enum by its member NAME only (`MAMBA`), not by its value (`mamba`).
  To keep `--backbone mamba`, a request field uses `Literal["transformer", "mamba"]` and the
  command converts it to the enum.
- A list flag changes form: `--horizons "[4, 8]"`, not `--horizons 4 8`.
- `nested_key=None` keeps the flat flag names of today.

- **L1 Requests as pydantic models + jsonargparse.** Move each help text from `cli.py` into the
  request model as a field description. Build each subcommand from its model. Each command
  receives the typed request (this completes the proof of concept on `benchmark`).
  DONE 2026-10-12: all 11 commands. `cli.py` is 149 lines (707 before). The shared request base
  is in `motion/contracts.py` (the import rule test forbids a new top-level module). An enum flag
  accepts its value (`--backbone mamba`); a bool flag needs no value (`--resume`). All 56 command
  lines in the drivers, the Makefile and the README parse. Not done: one real GPU run.
- **Result 2026-10-12:** L2, L4, L5 applied; L6 partial; L3, L7, L8 rejected by measurement.
  See STATUS.md for the numbers and the reasons.
- **L2 Config validation with pydantic.** Replace `app/config.py` and the manual checks in
  `app/config.py` (296 lines together). Keep the cross-field rules as model validators.
- **L3 Weight average from PyTorch.** Replace `tokenization/ema.py` with
  `torch.optim.swa_utils.AveragedModel`. Check first: resume files of old runs store the old
  format; the best checkpoints must still load.
- **L4 mlflow for run records.** Decide first: `mlflow-skinny` (records only, small) or full
  `mlflow` (with the web interface, large). Keep `manifest.json` and `metrics.jsonl` until the
  thesis tables are checked against the mlflow records.
- **L6 Rotation mathematics from `roma`** (installed, 1.5.6, not used). `motion/kinematics.py` has
  15 quaternion and 6D functions in torch and numpy pairs (about 150 of 318 lines). Risk: the
  263-feature contract. Condition: the contract test gives identical features to 1e-6.
- **L7 Learning-rate schedule from a library.** `warmup_cosine_factor` + `LambdaLR` ->
  `transformers.get_cosine_schedule_with_warmup` (installed). Small. Check the curve is identical.
- **L8 Stream messages.** `MotionStreamProtocol` in `app/service.py` frames JSON and arrays by hand.
  `websockets` is installed. Change only if the studio tests stay green; low priority.
- **Do not replace:** FID and R-precision in `evaluation/metrics.py` (they must match the Guo
  protocol number for number); the FSQ tokenizer and the Mamba block (a library version changes
  the weights layout, so the e60 checkpoints do not load); the studio files (aitviewer already
  does the drawing; the remaining code is our scene logic).
- **L5 Split the two large model files (move only).** `generation/nn_generator.py` (610 lines, 10
  classes) and `tokenization/nn_tokenizer.py` (518 lines, 13 classes).
