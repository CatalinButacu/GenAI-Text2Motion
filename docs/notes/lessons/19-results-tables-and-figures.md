# Lesson 19 -- The results we will report: three tables and the streaming figure

This lesson defines the tables and the figure for the dissertation. Lesson 14 defines the evaluation metrics. Lesson 7 describes the tokenizer experiments. Lesson 13 describes the state of each backbone.

Each result needs a clear protocol and a source record. A historical score describes the model and the evaluator at that time. It does not establish the quality of the current implementation.

The final section contains all measured results. It separates historical scores, current checks, published reference values, and work that remains incomplete.

## How to read the quality columns

FID compares the distributions of motion features. MM-Dist measures the distance between a caption and its motion. Lower values are better for both metrics.

R-precision measures retrieval accuracy. R@1, R@2, and R@3 report whether the correct motion appears within the first one, two, or three candidates.

For a retrieval test with $N$ caption queries:

$$
R@k=\frac{\text{number of correct motions within the first }k\text{ candidates}}{N}.
$$

Higher R-precision is better. R-precision is an accuracy, not a distance.

Diversity measures differences between motions from different captions. MultiModality measures differences between motions for the same caption.

Diversity should approach the ground-truth value. A larger Diversity value does not always indicate better motion. MultiModality also needs a ground-truth reference.

For a generated Diversity value $D_{\mathrm{gen}}$ and a ground-truth value $D_{\mathrm{GT}}$, the comparison concerns:

$$
|D_{\mathrm{gen}}-D_{\mathrm{GT}}|.
$$

This difference explains the comparison. It does not replace the Diversity column.

## Repetitions and confidence intervals

The current evaluator generates motion again for each repetition. It reports a mean and a 95% confidence half-width for these metrics:

- FID.
- R@1, R@2, and R@3.
- MM-Dist.
- Diversity.

For $R$ repetitions with metric values $x_r$, the code uses:

$$
\bar{x}=\frac{1}{R}\sum_{r=1}^{R}x_r,
\qquad
sd=\sqrt{\frac{1}{R}\sum_{r=1}^{R}(x_r-\bar{x})^2},
\qquad
h_{95}=1.96\frac{sd}{\sqrt{R}}.
$$

The reported form is $\bar{x}\pm h_{95}$. The code uses a standard deviation with `ddof=0`. The half-width is not the standard deviation.

Historical “20-rep” evaluations reused one generated motion set. They repeated caption selection and batch resamples. Their intervals exclude variation between separate motion generations.

The current evaluator computes MultiModality separately, once. It uses the first captions and ten pair draws per caption. The draws permit repeated pairs and self-pairs.

The pair generator uses seed 0. MultiModality generation has no explicit seed in that method. The proposed seeds derived from 2026 remain drafted and are not applied.

## The evaluation population

The current evaluator creates motion pieces that correspond to whole-motion captions or time-tagged captions. It still removes mirror IDs.

The proposed official 4648-sample list remains drafted and is not applied. Guo's random length cut is also absent. Fixed Diversity draws and the MultiModality procedure remain unresolved.

The frozen Guo matcher supplies the evaluation feature space. Its reuse does not establish an identical evaluation protocol.

The tokenizer reconstruction count and the generator evaluation count have different meanings. A historical generator count does not define the current evaluation population.

## Parameters and state size

The parameter count measures model capacity. It does not measure the memory of a live stream.

Without a restart, a transformer retains a KV cache for its token history. Its state size increases with the token count $L$:

$$
S_{\mathrm{Transformer}}(L)=O(L).
$$

Without a restart, Mamba retains a fixed recurrent state:

$$
S_{\mathrm{Mamba}}(L)=O(1).
$$

For the Mamba backbone, the FP32 state includes an SSM tensor and a convolution tensor per layer. Its size is:

$$
S_{\mathrm{Mamba}}
=
n_{\mathrm{layers}}\,B\,d_{\mathrm{inner}}\,
\left(d_{\mathrm{state}}+d_{\mathrm{conv}}-1\right)\,4
\quad\text{bytes}.
$$

Here, $B$ is the batch size. The factor 4 is the number of bytes per FP32 value. This expression excludes model parameters and components outside the backbone.

Classifier-free guidance can require separate conditional and unconditional states. Each caption must state the batch size, precision, guidance arrangement, and state scope.

The retained service policy B restarts both twins every eight tokens. Each restart uses sixteen context tokens. This policy bounds the histories of both twins.

A transformer without a restart also has a learned-position limit. It cannot extend its history indefinitely.

## Interpretation discipline (write into every caption)

A matched parameter count, dataset, and seed do not establish a controlled comparison if the implementations differ.

The earlier twins had these defects:

- An incorrect initialization.
- Unequal dropout within the backbones.
- Fixed crop choices in persistent workers.
- A frozen EMA after a resume.

The final twins must train from zero with the reference initialization and dropout only at the shared embeddings. A resume from an old checkpoint mixes experimental conditions.

Use these rules for each table or figure:

- State the split and the actual sample count.
- State the repetition count and the interval definition.
- State the CFG scale and the length mode.
- Give the run-directory ID and the checkpoint identity.
- State the compute difference for published reference rows.
- Separate model-only measurements from service measurements.
- Mark historical scores and incomplete measurements.

Select a checkpoint on validation data. Evaluate that selected checkpoint on test data. Do not choose a different checkpoint because its test score is better.

Report negative and neutral results. The residual-FSQ collapse remains a result. An invalid extension experiment cannot establish a capacity limit or a lack of improvement.

A fixed state size does not establish stable motion over a long stream. The stream also needs a test of caption relevance over time.

## Results

### Table 1 -- Main twin table (historical test scores; current protocol: 20-rep, mean +/- 95% confidence half-width)

The historical rows below retain their recorded values. They do not represent twenty separate motion generations under the current protocol.

Lower FID and MM-Dist are better. Higher R-precision is better. Compare Diversity and MultiModality with ground truth.

| Row | FID | R@1/2/3 | MM-Dist | Diversity | MModality | params | stream state |
|---|---|---|---|---|---|---|---|
| Real (GT) | -- | 0.51/--/-- | -- | 9.5 | -- | -- | -- |
| Tokenizer ceiling (FSQ 8x1024 recon) | 0.017* | -- | -- | -- | -- | -- | -- |
| Transformer-34M (historical, test, 60 ep label) | 0.132 | 0.395/0.570/0.669 | 3.796 | 9.362 | 1.758 | 34M | O(L) without restart; bounded with B |
| **Mamba-34M** (historical, test, 60 ep label) | **0.121** | **0.418/0.594/0.693** | **3.677** | **9.437** | 1.822 | 34M | **O(1); 1,167,360 bytes, batch-one FP32 backbone** |
| Transformer-34M (historical, test, 100 ep label) | 0.163 (historical) | 0.391/0.566/0.667 | 3.823 | 9.373 | 1.718 | 34M | O(L) without restart; bounded with B |
| Mamba-34M (historical, test, 100 ep label) | 0.121 (unchanged) | same as historical 60 ep row; frozen EMA from epoch 30 | | | | 34M | O(1); 1,167,360 bytes, batch-one FP32 backbone |
| Transformer-100M (historical scale ablation, val only) | 1.63 (val) | 0.323/0.478/0.580 (val) | 4.43 | 8.79 | -- | 98M | O(L) without restart; bounded with B |
| Mamba-100M (twin) | not yet measured | | | | | ~100M | O(1) |
| *cited:* T2M-GPT | 0.116 | 0.417/--/-- | -- | -- | -- | ~230M | KV |
| *cited:* Mogo | 0.079 | -- | -- | -- | -- | -- | KV |
| *cited:* MoMask (not streamable) | 0.045 | 0.521/--/-- | -- | -- | -- | -- | bidir. |

The “60 ep” and “100 ep” labels describe historical runs. They do not establish the effective train budget of the evaluated weights.

The Mamba run resumed at epoch 30. Its evaluated EMA then remained frozen. Its “e60” model therefore contains the evaluated weights from thirty epochs.

The transformer completed sixty epochs without a resume before that point. The “e100” evaluation used the EMA from its resume point.

Both runs had a forty-epoch extension to one hundred epochs. The extension did not test newly updated EMA weights.

The historical loss curves decreased smoothly across the recorded runs. Those curves do not establish convergence of the evaluated models.

The recorded transformer test FID changed from 0.132 to 0.163. This difference cannot establish a genuine validation/test disagreement after further model updates.

The Mamba rows use the same evaluated checkpoint. Their equality does not establish that further train epochs fail to help.

The matched-budget conclusion and the data-constrained diagnosis therefore lack support. Lesson 17 and dissertation sec. 6.1 need the corrected experiments.

\*reconstruction FID (the generator's ceiling), not a generation number.

The historical GT check gave R@1 0.51 against the published 0.511. Keep this comparison as a historical check (Lesson 14.7).

The latest recorded GT check gives these comparisons:

- R@3: 0.799 against the published 0.797.
- MM-Dist: 2.964 against the published 2.974.
- Diversity: 9.505 against the published 9.503.

These checks do not resolve the protocol differences described above.

The Transformer-100M row uses validation data and 730 clips. It is a scale ablation, not a headline test comparison.

The historical generator rows do not establish coverage of today's full test population. The valid 2189-clip tokenizer reconstruction count applies to Table 2.

The pilot Mamba state has this batch-one FP32 size:

$$
15\times1\times1024\times(16+4-1)\times4
=
1{,}167{,}360\ \text{bytes}.
$$

The 2.68 MB value belongs to the 100M configuration, not the 34M pilot. Neither value describes total service memory.

### Table 2 -- Contribution A (seeded matrix; RVQ comparison requires corrected runs)

The matrix uses HumanML3D-263 test data, matched bits per step, a shared encoder and decoder, and seed 2026 (Lesson 7.2).

| Tokenizer | matched cell | recon-FID | codebook params |
|---|---|---|---|
| **Grouped-FSQ (best FSQ cell)** | 8 x 1024 | **0.0170** | **0** |
| Grouped-FSQ | 6 x 1000 | 0.0274 | 0 |
| Grouped-FSQ (iso-vocab) | 6 x 512 | 0.0305 | 0 |
| Strong-RVQ (historical best RVQ cell) | 8 x 1024 | 0.0221 | ~3M |
| Strong-RVQ (historical) | 6 x 512 | 0.0342 | ~1.6M |
| *cited:* MoMask RVQ | 6 x 512 | 0.019 (more compute) | learned |

The FSQ 8 x 1024 reconstruction result has a separate replication. It scored 2189 test clips and excluded three clips that were too short.

The replicated recon-FID is 0.016963. It agrees with the recorded 0.0170 within 1e-4. FSQ has zero codebook parameters.

The recorded FSQ value is below the cited MoMask value of 0.019. The compute difference prevents this comparison from proving controlled superiority.

The historical RVQ scores have an encoder/codebook EMA mismatch. Their code assignments also used a defective update order.

The default RVQ quantizer dropout previously caused an exception. The corrected implementation does not correct old checkpoints or old scores.

The original matrix reported an FSQ lead at all six matched cells. Corrected RVQ runs must test that claim again.

The recorded generalization check found test-minus-train gaps at or below zero for 13/15 cells (sec. 7.2). It remains a historical check.

The downstream generator comparison reported an FSQ lead on all three measured axes (sec. 7.4). Its historical “60ep” label does not establish convergence.

Corrected RVQ runs and corrected generator runs must establish the controlled downstream comparison. Variance bars across multiple seeds remain incomplete for both tokenizers.

### Table 3 -- Ablations

| Ablation | Status | Finding |
|---|---|---|
| CFG scale curve (val, pilot) | historical | FID 6.14 -> 3.55 at s=5 (-42%), R@1 x2.1 (Lesson 8.9) |
| temperature / top-p | recorded setting | locked (1.1, 0.9) |
| iso-vocab (FSQ 512 vs RVQ 512) | historical; corrected RVQ run required | recorded FSQ lead; quantizer attribution remains unverified |
| pkeep / weight-decay on/off | partial | the 06-09 run is the "without" row (Lesson B) |
| length: GT-length vs END-self-terminated | both modes available | report both (Lesson 8.8); historical evaluations used fixed length |
| tokenizer generalization gap | historical | gap = test - train <= 0 for 13/15 cells (`generalization.md`); corrected RVQ check required |
| AMASS-pretrained tokenizer | future | Lesson 18 protocol |

The availability of an END mode does not establish a completed evaluation in that mode. The audit records fixed length for all historical evaluations.

### Figure -- the streaming signature (MEASURED, 100M twins, to horizon 8192)

The figure reports state size and latency per token for random-weight 100M twins. It measures the model alone.

The plot uses `scripts/figures/plot_streaming_bench.py`. The current benchmark module is `evaluation\benchmark.py`, under `src\text2motion`.

The benchmark includes the token embeddings, backbone, and argmax operation. It excludes these components:

- The CLIP text encoder.
- The token choice procedure and classifier-free guidance.
- The tokenizer decoder.
- The transport.
- The body fit.
- The display.

The benchmark extends the transformer position capacity to cover the requested horizons. It omits the service restart policy.

The historical model-only state measurements remain valid:

- Mamba: 2.68 MB at every measured horizon.
- Transformer: 76.7 MB at 1024 tokens.
- Transformer: 303 MB at 4096 tokens.
- Transformer: 605 MB at 8192 tokens.

The historical state ratio is about 225 at 8192 tokens. It compares an unrestricted transformer history with a fixed Mamba state.

The historical transformer latency is about 12 ms per token at shorter horizons. It reaches 16.5 ms at 4096 tokens and 23.6 ms at 6144 tokens.

The historical eager Mamba latency is about 26 ms per token. The curves cross near a horizon of 6500 tokens.

Linear projections dominate the shorter transformer measurements. Attention cost increases with the history length.

The 4 GB GPU approaches its memory limit beyond about 6000 tokens. This condition amplifies the latency magnitude. The figure supports a trend, not an exact service crossover.

The kernel was unavailable for the historical Windows measurement. It later became available. The historical eager-mode penalty of about two does not describe every current execution path.

CUDA-graph coverage also differs between the twins. A graphed Mamba result and an eager transformer result do not establish a fair backbone latency comparison.

The figure is complete and needs no trained checkpoint. Its unrestricted-history crossover does not describe the deployed twins.

Policy B bounds both deployed histories. It also adds restart work that this figure does not measure.

### Long-stream caption relevance

The accepted audit records a November 3 experiment with 32 prompts, 300-second streams, and seed 2026.

Without a restart, Mamba R@3 falls from 0.72 to 0.09 at 60 seconds. Its state RMS increases by factors of three to five.

The peak state RMS increase reaches a factor of thirteen. A fixed tensor size therefore does not establish a stable state magnitude.

Policy B maintains Mamba R@3 between 0.47 and 0.69. Its state magnitude remains flat. Transformer policy B achieves R@3 between 0.63 and 0.66.

Text input at each token step, without the successful restart policy, achieves Mamba R@3 0.13. Text input alone does not resolve the measured loss of caption relevance.

Policy B remains the service policy for both twins. These measurements support that policy under the recorded experiment.

### Work that remains incomplete

Table 1 contains historical pilot scores. It does not establish a current quality lead at a matched, converged budget.

Table 2 contains a replicated FSQ reconstruction result. Its controlled RVQ comparison requires corrected RVQ runs.

The model-only figure is complete. A fair service comparison needs the same restart policy, execution path, and graph coverage.

The Mamba-100M quality row remains unmeasured. The rejected data-constrained diagnosis cannot justify its omission.

The final work also needs these items:

- Twin runs from zero with the corrected initialization and dropout policy.
- Evaluation with separate motion generations for each repetition.
- An explicit record of the actual evaluation population.
- Resolution of the drafted evaluator corrections.
- Variance bars across multiple seeds.
- Complete protocol fields and source IDs for every reported row.

The supplied historical rows omit some protocol fields and run-directory IDs. Recover those fields from the source reports before publication.
