# Metrics reference

This file explains each number that the project logs, reports or uses for a decision.
Read section 1 first. After that, each section is independent.

Sources: the code paths in each section are the definition. If this file and the code disagree, the code is correct and this file needs an update.
The derivations of the evaluator metrics are in `doc/notes/lessons/14-evaluation-metrics.md`.

## 0. Quick table

| Name in the logs | What it answers | Better is | Typical value (width 320, validation) |
|---|---|---|---|
| `ce` | How well does the model predict the next token during training (with noise)? | lower | 4.2 to 6.5 |
| `[clean-ce] train` / `val` | The same question, with no noise, on train and on validation | lower | train 3.9, val 4.4 to 4.5 |
| `gap` | How much does the model know the train data better than new data? | smaller | +0.55 to +0.65 |
| `FID` | Do the generated motions look like real motions, as a set? | lower | P1 0.27 to 0.32; step50 about 0.15 |
| `R@1`, `R@3` | Does the motion match its text? | higher | R@1 about 0.50, R@3 about 0.78 |
| `MM` (MM-Dist) | How far is a motion from its text in the evaluator space? | lower | about 3.0 |
| `Div` | Are the generated motions different from each other? | near the real value | about 9.3 |
| `ric`, `rot6d`, `foot` | Geometric errors of the decoded motion during training | lower | about 0.2 |
| `mpjpe_mm`, `recon_fid` | How well does the tokenizer rebuild a motion? | lower | see lesson 07 |
| `ms/step`, state bytes | Cost of training and of streaming | lower | see lesson 13 |

## 1. The base ideas

### 1.1 The three data splits

- **Train**: the model learns from these clips.
- **Validation (val)**: the model never learns from these clips. We use them to select the epoch and the recipe. In the training logs, the evaluation uses 628 validation clips.
- **Test**: the model never learns from these clips, and we do not use them for any choice. We score a final model on the test split one time. That score is the number for the thesis.

A number that you used for a choice is optimistic. This is why the validation number of a selected model is higher quality than its test number will be, and why the test split must stay unused until the end.

### 1.2 Tokens

The tokenizer changes a motion into a sequence of integers (tokens). One token step covers 4 frames. The generator predicts the next token from the text and the tokens before it. All training metrics measure this prediction. All evaluator metrics measure the motion after the decoder changes the tokens back into frames.

### 1.3 The evaluator feature space

FID, R-precision, MM-Dist, Diversity and MultiModality do not compare raw motions. A frozen network from Guo et al. (the "matcher") maps each motion and each caption to a vector of 512 numbers. A motion and its caption get vectors that are near each other.

All published HumanML3D results use this same matcher. So our numbers have the same definition as the published numbers. We do not train or change the matcher.

One consequence: these metrics measure what the matcher can see. A defect that the matcher does not see (for example, a small foot slide) does not change them. Section 6 has direct metrics for such defects.

## 2. Training metrics (file: `src/text2motion/generation/generator_trainer.py`)

### 2.1 Cross-entropy (`ce`)

For each position $t$, the model gives a probability $p_\theta(x_t \mid x_{<t}, c)$ to the correct next token $x_t$, where $c$ is the text. The cross-entropy is the mean negative log of that probability:

$$
\mathrm{CE} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}, c).
$$

How to read a value:
- A model that guesses uniformly over $V$ tokens has $\mathrm{CE} = \ln V$. For 1024 codes, $\ln 1024 = 6.93$.
- $e^{\mathrm{CE}}$ is the perplexity: the number of tokens between which the model is, in effect, undecided. $\mathrm{CE} = 4.5$ gives $e^{4.5} \approx 90$ tokens.
- A CE of 0 means that the model knows the next token with certainty. For motion this is not possible and not wanted, because many motions are correct for one text.

The `ce` in the epoch line is measured during training, **with** the training noise: some input tokens are replaced by random tokens (`pkeep`), and the text is sometimes removed (`cfg_dropout`). So this value is higher than the clean value, and it changes when the noise recipe changes. Do not compare `ce` between 2 recipes.

### 2.2 Clean cross-entropy (`[clean-ce] train`, `val`)

The same formula, measured with no noise, on 50 fixed train batches and 50 fixed validation batches, with the averaged (EMA) weights. Because the noise is off, this value **is** comparable between recipes.

- `val` falls: the model still learns something that applies to new data.
- `val` stays flat or goes up while `train` falls: the model now learns the train clips by heart.

### 2.3 The CE gap (`gap`)

$$
\mathrm{gap} = \mathrm{CE}_{\text{val}}^{\text{clean}} - \mathrm{CE}_{\text{train}}^{\text{clean}}.
$$

The gap measures memorization. A gap of 0 means that the model predicts new clips as well as train clips. A larger gap means more memorization.

Compare the gaps of 2 models at the **same validation CE**, not at the same epoch. A model that learned less has a small gap for a trivial reason.

### 2.4 The stop rule (patience)

The run stops when the clean validation CE did not reach a new lowest value for 6 epochs in sequence (`--val_ce_patience 6`). An equal value counts as no improvement. A new lowest value sets the counter back to 0.

Known limit: the rule has no minimum size for an improvement. A drop of 0.0001 sets the counter back. With the step50 recipe, the validation CE falls by such small amounts until the last epoch, so the rule does not stop the run.

### 2.5 The geometric terms (`ric`, `rot6d`, `foot`, `total`)

The trainer also decodes the predicted tokens to a motion and compares it with the true motion (file: `src/text2motion/generation/losses.py`).

- `ric`: error of the joint positions that are relative to the root.
- `rot6d`: error of the joint rotations in the 6D form.
- `foot`: error of the foot velocity on the frames where the true foot touches the floor.
- `total`: the CE plus the weighted sum of these terms. The optimizer minimizes `total`.

These values help to see a broken run. We do not use them to select a model.

### 2.6 `lr` and `min`

`lr` is the learning rate at the end of the epoch. It follows a cosine curve from 2e-4 to almost 0 at the last planned epoch. `min` is the time of the epoch in minutes.

## 3. Evaluator metrics (file: `src/text2motion/evaluation/metrics.py`)

Notation: $m_i$ is the feature vector of motion $i$, and $s_i$ is the feature vector of its caption.

### 3.1 FID (Fréchet distance)

FID compares 2 **sets**: the real motions and the generated motions. It fits a Gaussian to each set of feature vectors, with means $\mu_r, \mu_g$ and covariances $\Sigma_r, \Sigma_g$, and measures the distance between the 2 Gaussians:

$$
\mathrm{FID} = \lVert \mu_r - \mu_g \rVert_2^2 + \mathrm{Tr}\!\left(\Sigma_r + \Sigma_g - 2\,(\Sigma_r \Sigma_g)^{1/2}\right).
$$

- The first term: is the average generated motion equal to the average real motion?
- The second term: do the generated motions have the same spread and the same correlations as the real motions?

How to read a value:
- 0 means that the 2 sets have the same Gaussian statistics. Real motion against real motion gives about 0.002 on the test split.
- FID is not a score for one motion. It cannot tell you which clip is bad.
- FID does not use the text. A model that ignores the text but makes natural motion can have a good FID.
- **FID depends on the number of clips.** Fewer clips give a higher FID for the same model. Compare 2 FID values only at the same clip count. Our validation FID (628 clips) and a published test FID (about 4,000 clips) are not comparable.
- FID has noise. In epochs 50 to 60 of the step50 transformer, the validation CE was flat, but the FID moved between 0.149 and 0.198. A difference smaller than about 0.05 on 628 clips is not evidence.

### 3.2 R-precision (`R@1`, `R@2`, `R@3`)

R-precision is a retrieval test. It asks: can the matcher find the correct motion for a caption?

1. Take a group of 32 generated motions (`R_PRECISION_POOL = 32`).
2. For one caption $s_i$, compute the Euclidean distance to each of the 32 motions: its own motion and 31 others.
3. Sort the distances. The rank of the own motion is its position in that order.
4. $R@k$ is the share of captions for which the own motion is in the first $k$ positions.

$$
R@k = \frac{1}{N}\sum_{i=1}^{N} \mathbf{1}\!\left[\operatorname{rank}_i \le k\right].
$$

How to read a value:
- A random model gives $R@1 = 1/32 = 0.031$ and $R@3 = 3/32 = 0.094$.
- Real motions with their real captions give about $R@1 = 0.51$ and $R@3 = 0.80$ on the test split. This is the practical ceiling. A model can be a little above it, because generated motions can be more "typical" than real ones.
- R-precision measures the link between text and motion. It does not measure if the motion is natural.
- Noise: on 628 clips, R@1 moved between 0.446 and 0.515 in neighbour epochs. A difference below about 0.03 is not evidence.

### 3.3 MM-Dist (`MM` in the logs)

The mean Euclidean distance between each generated motion and its own caption:

$$
\mathrm{MMDist} = \frac{1}{N}\sum_{i=1}^{N} \lVert s_i - m_i \rVert_2.
$$

Lower is better. Real pairs give about 2.97. MM-Dist measures the same property as R-precision (text match), but as a distance and not as a rank. The 2 metrics usually agree.

Warning: `MM` in our logs is MM-Dist. It is not MultiModality (section 3.5).

### 3.4 Diversity (`Div`)

Draw 300 random pairs of generated motions (`DIVERSITY_PAIRS = 300`) and take the mean distance:

$$
\mathrm{Div} = \frac{1}{300}\sum_{q=1}^{300} \lVert m_{a_q} - m_{b_q} \rVert_2.
$$

Diversity is not "higher is better". The target is the value of real motion (about 9.5). A value far below it means that the model makes the same few motions (collapse). A value far above it usually means noise.

### 3.5 MultiModality

For one caption, generate several motions and measure the mean distance between pairs of them. It measures the variation for the **same** text. The function is in `src/text2motion/evaluation/evaluator.py`. The training logs do not show it. As for Diversity, a moderate value is the target, and the value alone does not prove that each variation is valid.

### 3.6 How the metrics work together

| FID | R@1 | Meaning |
|---|---|---|
| low | high | Natural motion that follows the text. This is the goal. |
| low | low | Natural motion that ignores the text. |
| high | high | The motion follows the text, but it does not look real. |
| high | low | The model failed. |

The guidance scale (`cfg_scale`) moves a trained model between these cases: a higher scale gives more text match and, above some point, a worse FID.

## 4. The evaluation protocol

- **Sampling settings**: guidance scale 6.0, temperature 1.0 in the training evaluations. A number is valid only with its settings.
- **Repetitions**: generation is random. A final number is the mean of 20 repetitions, with an interval. The evaluations inside a training run use 1 repetition, so they have more noise.
- **Dropped clips**: the evaluator reports clips that it did not score (for example, `out_of_length`). The clip count belongs to the result.
- **The default split of the CLI is test.** Each validation call must say `--split val`.

## 5. Statistics: how sure are we?

### 5.1 The 95% interval over repetitions

For $n$ repetitions with mean $\bar{x}$ and standard deviation $\sigma$, the code reports

$$
\bar{x} \pm 1.96\,\frac{\sigma}{\sqrt{n}}.
$$

This is the "± value" in the published tables. It covers the randomness of the generation. It does not cover the choice of the clips or the training seed.

### 5.2 The bootstrap interval over clips (`bootstrap_fid`)

Draw the same number of clips again, with replacement, many times, and compute the metric each time. The range that holds 95% of these values is the interval. It answers: how much would the number change with a different sample of clips?

### 5.3 The paired comparison (`paired_fid_delta`)

To compare 2 models, score both on the **same** resampled clips and keep the difference each time. If the 95% interval of the difference does not contain 0, the models are different on this split (`separates = True`). A paired test sees a smaller difference than 2 independent intervals, because the clip effect cancels.

### 5.4 What no interval covers

We train with one seed (2026). No statistic here measures how much a result changes with another training seed. The thesis must say this in its limits.

## 6. Direct motion-quality metrics (file: `src/text2motion/evaluation/metrics.py`)

These metrics use the 22 joint positions in metres, at 20 frames per second. They do not use the matcher.

| Metric | Definition | Meaning |
|---|---|---|
| Foot skate (m/s) | Mean horizontal speed of a foot joint on the frames where it is near the floor | A foot on the floor must not move. Higher is worse. |
| Root teleports | Number of frames where the root moves more than 0.25 m in one frame | 0.25 m per frame is 5 m/s. A count above 0 is a break in the motion. |
| Frames over the acceleration limit (%) | Share of frames where a joint acceleration is above 0.08 m per frame² | Sudden jumps. |
| Mean jerk | Mean size of the third difference of the joint positions | Jerk is the change of acceleration. High jerk looks like a shake. Too low looks like a slow blur. |
| Peak speed, peak acceleration | 99th percentile of the joint speed and acceleration, relative to the root | Sharpness of the motion. Compare with the real value. |
| Transition statistics | Jerk in a window around the frame where 2 segments join | Quality of the switch between 2 prompts in a long stream. |

## 7. Tokenizer metrics (file: `src/text2motion/tokenization/reconstruction_metrics.py`)

The tokenizer encodes a real motion to tokens and decodes it back. These metrics compare the result with the input.

- **`mpjpe_mm`** (mean per-joint position error): the mean distance in millimetres between each rebuilt joint and the true joint. It is the most direct error. 0 is perfect.
- **`feature_l2`**: the mean distance between the rebuilt and the true 263-number frame vectors.
- **`recon_fid`**: the FID between the rebuilt motions and the real motions. It is the floor for the generator: a generator on this tokenizer cannot have a better FID than this, except by chance.
- **Code usage and perplexity** (in the tokenizer training logs): how many codes of the codebook the data uses, and how evenly. Unused codes are wasted capacity.

## 8. Cost and streaming metrics (folder: `scripts/benchmarks/`)

- **`ms/step`**: the time of one training step (one batch) in milliseconds. It depends on the precision mode. Our runs use TF32.
- **Peak GPU memory (GB)**: the largest memory that the training step allocates. The GPU has 4 GB.
- **Time per token (ms)**: the time to generate one token in a stream. One token is 4 frames, which is 200 ms of motion at 20 fps. Generation is real-time if one token takes less than 200 ms.
- **State bytes**: the memory that the generator keeps between 2 tokens. For Mamba this size is constant. For the transformer it grows with the context, up to the window limit. This is the core of the streaming claim.

## 9. Common errors of interpretation

1. Comparing the noisy `ce` of 2 recipes. Use the clean CE.
2. Comparing FID values at different clip counts, or validation against test.
3. Reading a difference of 0.02 in FID or R@1 on 628 clips as a result. It is inside the noise.
4. Reporting the best epoch of a run as "the" result. The best of many noisy values is optimistic. The test score of the selected model is the honest number.
5. Reading `MM` in the logs as MultiModality. It is MM-Dist.
6. Reading a higher Diversity as better. The target is the real value.
7. Judging the text match from FID. FID does not use the text.

## 10. Reference values

| Source | Split | FID | R@1 | R@3 | MM-Dist | Diversity |
|---|---|---|---|---|---|---|
| Real motion (Guo et al.) | test | 0.002 | 0.511 | 0.797 | 2.974 | 9.503 |
| T2M-GPT (arXiv:2301.06052, Table 1) | test | 0.116 | 0.491 | 0.775 | n/a | n/a |
| P1 transformer, epoch 30 | val, 628 clips | 0.272 | 0.510 | n/a | n/a | n/a |
| P1 Mamba, epoch 24 | val, 628 clips | 0.316 | 0.502 | n/a | n/a | n/a |
| step50 transformer, epoch 60 | val, 628 clips | 0.149 | 0.498 | n/a | n/a | n/a |
| step50 Mamba, epoch 24 (run not ended on 2026-11-09) | val, 628 clips | 0.153 | 0.507 | n/a | n/a | n/a |

The validation rows and the test rows are not comparable (section 3.1). The real-motion row is from the published table and was not measured again for this file.
