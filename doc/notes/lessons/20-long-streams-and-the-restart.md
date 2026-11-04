# Lesson 20 -- Long streams and the restart

> Lesson 12 explains the S6 recurrence. Lesson 13 explains the state size and the cost of a stream.
> This lesson explains how the generator keeps the prompt active after the duration of a training clip.
> A fixed-size state limits memory use. It does not guarantee that the motion follows the prompt for minutes or hours.

## 20.1 The problem: a stream exceeds the training length

The generator predicts one motion token at each step. Each token represents four frames. The HumanML3D frame rate is 20 frames per second.

The files in `configs\generator\*.yaml` set `max_motion_len: 196`. Thus, a training clip contains at most 49 motion tokens.

$$
L_{\mathrm{train}}=\frac{196}{4}=49,
\qquad
T_{\mathrm{train}}=\frac{196}{20}=9.8\ \mathrm{s}.
$$

Here, $L_{\mathrm{train}}$ counts motion tokens. It excludes the text prefix and the END target.

Let $L$ denote the number of motion tokens that the generator produces in a rollout. The ratio compares the rollout length with the training length.

$$
r=\frac{L}{L_{\mathrm{train}}}=\frac{L}{49}.
$$

A stream with 1500 motion tokens lasts 300 seconds. Its rollout length exceeds the maximum training length by more than a factor of 30.

$$
T=\frac{4L}{20}=0.2L\ \mathrm{s},
\qquad
L=1500
\quad\Rightarrow\quad
T=300\ \mathrm{s},
\qquad
r=\frac{1500}{49}\approx30.6.
$$

The model learns from short clips. A long rollout asks the model to use its own predictions far beyond those clips.

Two separate questions follow:

- Does the generator keep the text signal?
- Does the generator state stay within a range that the model can use?

The Mamba state has a fixed size, as Lesson 12 explains. Its values can still change as the rollout becomes longer.

The transformer has a different constraint. Its default configuration permits 96 positions after a one-position text prefix.

`GeneratorConfig` sets `max_seq_len: 96` and `text_prefix_len: 1` in `src\text2motion\generation\contracts.py:88–90`.

The transformer creates 97 position embeddings, including the prefix position, in `src\text2motion\generation\nn_backbones.py:321`.

The code does not automatically remove old keys and values. Without a restart, the stream must obey the position limit.

`check_stream_budget` checks this limit in `src\text2motion\generation\nn_model.py:188–223`. A longer text prefix reduces the available motion budget.

Policy B makes the transformer cache bounded through repeated reconstruction. Section 20.3 explains this policy.

## 20.2 Two failure hypotheses

### 20.2a The recurrence and the text signal

Lesson 12 derives the S6 recurrence. The following equation uses its diagonal state update.

$$
h_t=\bar A_t\odot h_{t-1}+\bar B_t\odot x_t.
$$

The state $h_t$ contains information from earlier inputs. The input $x_t$ also determines the selective coefficients.

The retention term carries part of the old state. The insertion term adds information from the current input.

Define the accumulated retention between positions $j$ and $t$ as follows.

$$
P_{j,t}=\prod_{k=j+1}^{t}\bar A_k,
\qquad
P_{t,t}=1.
$$

Then the recurrence expands into a retained initial state and a sum of retained inputs.

$$
h_t=P_{0,t}\odot h_0+
\sum_{j=1}^{t}P_{j,t}\odot\bar B_j\odot x_j.
$$

This equation describes one selective recurrence. The complete generator also contains convolutions, nonlinear functions, residual connections, and several layers.

### 20.2b H1: the state leaves the training range

Hypothesis H1 states that a long rollout produces states outside the range that the model encounters during training.

Excessive retention can preserve many earlier insertions. Their sum can produce a large state, even when each retention coefficient remains below one.

A norm bound makes this distinction explicit.

$$
\|h_t\|
\leq
\|\bar A_t\|\,\|h_{t-1}\|
+
\|\bar B_t\odot x_t\|.
$$

A contractive retention term does not impose the state range of a short training clip. The insertion term also affects that range.

The paper *Stuffed Mamba: Oversized States Lead to the Inability to Forget* studies excessive retention in language models.

Its evidence motivates H1. It does not establish the cause of a failure in this motion generator.

Source: [arXiv:2410.07145v3](https://arxiv.org/html/2410.07145v3), §§3.2.2, 4.2, 5.1–5.2.

A state range requires a reference from actual training inputs. An early generated window does not establish that reference.

### 20.2c H2: the text signal decays

Hypothesis H2 states that the text signal becomes too weak to affect later predictions.

With prefix-only text, the generator receives the text before the motion tokens. Later state updates can reduce its contribution.

For a prefix contribution $h_{\mathrm{text}}$, its retained component has the following form.

$$
h_{\mathrm{text},t}
=
P_{\mathrm{text},t}\odot h_{\mathrm{text}}.
$$

If the relevant retention products become small, this component also becomes small. Later motion inputs can further reduce the effect of the prompt.

This expression explains a possible mechanism. It does not isolate text information in the complete nonlinear generator.

### 20.2d What per-step text changes

Per-step text adds the projected text vector to each motion-token embedding.

$$
u_t=e(z_t)+c,
$$

where $e(z_t)$ is the motion-token embedding and $c$ is the projected text vector.

The code implements this addition in `src\text2motion\generation\nn_model.py:164–175`. The option applies during both normal prediction and context replay.

Per-step text supplies a new text signal at each step. It therefore addresses the missing-input mechanism in H2.

It does not impose an upper limit on the recurrent state. It also does not remove old state information.

Thus, per-step text alone does not resolve H1. It can affect the recurrence, but that effect requires a separate check.

## 20.3 Policy B: periodic reconstruction

### 20.3a The policy name and its parameters

The policy name is **periodic sliding-window recomputation with text re-prefixing**.

Policy B uses these parameters:

- A rebuild stride of 8 generated motion tokens.
- A replay history of at most 16 generated motion tokens.
- The current text prefix at each rebuild.

Policy A carries the generator state without these rebuilds.

The implementation accepts the parameters through `anchor_every` and `anchor_context`. Policy B selects `anchor_every=8` and `anchor_context=16`.

These values identify the policy. They are not the default values of `stream_token_indices`.

### 20.3b The exact sequence

`stream_token_indices` starts at `src\text2motion\generation\nn_model.py:226`.

Its `start_from` function reconstructs the state at lines 258–276. The main loop stores recent tokens and schedules rebuilds at lines 278–309.

The generator performs this sequence:

1. Create the initial generator state from the text prefix.
2. Predict and emit motion tokens.
3. Keep at most the latest 16 motion tokens.
4. After each group of 8 tokens, create a new generator state.
5. Process the current text prefix in the new state.
6. Replay the retained motion tokens in their original order.
7. Continue prediction from the reconstructed state.

The first rebuild follows token 8 and replays eight tokens. The next rebuild follows token 16 and replays sixteen tokens.

Later rebuilds replay the latest sixteen tokens. Between rebuilds, the generator carries the state normally.

Thus, sixteen is the replay history size. It is not a strict limit on the history available to every prediction.

In steady operation, predictions depend on 16 to 23 motion tokens since the most recent rebuild.

$$
K=16,
\qquad
S=8,
\qquad
K\leq H_t\leq K+S-1=23.
$$

Here, $H_t$ counts the motion tokens that contribute through the reconstructed state. The text prefix remains an additional input.

### 20.3c What each backbone reconstructs

Mamba creates zero SSM states and zero convolution states. It then processes the prefix and the retained motion tokens.

`MambaMixer.init_state` creates both zero states in `src\text2motion\generation\nn_backbones.py:117–120`.

The transformer performs a fresh prefill. It creates new keys and values from the prefix and the retained motion tokens.

`TransformerBackbone.prefill` implements this operation in `src\text2motion\generation\nn_backbones.py:336–343`. Its position indices start at zero for each prefill.

Both backbones discard the old generator state. Neither backbone changes motion tokens that the generator already emitted.

Both CFG branches undergo the rebuild. The conditional branch receives the current text, and the unconditional branch receives the null text input.

The code forms these branches in `src\text2motion\generation\nn_model.py:244–247`. Each branch replays the same generated motion tokens.

The decoder remains active throughout the rebuild. Its history and pending tokens belong to a separate iterator.

`StreamingMotionDecoder.stream_tokens` keeps these buffers in `src\text2motion\generation\decoder.py:121–132`.

The generator restart therefore does not reset the decoder or regenerate earlier visible motion. Lesson 15 explains the decoder contract.

### 20.3d The cost

Let $p$ denote the text prefix length. Each rebuild processes at most $p+K$ inputs.

For the one-position prefix, this gives one replay of at most 17 inputs every 8 generated tokens.

$$
p=1,\qquad K=16,\qquad S=8,
\qquad
p+K=17.
$$

The amortized replay work is bounded.

$$
\frac{p+K}{S}
=
\frac{17}{8}
\quad\text{inputs per generated token}.
$$

This quantity describes replay work, not total runtime. Normal prediction, CFG, and decoder work also contribute to the cost.

The code supports longer text prefixes. Their replay cost is $p+16$, not necessarily 17.

For fixed $p$, $K$, and $S$, the replay overhead does not grow with the rollout length.

$$
C_{\mathrm{replay}}(L)
=
O\!\left(\frac{L}{S}(p+K)\right)
=
O(L).
$$

The transformer also keeps a bounded cache under this policy. Its attention cost depends on the bounded local sequence, not the full emitted history.

This operating mode differs from the transformer with an expanding cache in Lesson 13.

### 20.3e The published precedent

The closest explicit precedent is the sliding-window recomputation baseline in *Efficient Streaming Language Models with Attention Sinks*.

Source: [arXiv:2309.17453](https://arxiv.org/html/2309.17453), §1 and Figure 1(c).

That baseline reconstructs representations from a recent input window. Our adaptation adds an eight-token stride and text re-prefixing.

Policy B is not the attention-sink method of StreamingLLM. That method retains initial sink keys and values plus recent keys and values.

Ordinary sliding-window attention retains cached representations. Policy B reconstructs them.

*Mistral 7B* explains sliding-window attention and cache management.

Source: [arXiv:2310.06825](https://arxiv.org/html/2310.06825), §2 and Figures 1–3.

A semantic text prefix is not evidence of an attention sink. The citation establishes a reconstruction precedent, not a motion-quality guarantee.

## 20.4 How published systems keep text and history

A history buffer contains motion information that conditions future generation. It differs from an archive of all emitted frames.

The following table uses the verified paper descriptions. It does not assume a hidden-cache implementation where the paper leaves that detail unspecified.

| System | Motion history | Text access | Information that persists |
|---|---|---|---|
| DartControl | Last 2 frames; each primitive predicts 8 future frames. | The current text conditions every primitive through its denoiser calls. | Canonicalized motion history persists. Each primitive starts from fresh diffusion noise. |
| MotionStreamer | The immediately preceding prompt round, plus the generated latents of the current round. Older rounds are removed. | A text prefix enters once per round and remains accessible through causal attention. | The next round receives new text plus the preceding round's motion latents. Exact KV-cache reuse is unspecified. |
| ARDY | A configurable history window of up to 8 seconds. | A text token enters each denoiser call. | Earlier motion tokens enter the new plan, with recentered root coordinates. Persistent KV-cache reuse is unspecified. |
| CLoSD | Last 20 executed simulator frames; the planner predicts 40 frames. | Text cross-attention operates in every transformer layer at every denoiser step. | The simulator state persists. Each new plan uses executed history and fresh noise. |
| TimelineControl | Up to 8 clean history latents, with up to 78 prediction latents. | Interval-masked text attention operates throughout the denoiser calls. | Committed motion stays fixed. New instructions replace only uncommitted predictions. KV-cache reuse is unspecified. |
| FlowMDM | The whole offline composition. Its relative attention horizon is 100/150 frames for BABEL/HumanML3D. | Pose-specific text conditions the queries throughout the denoiser calls. | The complete noisy motion persists across diffusion iterations. This is not an online history policy. |
| ActionPlan | Completed latents remain within the modeled N-position sequence. An open-ended history budget is unspecified. | Sequence-level text conditions every model evaluation. An action plan also guides motion. | The clean action plan stays fixed during progressive motion generation. Hidden-cache reuse is unspecified. |

Sources:

- DartControl: [arXiv:2410.05260v3](https://arxiv.org/html/2410.05260v3), §§3.1–3.2 and Algorithm 1.
- MotionStreamer: [arXiv:2503.15451](https://arxiv.org/html/2503.15451), §3.3.
- ARDY: [arXiv:2607.08741](https://arxiv.org/html/2607.08741), §§3.3–3.5 and 4.1–4.2.
- CLoSD: [arXiv:2410.03441](https://arxiv.org/html/2410.03441), §§4.1, 4.3, and 5.1.
- TimelineControl: [arXiv:2610.03873](https://arxiv.org/html/2610.03873), §§3, 4.2–4.3, 5.1, and Appendix B.4.
- FlowMDM: [arXiv:2402.15509](https://arxiv.org/html/2402.15509), §§3.1–3.3 and 4.1.
- ActionPlan: [arXiv:2603.13500](https://arxiv.org/html/2603.13500), §§3.2–3.4 and Appendix D.

The online systems keep the text available during generation and reuse recent motion context. They do not demonstrate an unlimited persistent motion history.

MotionStreamer retains a variable previous-round history. ActionPlan does not specify an open-ended eviction budget, and FlowMDM operates offline.

Thus, “text at every step” means continued text access. It does not mean that every system inserts a fresh text prefix at every step.

Policy B follows the recent-history pattern. It does not inherit the transition quality or guarantees of these published systems.

## 20.5 Alternatives without a hard restart

The following methods change exposure to old states, retention, insertion, or feedback from generated inputs.

**None of these alternatives is tested in this project.**

### 20.5a Longer continuous training

Longer continuous sequences expose the model to older states. The model can then learn state updates beyond the current clip length.

This approach requires genuinely longer motion sequences and suitable captions. A larger crop limit alone does not create those data.

Full sequence backpropagation also requires more activation memory.

*Stuffed Mamba* studies a training-length threshold that depends on the state size. Its language-model threshold does not define a motion threshold.

Source: [arXiv:2410.07145v3](https://arxiv.org/html/2410.07145v3), §§4.2 and 5.1–5.2.

### 20.5b Stateful training with truncated backpropagation

Truncated backpropagation through time, or TBPTT, divides a sequence into chunks. The next chunk receives the final state of the previous chunk.

The gradient stops at each chunk boundary.

$$
h^{(q+1)}_0
=
\operatorname{detach}\!\left(h^{(q)}_{\mathrm{final}}\right).
$$

The model encounters old states without a gradient graph that covers the entire history. Mamba must carry the convolution state as well.

A valid continuous experiment requires ordered chunks from the same sequence. State carry across unrelated shuffled clips creates a different input distribution.

*Stuffed Mamba*, Appendix F, describes TBPTT as a method to vary initial states.

Source: [arXiv:2410.07145v3](https://arxiv.org/html/2410.07145v3), Appendix F.

### 20.5c Reduced retention and insertion

Reduced retention and insertion applies separate factors to the old state and the new input.

$$
h_t
=
\rho\,\bar A_t\odot h_{t-1}
+
\eta\,\bar B_t\odot x_t.
$$

The factor $\rho$ reduces retention. The factor $\eta$ reduces insertion.

*Stuffed Mamba* applies this intervention without retraining in its published experiment. It can reduce performance on short contexts.

The motion generator requires its own parameter check. The sequential and parallel paths must use the same modified recurrence.

Source: [arXiv:2410.07145v3](https://arxiv.org/html/2410.07145v3), §3.2.2.

### 20.5d Sliding-window recurrence

A sliding-window recurrence removes the decayed contribution that precedes a fixed window. It uses delayed state information instead of a periodic hard restart.

The paper describes this method for the weighted sum within a recurrence.

This exact layer operation does not establish equivalence to a fresh replay of the complete nonlinear generator. The method also requires bounded auxiliary history.

Source: [arXiv:2410.07145v3](https://arxiv.org/html/2410.07145v3), §3.2.2 and Equation 8.

### 20.5e Griffin variance control

Griffin uses a real-gated linear recurrent unit, or RG-LRU. Its input scale depends on the retention coefficient.

The relevant recurrence has this form.

$$
h_t
=
a_t\odot h_{t-1}
+
\sqrt{1-a_t^2}\odot i_t.
$$

The insertion scale controls variance as retention changes. This mechanism differs from arbitrary clipping of the Mamba state RMS.

An adaptation changes the learned recurrence and requires retraining. Griffin's language-model results do not establish motion performance.

Source: [arXiv:2402.19427](https://arxiv.org/html/2402.19427), §2.4, Equation 4, and §6.1.

### 20.5f LongMamba

LongMamba identifies global channels and skips both decay and insertion for selected low-importance inputs. It uses calibrated thresholds that depend on length.

Its purpose is to preserve information over longer contexts. That purpose can conflict with H1, which concerns excessive retention.

An unbounded target horizon also complicates length-dependent calibration. The published intervention does not establish indefinite motion quality.

Source: [arXiv:2504.16053](https://arxiv.org/html/2504.16053), §§5.1–5.2 and 6.2.

### 20.5g DeciMamba

DeciMamba removes selected inputs to enlarge the effective receptive field. The published method principally changes prefill.

Its recurrent decode path remains unchanged. Our generator has a short text prefix, not a long document that requires prefill reduction.

Removal of generated motion tokens would also affect temporal alignment. Thus, DeciMamba does not directly address this rollout problem.

Source: [arXiv:2406.14528](https://arxiv.org/html/2406.14528), §§4–5.

### 20.5h Auto-conditioned training

Auto-conditioned recurrent networks alternate blocks of recorded inputs and self-generated inputs. This exposes the model to errors from its own predictions.

The method addresses accumulated feedback errors. It does not specifically isolate excessive state retention or text decay.

The acLSTM paper studies extended human motion. It also reports physical defects and gives no general stability proof.

Source: [arXiv:1707.05363v5](https://arxiv.org/html/1707.05363v5), §§3–5.

## 20.6 Progress and “walk 20 steps”

### 20.6a Duration differs from event count

A duration budget tells the controller when to stop or change the prompt. It does not tell the controller how many footfalls occurred.

For a requested duration $D$, the nominal token budget is:

$$
N=\frac{D}{0.2}=5D.
$$

This equation assumes a duration that corresponds to a whole number of tokens. The controller must define a policy for other durations.

For “walk 20 steps,” this lesson defines a step as one completed left-foot or right-foot touchdown.

Let $K$ denote the requested count and $k_t$ denote the observed count.

$$
K=20,
\qquad
k_t=\sum_{\tau\leq t}e_\tau,
$$

where $e_\tau$ denotes a detected touchdown event.

A time budget does not establish $k_t=K$. Gait phase also differs from the completed event count.

### 20.6b Published control mechanisms

| Mechanism | Published example | What it controls |
|---|---|---|
| Duration or token length | MoMask specifies the number of motion-token positions. | The sequence length, not the number of completed actions. |
| Normalized progress | Action2Motion supplies the counter $t/T$. | Progress toward an externally supplied total length. |
| END token | T2M-GPT appends a learned End symbol. | A learned clip boundary, not a verified repetition count. |
| Timeline intervals | STMC, FlowMDM, and TimelineControl use specified action intervals or segment durations. | External action boundaries and text applicability. |
| Repetition count | RepNet predicts period and periodicity from video. | An observed repetition count, not generator completion. |

Sources:

- Action2Motion: [arXiv:2007.15240](https://arxiv.org/abs/2007.15240), §3.2.2 and Figure 2.
- MoMask: [arXiv:2312.00063](https://arxiv.org/abs/2312.00063), §3.4.
- T2M-GPT: [arXiv:2301.06052](https://arxiv.org/abs/2301.06052), §3.2.
- STMC: [arXiv:2401.08559](https://arxiv.org/abs/2401.08559), §§3.1–3.3.
- FlowMDM: [arXiv:2402.15509](https://arxiv.org/abs/2402.15509), §§3 and 3.2–3.3.
- TimelineControl: [arXiv:2610.03873](https://arxiv.org/abs/2610.03873), §§3 and 4.2.
- RepNet: [arXiv:2006.15418](https://arxiv.org/abs/2006.15418), §§3.3–3.5.

RepNet estimates repetition count through period and periodicity values.

$$
\hat K=\sum_i\frac{p_i}{l_i}.
$$

Here, $p_i$ is the periodicity estimate and $l_i$ is the period estimate. RepNet is a video observer, not a motion generator.

Neither RepNet nor an END token establishes exact touchdown control in this project.

### 20.6c Current code behavior

The training code constructs END targets at the valid clip boundary.

`append_end_targets` implements this operation in `src\text2motion\generation\trainer.py:234–248`.

The service disables END-based termination through `stop_at_end=False` in `src\text2motion\app\service.py:415`.

The stream masks the END symbol when termination is disabled. This behavior appears in `src\text2motion\generation\nn_model.py:286–293`.

Thus, END targets exist during training, but serving does not use them to stop a stream.

The stream also accepts a `text_at` callback. It requires an active restart policy.

The callback supplies new text only at a rebuild, at `src\text2motion\generation\nn_model.py:305–308`.

A prompt change therefore takes effect at a rebuild. It does not immediately affect a prediction between rebuilds.

### 20.6d Keep progress outside the generator state

The controller must store action progress outside the recurrent state and the KV cache. Otherwise, a rebuild can erase progress information.

Let $j$ denote the global token index and $j_0$ denote the action start index.

$$
j_{\mathrm{action}}=j-j_0,
\qquad
q_j=\frac{j_{\mathrm{action}}}{N}.
$$

A rebuild changes the generator state. It must not change $j$, $j_0$, $N$, or the observed event count.

The current loop already separates its global `index` from `since_anchor`. Their updates appear in `src\text2motion\generation\nn_model.py:278–309`.

A future progress-conditioned model must also use the original timestamps during replay. Replay must not increment the external counter.

This is a proposed extension. The current code does not establish a learned progress signal or an exact step counter.

### 20.6e The cheapest demonstration

The cheapest demonstration is a duration-labelled prompt queue with policy B. It uses the existing tokenizer and generator checkpoints.

The controller can perform this sequence:

1. Read each prompt and its duration.
2. Convert the duration to a token budget.
3. Keep the action counter outside the generator state.
4. Supply the next prompt at a rebuild.
5. Preserve the decoder state across prompt changes.

This demonstration controls scheduled duration. It does not establish that “walk 20 steps” produces exactly twenty steps.

An experimental touchdown counter requires a defined event rule and audited labels. Existing foot-contact extraction does not itself count completed steps.

The decoder also buffers future tokens before it emits frames. A detected twentieth step can therefore occur after the generator predicts additional motion.

Exact-count control requires a separate boundary policy and an independent count-error measure.

## 20.7 Results and limits

### 20.7a The experiment

The experiment date is 2026-11-03. It uses seed 2026 and 32 prompts, with 300 seconds per prompt.

The sampling parameters are:

- CFG scale: 6.
- Top-p: 0.9.
- Temperature: 1.0.

The table gives means over the 32 prompts. Each sequence lists the windows at 0, 60, 120, and 300 seconds.

Lesson 14 explains R@3. Higher R@3 indicates better text retrieval for these windows.

Slide measures foot motion during contact. Jerk measures rapid changes in acceleration.

The SSM state RMS summarizes state magnitude. It does not measure semantic information or the full state distribution.

| Arm | R@3 | slide m/s | jerk m/s^3 | SSM state RMS mean (max) |
|---|---|---|---|---|
| Mamba prefix A (no restart) | 0.72/0.09/0.13/0.13 | 0.20/0.56/0.55/0.25 | 37/110/97/47 | 0.032 (0.12) / 0.168 (1.58) / 0.112 (0.91) / 0.077 (1.00) |
| Mamba prefix B | 0.72/0.47/0.56/0.56 | 0.22/0.32/0.28/0.35 | 41/34/39/41 | 0.030 (0.11) / 0.035 (0.12) / 0.035 (0.10) / 0.035 (0.13) |
| Mamba per-step A | 0.75/0.13/0.13/0.19 | 0.15/0.61/0.62/0.62 | 37/124/132/93 | 0.033 (0.18) / 0.107 (0.62) / 0.100 (0.52) / 0.073 (0.27) |
| Mamba per-step B | 0.78/0.63/0.63/0.69 | 0.19/0.20/0.19/0.18 | 45/38/37/34 | 0.030 (0.18) / 0.034 (0.15) / 0.034 (0.13) / 0.034 (0.13) |
| Transformer prefix B | 0.66/0.63/0.66/0.63 | 0.22/0.22/0.24/0.23 | 51/47/50/48 | n/a |

The Mamba A arms lose retrieval performance in later windows. Per-step text alone does not prevent this loss.

The Mamba B arms keep state RMS close to their initial generated-window values. Their R@3 is higher than A in every later window.

Most physical measures also improve relative to A. One exception exists: at 300 s, the prefix A slide (0.25 m/s) is lower than the prefix B slide (0.35 m/s).

This association is consistent with H1. It does not establish H1 as the cause.

### 20.7b Memory, decoder parity, and runtime

Peak GPU memory is 0.82–0.84 GiB for every arm.

The decoder stream-versus-full parity error is 2.4e-6 at full precision. This result concerns decoder output, not equivalence between generator policies A and B.

The runtime measurements use a batch of four prompts with CFG.

| Generator mode | Time per batch token |
|---|---|
| Mamba without restart | 27 ms |
| Mamba with restart | 78 ms |
| Transformer with restart | 16.5 ms |

The motion rate is five tokens per second. Thus, real-time generation allows 200 ms per token.

$$
\Delta t_{\mathrm{token}}
=
\frac{1}{5}\ \mathrm{s}
=
200\ \mathrm{ms}.
$$

Each measured generator time is below this budget. These times do not establish the complete viewer latency.

The transformer with policy B is faster than Mamba with policy B in this experiment. A bounded cache changes the long-stream comparison from Lesson 13.

### 20.7c The recorded decision

The recorded decision is `RETAIN_B` for both twins.

The decision rule compares each A arm with its B reference at 60, 120, and 300 seconds.

It uses these tolerance thresholds:

- An R@3 drop of 0.05.
- A slide ratio of 1.1.
- A jerk ratio of 1.1.
- A height-drift excess of 0.05 m.

For the physical ratios, A is compared with B. For retrieval, the rule applies two checks: the A drop relative to its own first window, and the A drop relative to B.

An A arm fails when any check fails in any later window. Both Mamba A arms failed the retrieval checks at 60, 120, and 300 seconds.

A PASS for B means “valid reference” under this decision procedure. It is not an absolute motion-quality claim.

### 20.7d Limits of the comparison

The per-step checkpoint comes from a different training run than the prefix checkpoint. The checkpoints are not a matched pair.

The 32 prompts give R@3 noise of about ±0.08. No paired test compares per-step text with prefix-only text.

Thus, this experiment does not establish that per-step text is superior. The stronger comparisons use A and B within the same checkpoint.

The experiment covers 300 seconds per prompt. It does not verify hours of prompt-following motion.

State RMS is only a magnitude summary. The initial generated window does not establish the state range from training.

The recorded summaries also do not establish the first token at which the state exceeds a training reference.

### 20.7e The causal test is not run

Policy B changes both state history and text access. The A-versus-B comparison therefore does not isolate H1 from H2.

A causal test is designed but **not run**. It uses three interventions:

- Sham refresh: insert a null text input without a reset.
- Clamp plus sham: restrict state magnitude and apply the same sham input.
- Text refresh without reset: insert the original text and retain the old state.

The test requires an empirical RMS envelope from actual training inputs. It also requires paired continuations and tokenwise state records.

A clamp tests excessive magnitude, not every form of state-distribution mismatch. A text insertion can itself change the dynamics.

Both interventions can help, or neither can help. Such results can remain mixed or inconclusive.

The current evidence supports policy B as the reference for long streams. It does not prove a causal state-range explanation or exact action completion.
