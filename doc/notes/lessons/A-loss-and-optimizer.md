# Applied Lesson A -- The loss function and the optimizer (what we actually run)

> This lesson follows the representation theory in lessons 1-3. It explains how the trainer updates the generator.
> Sources: `src\text2motion\generation\losses.py`, `src\text2motion\generation\trainer.py`, and `src\text2motion\generation\contracts.py` (`GeneratorTrainingConfig` and `LossWeights`).

## A.1 Two families of loss

The generator predicts discrete tokens. The total loss combines two loss families:

1. Token cross-entropy (CE) measures the error in each codebook choice.
2. Geometric loss measures the error in the motion that the decoder produces.

For a 1000-code codebook, CE asks which of the 1000 codes is correct. The current grouped FSQ configuration uses 1024 codes per group.

For a target code $y$, CE uses the probability of that code:

$$
\mathcal{L}_{\text{CE}}=-\log p_y.
$$

CE does not measure the motion distance between codes. It does not distinguish wrong codes through their decoded motion.

The geometric loss supplies this distance through soft decoding. The decoder receives expected code vectors instead of discrete code indices.

The soft decode follows these steps:

1. Remove the final appended-END time position when the targets include END.
2. Exclude the END vocabulary entry from each codebook.
3. Apply softmax to the logits for the real codes.
4. Calculate the expected vector for each codebook.
5. Combine these vectors with the tokenizer.
6. Pass the combined vectors through the frozen decoder.

For codebook $q$, let $e_{q,k}$ be the vector for real code $k$. The expected vector is:

$$
\bar e_q=\sum_{k=0}^{K-1}p_{q,k}e_{q,k}.
$$

Here, $K$ counts real codes only. END has no motion vector.

The decoder produces `(B, 4M, 263)` from `(B, M, Q, V)` motion logits with the current downsample factor of four. The frame count differs from the token count.

The tokenizer parameters stay fixed. The decoder still passes gradients from the geometric loss to the generator logits.

The reconstruction loss covers all 263 channels through separate channel groups:

- The root channels.
- The relative joint positions (`ric`).
- The joint rotations (`rot6d`).
- The joint velocities (`vel`).
- The foot contacts (`foot`).

Thus, the geometric loss penalizes position, rotation, and velocity errors in the same objective. The root and foot groups have separate weights.

## A.2 The exact recipe

Each $\mathcal{L}_g$ is an L1 loss on one channel group of the soft-decoded motion. $\mathcal{L}_{\text{CE}}$ has a fixed weight of one.

The fixed-weight objective starts with:

$$
\mathcal{L} = \mathcal{L}_{\text{CE}}
+ w_{\text{ric}}\mathcal{L}_{\text{ric}}
+ w_{\text{rot6d}}\mathcal{L}_{\text{rot6d}}
+ w_{\text{vel}}\mathcal{L}_{\text{vel}}
+ w_{\text{foot}}\mathcal{L}_{\text{foot}}
+ w_{\text{root}}\mathcal{L}_{\text{root}}
\;\big(+\, w_{\text{fk}}\,\mathcal{L}_{\text{fk}}\big)
$$

The default weights are $w_{\text{ric}}=0.5,\ w_{\text{rot6d}}=0.5,\ w_{\text{vel}}=0.3,\ w_{\text{foot}}=0.1,\
w_{\text{root}}=0.3$.

The notation $\mathcal{L}_{\text{fk}}$ groups two optional FK losses. Both are off by default ($w_{\text{fk}}=0$). The A/B ablation enables them.

For a channel group with $C_g$ channels and a valid-frame mask $m_{bt}$, the geometric loss is:

$$
\mathcal{L}_g=
\frac{\sum_{b,t,c}m_{bt}\left|\hat x_{btc}-x_{btc}\right|}
{C_g\max\left(1,\sum_{b,t}m_{bt}\right)}.
$$

The mask excludes padded frames. CE also excludes padded token positions when token lengths are available. CE includes the valid END target.

CE measures the discrete choice. The geometric terms measure motion error. The trainer records each term separately.

The complete fixed-weight recipe also supports two optional dynamics terms:

$$
\mathcal{L}_{\text{full}}=
\mathcal{L}
+w_{\text{foot\_skate}}\mathcal{L}_{\text{foot\_skate}}
+w_{\text{accel}}\mathcal{L}_{\text{accel}}.
$$

Both dynamics weights default to zero. The configuration `configs\generator\gen_pilot_fsq8x1024_dynamics.yaml` sets `w_foot_skate: 10.0` and `w_accel: 18.0`.

The foot-skating loss measures horizontal foot displacement where the ground-truth contact channels indicate contact. This differs from the L1 loss on foot-contact channels.

The acceleration loss compares second differences of recovered joint positions:

$$
a_t=p_{t+2}-2p_{t+1}+p_t.
$$

It measures the coordinate-wise L1 difference between predicted and ground-truth acceleration. Its mask requires all three frames to be valid.

The FK and dynamics terms first restore the feature units with the dataset mean and standard deviation. These terms require both statistics.

The trainer also supports optional uncertainty weighting. Its default remains fixed weights. With uncertainty weighting, the geometric weights become learned parameters:

$$
\mathcal{L}_{\text{uncertainty}}
=\mathcal{L}_{\text{CE}}
+\sum_g\left(e^{-s_g}\mathcal{L}_g+s_g\right).
$$

Each $s_g$ is a learned log-variance parameter for an active geometric term. CE retains its fixed weight of one.

## A.3 Which cost function is better

- Tokens use cross-entropy because each target selects a codebook entry.
- **Geometry -> L1, not L2.** L2 punishes large errors quadratically -> the model hedges toward the
  *average* of plausible motions -> **blurry, over-smoothed** output. **L1 is outlier-robust and
  yields crisp, decisive motion.** Standard choice for motion regression; used throughout.

For a scalar error $e$, the two costs are:

$$
L_1(e)=|e|,\qquad L_2(e)=e^2.
$$

The implementation uses L1 for the geometric terms. The FK terms have a separate scale issue, which section A.4b explains.

## A.4 Which optimizer

The trainer uses AdamW. AdamW adjusts the update scale for each parameter and applies weight decay separately from the loss gradient.

The default parameter rates are:

- A generator learning rate of 2e-4.
- A learning rate of 1e-5 for the unfrozen CLIP parameters.
- A weight decay of 0.01 for parameters in the decay groups.

The generator contains embeddings, a sequence backbone, and output heads. These parameter groups can have different gradient scales.

AdamW separates weight decay from the adaptive gradient update. Plain Adam with an L2 penalty does not make this separation.

The default parameter partition excludes these parameters from weight decay:

- The embeddings.
- The normalization parameters.
- The parameters with at most one dimension.
- The Mamba parameters whose names end in `a_log` or `d_skip`.

The trainer uses a warmup schedule followed by a cosine schedule. The default warmup lasts 1000 optimizer steps.

The schedule increases the learning rate from zero to the group rate. It then reduces the rate toward 0.01 times that rate.

The schedule horizon uses the configured total optimizer steps. The default request specifies 60 epochs. This setting does not prove convergence.

The trainer also uses an exponential moving average (EMA) with a decay of 0.999:

$$
\bar\theta_u=0.999\bar\theta_{u-1}+0.001\theta_u.
$$

The EMA averages the generator parameters after each optimizer update. Evaluation uses the average parameters instead of the live parameters.

The trainer clips the gradient norm at 1.0 before each optimizer update:

$$
g_{\text{clip}}=
g\min\left(1,\frac{1.0}{\lVert g\rVert_2}\right).
$$

This limit controls large finite gradients. The trainer raises an error for a non-finite gradient norm.

### Why not SGD?

SGD can require careful rates for each layer and a suitable momentum value to match Adam on Transformers or SSMs.

AdamW supplies adaptive rates for these models. SGD remains competitive mainly for CNNs with substantial parameter tuning.

## A.4b FK-consistency loss (proposed, data-validated 2026-06-14)

The owner proposed a forward-kinematics consistency loss. The HumanML3D representation stores both `rot6d` and `ric` positions.

Forward kinematics (FK) calculates joint positions from the rotations. The loss compares these positions with a second position estimate.

The implementation supports two terms:

1. Self-consistency: `FK(rot6d) vs model ric`.
2. FK-to-GT: `FK(rot6d) vs GT positions`.

Self-consistency checks agreement between the predicted rotations and predicted positions. FK-to-GT anchors the rotations to ground-truth positions and addresses error accumulation along the joint chains.

The FK comparison checks the complete pose. A bone-length loss checks only segment lengths. The proposed recipe therefore drops the bone-length loss.

The differentiable functions `recover_from_rot` and `recover_from_ric` already provide the required position estimates.

The implementation has an important normalization detail. With a length mask, the FK denominator counts valid frames and three coordinates, but omits the 22 joints:

$$
\mathcal{L}_{\text{fk,masked}}=
\frac{\sum_{b,t,j,c}m_{bt}
\left|\hat p_{btjc}-p_{btjc}\right|}
{3\max\left(1,\sum_{b,t}m_{bt}\right)}.
$$

This value is 22 times the joint-coordinate mean. Without a length mask, the ordinary L1 mean includes the joint dimension.

Both FK weights default to zero. An ablation weight must account for this scale difference before a comparison with the other terms.

## A.5 What to hold onto

1. CE measures the discrete token choice.
2. The geometric L1 loss passes gradients through the frozen decoder.
3. The geometric groups cover all 263 channels, so positions and rotations receive penalties together.
4. **L1 over L2** for geometry -> no blur.
5. AdamW, warmup/cosine, EMA, and gradient clipping form the adopted optimizer stack.
6. FK, foot-skating, and acceleration losses are optional terms.

## A.6 Results and limits

### FK data check

The read-only check on 2026-06-14 compared `recover_from_ric` with `recover_from_rot` on 30 random ground-truth clips.

The recorded measurements were:

- A mean per-joint L2 distance of 0.84 mm.
- A median distance of 0.00 mm.
- A mean distance of 0.05% of body height.

The root positions matched exactly. The largest differences occurred at the wrists and hands, where errors accumulate along the FK chains.

This check found a near-zero discrepancy between the two ground-truth representations. It supported the proposed loss as a viable consistency check.

The decision was to test both FK terms in a local A/B comparison before the final run. Data agreement does not establish a generator quality gain.

### Historical EMA evaluations

The historical Mamba e60 evaluation used the frozen epoch-30 EMA. The Transformer e100 evaluation used the EMA from its resume point.

These evaluations did not measure matched convergence. They also do not support the claim that an extended run changed nothing.

The same EMA fault invalidates the historical conclusions that the optional loss terms gave no gain. The code and weights establish available choices, not their measured benefit.
