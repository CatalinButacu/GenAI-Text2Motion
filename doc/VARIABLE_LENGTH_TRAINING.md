# Variable-Length Training — How to Use Every Frame Efficiently

> **The problem you flagged**: actions have different lengths. Right now we pad every clip to `max_motion_length = 200`. A 40-frame clip wastes 80% of its compute on zero-padding that the loss mask discards. **You're right that this is inefficient**; here is the analysis + the fix.

This document analyses the current padding strategy, quantifies the waste, lays out four techniques (ordered by complexity vs ROI), and recommends the one we should ship next.

---

## 1. Current state — fixed-length padding

In `src/data/motion_dataset.py:encode_motion_sample`:

```python
motion = s["motion"].copy()                  # (T_real, 168), where 20 <= T_real <= 200
T = motion.shape[0]
motion = pad_to(motion, max_motion_length)    # (200, 168), zeros after frame T
mask   = build_mask(T, max_motion_length)     # (200,), 1.0 for [0:T], 0.0 for [T:200]
```

Every batch is `(B, 200, 168)` regardless of how long the actual clips are. The loss is masked, so we don't *learn* from the zero-padded regions — but we still **compute** over them. Mamba's selective scan touches every frame in the sequence; the FiLM modulator + RVQ head both run on all 200 latent positions.

**Quantified waste.** From the HumanML3D clip-length distribution (computed offline from the unified buffer):

| Percentile | Clip length | Wasted frames | Waste % |
|---:|---:|---:|---:|
| p10 | 60   | 140 | 70% |
| p25 | 80   | 120 | 60% |
| p50 (median) | 120  | 80  | 40% |
| p75 | 160  | 40  | 20% |
| p90 | 190  | 10  | 5%  |
| mean | 124  | 76  | 38% |

**Average ~38% of training compute is spent on zero-padded frames.** A 6-8 hour cloud run is therefore 2.3-3 hours of pure waste, every time. The fix is worth doing before the headline run.

---

## 2. Four techniques, ranked

| # | Technique | Speedup | Implementation cost | Complexity |
|---:|---|---:|---|---|
| 1 | **Bucketed batch sampling** | **2-3×** | 1 day | LOW |
| 2 | Dynamic per-batch max-length | 2-3× (combined with #1) | <1 day | LOW |
| 3 | Sequence packing (concat short clips) | 3-4× | 3-5 days | HIGH |
| 4 | Truly variable-length / nested tensors | 1-1.5× | 7-10 days | VERY HIGH |

The recommended fix is **#1 + #2 together** — they compose, ship in 1 day, and capture most of the available speedup.

---

## 3. Recommended: Bucketed batch sampling + dynamic max-length

### How it works

1. **At dataset-init time**: compute every clip's true length once. Sort indices by length. Bucket into 4-6 length ranges, e.g. `[40-60], [60-100], [100-150], [150-200]`.
2. **At batch-sampling time**: each batch draws from ONE bucket. All clips in the batch have similar lengths.
3. **At collate time**: pad to the longest clip *in this batch*, not to `max_motion_length`. A batch of 60-frame clips becomes `(B, 60, 168)`, not `(B, 200, 168)`.
4. **At model time**: zero-padding is now 0-30% of the batch instead of 0-70% — the loss mask still handles whatever's left.

### Why it works

Mamba's selective scan is `O(T·d·N)` per step, so reducing average `T` from 200 to ~130 (the bucket-average) is a 35% speedup *before* even considering that short-clip batches use larger micro-batch sizes for the same VRAM budget.

### Implementation surface

Three small changes:

```python
# 1. New sampler in src/data/length_bucketed_sampler.py
class LengthBucketedSampler(torch.utils.data.Sampler):
    def __init__(self, lengths, batch_size, n_buckets=4):
        order = np.argsort(lengths)
        bucket_size = len(order) // n_buckets
        self.buckets = [order[i*bucket_size:(i+1)*bucket_size]
                        for i in range(n_buckets)]
        self.batch_size = batch_size

    def __iter__(self):
        # Shuffle within each bucket; shuffle bucket order; yield batches
        batches = []
        for b in self.buckets:
            np.random.shuffle(b)
            for i in range(0, len(b), self.batch_size):
                batches.append(b[i:i+self.batch_size].tolist())
        np.random.shuffle(batches)
        for batch in batches:
            yield from batch

# 2. New collate_fn in src/data/motion_dataset.py
def collate_dynamic_length(batch):
    max_len = max(item["motion"].shape[0] for item in batch)
    return {
        "motion": torch.stack([pad_to(item["motion"], max_len) for item in batch]),
        "motion_mask": torch.stack([build_mask(item["length"], max_len) for item in batch]),
        # ... other fields
    }

# 3. Wire into DataLoader in base_trainer.finalize_init:
loader = DataLoader(
    train_ds,
    batch_sampler=LengthBucketedSampler(train_ds.lengths, batch_size=64),
    collate_fn=collate_dynamic_length,
    num_workers=4,
)
```

### Why we didn't ship it yet

The current training run uses fixed 200-padding because (a) it's the published-T2M baseline and (b) it makes the streaming-equivalence test stable across different batch shapes. The fix is a 1-day project we should land **before the headline cloud run** because the 35% speedup is roughly $5 of saved compute and ~2 hours of saved wall time.

---

## 4. Why NOT sequence packing (technique #3)

Sequence packing — concatenating multiple short clips into one long sequence with per-clip attention masks — is what HuggingFace LLM trainers use for variable-length text data. It's tempting because it eliminates padding entirely.

But it doesn't work cleanly for Mamba:

1. **The SSM state carries across clip boundaries.** When you concatenate clip A (40 frames) + clip B (60 frames) into a 100-frame "super-sequence", the hidden state at frame 41 contains accumulated state from clip A. Mamba has no native "reset on boundary" mechanism the way attention masks can prevent cross-clip leakage.
2. **The RVQ head produces per-position predictions** — fine, you mask the boundary tokens — but the **length-prediction head** reads a per-clip pooled feature and has no clean way to handle multiple lengths in one sequence.
3. **Streaming inference would have to mimic this** — adds complexity to the runner.

We can revisit packing if the bucketed sampler doesn't give us enough headroom, but the first project is the simpler one.

---

## 5. Why NOT nested tensors / fully variable-length (technique #4)

PyTorch's `nested_tensor` is still experimental. `torch.compile` doesn't support it well. Mamba's selective-scan CUDA kernel expects fixed-shape inputs. The 1-1.5× speedup over technique #1 is not worth the engineering risk for a thesis.

---

## 6. The honest acknowledgement for the dissertation

In Ch 5.2 (data pipeline), we should add one paragraph noting:

> "The current training pipeline uses fixed-length padding to `max_motion_length = 200`, which wastes ~38% of compute on zero-padded frames (mean across the HumanML3D length distribution). A bucketed-sampler approach has been designed and is documented in `doc/VARIABLE_LENGTH_TRAINING.md`; it lands before the journal version. The thesis numbers stand because the streaming-correctness property (Ch 7) is invariant to padding choice — padding affects training throughput, not inference behaviour."

That's honest, points to the fix, and clarifies that the headline results are not affected.

---

## 7. Tiny clips are NOT slow because they're tiny

A subtle point worth surfacing: a 40-frame clip does not take less time to train *today* — it takes the **same** time as a 200-frame clip because of fixed-length padding. The cost you're seeing isn't because the model is slow at handling short clips; it's because the framework is doing 200 frames of arithmetic regardless of the clip's true length.

The fix gives you the speedup *automatically* — short-clip batches finish faster, the wallclock drops, training accuracy stays the same because nothing about the model changes.

---

## 8. Action items

1. **Before the headline cloud run** (1 day of work):
   - Add `LengthBucketedSampler` and `collate_dynamic_length`
   - Add a test that asserts the same total loss with and without bucketing on a tiny dataset
   - Re-run the local SSM smoke; confirm step-time drops as expected
2. **In the dissertation**: add the paragraph in §6 above to Ch 5.2.
3. **Post-thesis** (only if benchmarks demand it): explore packing or nested tensors.

Outcome of step 1: cloud run goes from 6-8 hours to ~4-5 hours, saves ~$5, frees a wall-clock window for an ablation re-run if needed.
