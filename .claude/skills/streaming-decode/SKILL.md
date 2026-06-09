---
name: streaming-decode
description: >
  The real-time incremental decode loop and its bounded-memory contract — the thesis novelty. Use
  when implementing/validating chunked generation, the stream_step / recurrent-state interface, or
  wiring the decoder to the live aitviewer queue.
---

# Streaming decode (bounded-memory, real-time)

The contribution is generating motion **incrementally in bounded memory**, not regenerating the
whole sequence. Keep this invariant intact in all generator edits.

## The contract
The generator exposes a streaming interface, e.g.:
```
state = gen.init_state(text_cond)              # fixed-size recurrent/SSM state (NOT growing history)
for step in range(num_steps):
    token, state = gen.stream_step(state)      # O(1) memory in T; one token/frame chunk per call
```
- **Bounded memory:** peak memory must be ~constant in sequence length T. SSM/recurrent state is
  fixed-size; if using a transformer it needs a bounded KV-window (else it's a baseline, not the
  runtime generator — see the streaming gate in `arch-decision`).
- **Causal:** `stream_step` may depend on past + text only, never future.

## Chunked motion output
1. Accumulate generated tokens into a **window** (e.g. the tokenizer's downsample factor).
2. When a window completes, decode tokens → 168-dim features → `to_smplx_params` → push the frame
   chunk onto the viewer queue. Decode chunk-by-chunk; do not wait for the full sequence.
3. Chunk size trades latency vs smoothness — make it a config knob; report time-to-first-frame.

## Live viewer hand-off (coordinate with aitviewer-studio)
- Producer (decoder) runs in a background thread, pushing chunks onto a bounded `queue.Queue`.
- Consumer (aitviewer update callback) pops chunks and appends frames; **never block the decoder**
  on rendering. If the queue backs up, drop-to-latest rather than stall generation.

## Validation
- Assert streamed output == batch (non-streaming) output for the same text+seed (numerical parity).
- Run the `t2m-eval` streaming benchmark: peak memory ~flat in T, bounded per-chunk latency.
- Sampling here uses the shipped non-greedy sampler (temperature/top-p/CFG), matching eval.
