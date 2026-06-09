---
name: aitviewer-studio
description: >
  Load and view SMPL-X motion in aitviewer as a studio — interactive inspection, headless render to
  MP4, and the live-updating viewer fed by the streaming decoder. Use when motion needs to be seen,
  the body looks wrong, or the live demo viewer is being wired.
---

# aitviewer studio (SMPL-X)

Visualization is aitviewer, not a website. Two modes: interactive studio + headless render, plus a
live mode for the streaming demo.

## Load an SMPL-X sequence
```python
from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer

layer = SMPLLayer(model_type="smplx", gender="neutral", device=device)   # needs SMPLX_MODELS path
seq = SMPLSequence(
    poses_body=body_pose,            # (T, 63)
    smpl_layer=layer,
    poses_root=global_orient,        # (T, 3)
    trans=transl,                    # (T, 3), meters, Y-up
    betas=betas,                     # (1, 10) neutral default
    poses_left_hand=left_hand,       # (T, 45)
    poses_right_hand=right_hand,     # (T, 45)
)
v = Viewer(); v.scene.add(seq); v.run()
```
Build these tensors from the 168-dim vector via `to_smplx_params` (`motion-representation`) — never
hand-slice in render code.

## Studio scene
Add a floor at y=0, a key light, and a fixed camera framing the body; lock the up-axis to **Y**. A
correct T-pose and a known clip must render without axis flips or mesh explosions before anything else.

## Headless render to MP4
```python
from aitviewer.headless import HeadlessRenderer
r = HeadlessRenderer(); r.scene.add(seq); r.save_video(video_dir="outputs/", output_path="clip.mp4")
```
Headless needs the `[viewer]` extra and a working GL/EGL context.

## Live mode (streaming demo — coordinate with streaming-decode)
- Create an empty/short `SMPLSequence`; register a per-frame update callback on the `Viewer`.
- The callback pops a chunk from the decoder's bounded queue and **appends** frames (extends the
  poses tensors) — it must not block on generation. If the queue is empty, hold the last frame.
- This live path is the studio demo that backs the real-time claim — keep it smooth, not stalling.

## Gotchas
- SMPL-X model files are license-gated (`donor data\models_smplx_v1_1.zip`) → unzip to `SMPLX_MODELS`,
  load from config, never commit.
- If the body lies down / sinks through the floor, suspect Z-up→Y-up upstream, not the viewer.
- `use_pca=False` hands → 45-dim each, matching the representation contract.
