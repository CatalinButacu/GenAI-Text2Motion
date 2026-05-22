"""End-to-end streaming-avatar demo: instruction in -> motion frames out.

This is the dissertation's headline demo script. It wires up every M5
component built today into a single CLI that you can drive interactively:

    python scripts/demo/run_streaming_demo.py \
        --planner-ckpt checkpoints/planner_lm \
        --motion-ckpt  checkpoints/motion_ssm/baseline/best_model.pt \
        --rvq-ckpt     checkpoints/rvq_tokenizer/best_model.pt \
        --instruction "walk forward until you reach the tree, then turn left, then wave"

Pipeline:
  1. ActionPlanner (fine-tuned GPT-2-small) decomposes the instruction
     into a list of (action_text, until_predicate) tuples.
  2. StreamingRunner yields raw 168-d SMPL-X frames as the SSM streams
     them. The Mamba hidden state carries over between actions for
     smooth transitions.
  3. Frames are appended to a numpy array and saved to disk; a JSON log
     records the per-action timing and termination conditions so the
     dissertation Chapter 8 figure has clean source data.

NOTE: the script gracefully fails if any required checkpoint is missing
-- the user gets a clear "train X first" message rather than a stack
trace. All three checkpoints must exist; there is intentionally no
"fake planner" fallback (decision: no hardcoded action lists anywhere).

Output structure:
    runs/streaming_demo/{timestamp}/
        instruction.txt
        plan.json           -- planner output
        frames.npy          -- (N, 168) raw motion frames
        events.jsonl        -- per-frame timing + condition state
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Default named scene objects with placeholder positions. Real scenes
# come from the M2 ScenePlanner via PlannedScene.entities; the demo lets
# the user override these via --scene-object.
DEFAULT_SCENE = {
    "tree": np.array([5.0, 0.0, 0.0]),
    "ball": np.array([3.0, 0.0, 2.0]),
    "chair": np.array([0.0, 0.0, 4.0]),
    "wall": np.array([0.0, 0.0, 8.0]),
}


def _parse_scene_overrides(overrides: list[str] | None) -> dict[str, np.ndarray]:
    """Convert --scene-object NAME=x,y,z entries into a position dict."""
    scene = {k: v.copy() for k, v in DEFAULT_SCENE.items()}

    for override in overrides or []:
        name, _, xyz = override.partition("=")

        if not xyz:
            raise ValueError(
                f"--scene-object expects NAME=x,y,z; got {override!r}"
            )
        parts = [float(p) for p in xyz.split(",")]

        if len(parts) != 3:
            raise ValueError(
                f"--scene-object position needs 3 floats; got {xyz!r}"
            )
        scene[name.strip()] = np.array(parts)

    return scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-ckpt", default="checkpoints/planner_lm",
                        help="Directory with the fine-tuned GPT-2-small planner LM")
    parser.add_argument("--motion-ckpt", default="checkpoints/motion_ssm/baseline/best_model.pt",
                        help="MotionSSM .pt checkpoint")
    parser.add_argument("--rvq-ckpt", default="checkpoints/rvq_tokenizer/best_model.pt",
                        help="RVQ tokenizer .pt checkpoint (causal_decoder=True recommended)")
    parser.add_argument("--instruction", required=True,
                        help="Natural-language instruction to decompose + render")
    parser.add_argument(
        "--scene-object", action="append", default=[],
        help=("Override default scene-object position. NAME=x,y,z. "
              "Repeat for multiple. Default scene: " + ", ".join(DEFAULT_SCENE.keys())),
    )
    parser.add_argument(
        "--max-action-latents", type=int, default=50,
        help="Defensive cap on latent steps per action (planner may emit "
             "a condition that never fires; we abort the action and advance).",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Where to write frames + plan + events. Default: "
             "runs/streaming_demo/<timestamp>/",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # ---------- preflight ---------- #
    missing: list[str] = []

    for label, p in [
        ("planner LM", args.planner_ckpt),
        ("MotionSSM", args.motion_ckpt),
        ("RVQ tokenizer", args.rvq_ckpt),
    ]:
        if not Path(p).exists():
            missing.append(f"  {label}: {p}")

    if missing:
        print("Missing checkpoints; cannot run the demo:", file=sys.stderr)

        for m in missing:
            print(m, file=sys.stderr)
        print(file=sys.stderr)
        print("Train them first:", file=sys.stderr)
        print("  python scripts/data/synthesize_planner_data.py", file=sys.stderr)
        print("  python scripts/training/train_planner_lm.py", file=sys.stderr)
        print("  python scripts/training/train_rvq_tokenizer.py --causal-decoder", file=sys.stderr)
        print("  python scripts/training/train_motion_ssm.py --pose-prefix-prob 0.5",
              file=sys.stderr)

        return 1
    # ---------- imports (deferred so missing-checkpoint message is fast) ---------- #
    from src.modules.agent.planner import ActionPlanner
    from src.modules.agent.runner import StreamingRunner
    from src.modules.agent.world_state import WorldState
    from src.modules.motion.ssm_model import SSMMotionModel
    # ---------- build agent ---------- #
    log.info("Loading planner LM from %s", args.planner_ckpt)
    planner = ActionPlanner(args.planner_ckpt)
    log.info("Loading MotionSSM from %s", args.motion_ckpt)
    motion = SSMMotionModel(
        checkpoint_path=args.motion_ckpt,
        rvq_checkpoint_path=args.rvq_ckpt,
    )
    # The runner needs the raw motion model + tokenizer, not the SSMMotionModel
    # facade (which is for offline batch inference). Pull them out:
    motion_model = motion.model
    tokenizer = motion.tokenizer
    world = WorldState()

    for name, pos in _parse_scene_overrides(args.scene_object).items():
        world.add_scene_object(name, pos)
    log.info("Scene: %s", sorted(world.scene_objects.keys()))
    runner = StreamingRunner(
        planner=planner.__call__,  # structurally Callable[str, list[PlannedAction]]
        motion_model=motion_model,
        tokenizer=tokenizer,
        world=world,
        max_action_latents=args.max_action_latents,
    )
    # ---------- run ---------- #
    out_dir = Path(
        args.output_dir
        or f"runs/streaming_demo/{time.strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "instruction.txt").write_text(args.instruction, encoding="utf-8")
    log.info("Output dir: %s", out_dir)
    log.info("Instruction: %r", args.instruction)
    plan = planner(args.instruction)
    (out_dir / "plan.json").write_text(
        json.dumps(
            [{"action": a.action_text, "until": a.until.source} for a in plan],
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("Plan: %d actions", len(plan))

    for i, action in enumerate(plan, 1):
        log.info("  %d. %r (until %s)", i, action.action_text, action.until.source)
    frames: list[np.ndarray] = []
    events: list[dict] = []
    t_start = time.perf_counter()
    t_first_frame: float | None = None

    for frame_idx, frame in enumerate(runner.run(args.instruction)):
        t_now = time.perf_counter() - t_start

        if t_first_frame is None:
            t_first_frame = t_now
            log.info("Time-to-first-frame: %.3f s", t_first_frame)
        frames.append(frame)
        events.append({
            "frame_idx": frame_idx,
            "t_seconds": t_now,
            "position": frame[3:6].tolist(),
            "heading_yaw": float(frame[1]),
        })
    t_total = time.perf_counter() - t_start
    log.info(
        "Done: %d frames in %.2f s (%.1f fps gen)",
        len(frames), t_total, len(frames) / max(t_total, 1e-9),
    )

    if frames:
        np.save(out_dir / "frames.npy", np.stack(frames))

        with (out_dir / "events.jsonl").open("w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
    summary = {
        "instruction": args.instruction,
        "actions_planned": len(plan),
        "frames_emitted": len(frames),
        "ttff_seconds": t_first_frame,
        "total_seconds": t_total,
        "generation_fps": len(frames) / max(t_total, 1e-9),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
