"""Interactive aitviewer window for multi-actor SSM output (no MP4 written)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import fields as dc_fields

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.plane import ChessboardPlane
from aitviewer.viewer import Viewer

from src.architecture.nn_models import TextToMotionSSM
from src.architecture.rvq_tokenizer import MotionRVQTokenizer
from src.data.motion_normalize import denormalize
from src.modules.motion.ssm_model import sampleIndices
from src.modules.render.smplx_render import SMPLX_DIR, smplxParams2Sequence
from src.shared.config import TrainingConfig
from src.shared.tokenizer import tokenize

# Distinct colours per actor so you can tell them apart at a glance.
ACTOR_COLORS = [
    (0.72, 0.60, 0.52, 1.0),  # warm beige (actor 0)
    (0.45, 0.65, 0.85, 1.0),  # cool blue  (actor 1)
    (0.65, 0.85, 0.50, 1.0),  # green
    (0.85, 0.55, 0.65, 1.0),  # pink
]


def parseArgs() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive multi-actor motion viewer")
    p.add_argument("prompt", nargs="?", default="two people shake hands")
    p.add_argument("--checkpoint", default="checkpoints/motion_ssm/best_model.pt")
    p.add_argument("--rvq-checkpoint", default="checkpoints/rvq_tokenizer/best_model.pt",
                   dest="rvqCheckpoint")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0, dest="topP")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def coerceConfig(raw) -> TrainingConfig:
    if isinstance(raw, TrainingConfig):
        return raw
    rawFields = {f.name: getattr(raw, f.name) for f in dc_fields(raw)}
    keep = {f.name for f in dc_fields(TrainingConfig)}
    fieldVals = {k: v for k, v in rawFields.items() if k in keep}

    # Back-compat: old dualPerson flag -> numActors=2
    if rawFields.get("dualPerson", False) and fieldVals.get("numActors", 1) == 1:
        fieldVals["numActors"] = 2
    return TrainingConfig(**fieldVals)


def loadModel(args, log):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    if not os.path.exists(args.rvqCheckpoint):
        raise FileNotFoundError(args.rvqCheckpoint)
    device = torch.device(args.device)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sdKeys = list(ck["model_state_dict"].keys())

    cfg = coerceConfig(ck["config"])

    if any(k.startswith("decoder_actors.") for k in sdKeys):
        maxIdx = max(
            int(k.split(".")[1]) for k in sdKeys if k.startswith("decoder_actors.")
        )
        cfg.numActors = max(getattr(cfg, "numActors", 1), maxIdx + 2)
    elif any(k.startswith("decoder_p2.") for k in sdKeys):
        cfg.numActors = 2
    log.info("numActors=%d  cross_actor_attention=%s",
             cfg.numActors, getattr(cfg, "crossActorAttention", False))

    model = TextToMotionSSM(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    rvqCk = torch.load(args.rvqCheckpoint, map_location=device, weights_only=False)
    tokenizer = MotionRVQTokenizer(
        motionDim=cfg.motionDim, latentDim=cfg.rvqLatentDim,
        nCodebooks=cfg.rvqNCodebooks, codebookSize=cfg.rvqCodebookSize,
        downT=cfg.rvqDownT,
    ).to(device)
    tokenizer.load_state_dict(rvqCk["model_state_dict"])
    tokenizer.eval()

    return model, tokenizer, cfg, ck["motion_stats"], ck["vocab"]


def generateMotions(args, model, tokenizer, cfg, motionStats, vocab, log):
    """Run the model once and return a list of (T, 168) numpy arrays, one per actor."""
    numFrames = int(args.duration * args.fps)

    if cfg.useSbert:
        inputs = [args.prompt]
    else:
        tokIds = torch.tensor(tokenize(args.prompt, vocab),
                              dtype=torch.long).unsqueeze(0).to(model.cond.device)
        inputs = tokIds

    with torch.no_grad():
        out, _ = model(inputs, numFrames)

        # K=1 returns single logits, K>1 returns a tuple. Normalise to list.
        logitsAll = list(out) if isinstance(out, tuple) else [out]
        motions = []

        for k, logitsK in enumerate(logitsAll):
            idxK = sampleIndices(logitsK, args.temperature, args.topP)
            motionK = tokenizer.decode(idxK).cpu().numpy()[0][:numFrames]
            motionK = denormalize(motionK, motionStats)
            log.info("actor %d motion shape=%s", k, motionK.shape)
            motions.append(motionK)

    return motions


def main() -> None:
    args = parseArgs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("view_dual_live")

    log.info("prompt=%r duration=%.1fs fps=%d device=%s",
             args.prompt, args.duration, args.fps, args.device)
    model, tokenizer, cfg, motionStats, vocab = loadModel(args, log)
    motions = generateMotions(args, model, tokenizer, cfg, motionStats, vocab, log)

    # --- Build aitviewer scene ---
    # Tell aitviewer where the SMPLX body model files live (same path the
    # headless renderer uses).
    C.update_conf({  # type: ignore[union-attr]
        "smplx_models": SMPLX_DIR,
        "window_width": 1280,
        "window_height": 720,
    })

    viewer = Viewer()
    viewer.scene.fps = args.fps
    viewer.playback_fps = args.fps
    viewer.scene.background_color = [0.92, 0.93, 0.95, 1.0]

    # Auto-play on open so the animation starts moving immediately.
    # Without this, aitviewer defaults to paused at frame 0 and you have to
    # press SPACE to start. Toggle: SPACE pauses/resumes once in the window.
    viewer.run_animations = True

    # Replace the default floor with a clearer chessboard.
    if viewer.scene.floor is not None:
        viewer.scene.remove(viewer.scene.floor)
    floor = ChessboardPlane(100.0, 200, (0.82, 0.83, 0.84, 1.0),
                            (0.80, 0.81, 0.82, 1.0), "xz")
    viewer.scene.floor = floor
    viewer.scene.add(floor)

    # Add each actor as a coloured SMPLSequence.
    for k, motionK in enumerate(motions):
        color = ACTOR_COLORS[k % len(ACTOR_COLORS)]
        seq = smplxParams2Sequence(motionK, color=color, inputCoordSystem="yup")
        seq.name = f"actor_{k}"
        viewer.scene.add(seq)

    # Tilt the camera slightly so two actors are both in frame.
    cam = viewer.scene.camera
    if cam is not None:
        cam.position = np.array([2.0, 1.8, 4.5])
        cam.target = np.array([0.0, 1.0, 0.0])

    log.info("opening viewer window (close it to exit). %d actor(s) loaded.", len(motions))
    viewer.run()


if __name__ == "__main__":
    main()
