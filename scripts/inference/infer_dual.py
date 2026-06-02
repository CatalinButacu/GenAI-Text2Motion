"""Dual-person inference; renders each agent to <name>_p1.mp4 / <name>_p2.mp4."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import fields as dc_fields

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.architecture.nn_models import TextToMotionSSM
from src.architecture.rvq_tokenizer import MotionRVQTokenizer
from src.data.motion_normalize import denormalize
from src.modules.motion.ssm_model import sampleIndices
from src.modules.render.smplx_render import renderSmplx2Video
from src.shared.config import TrainingConfig
from src.shared.tokenizer import tokenize


def parseArgs() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual-person text-to-motion inference")
    p.add_argument("prompt", nargs="?", default="two people shake hands")
    p.add_argument("--checkpoint", default="checkpoints/motion_ssm/best_model.pt",
                   help="Dual-person model checkpoint (must contain decoder_p2.*)")
    p.add_argument("--rvq-checkpoint", default="checkpoints/rvq_tokenizer/best_model.pt",
                   dest="rvqCheckpoint")
    p.add_argument("--name", default="dual_out")
    p.add_argument("--output-dir", default="outputs/videos", dest="outputDir")
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0, dest="topP")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def coerceConfig(raw) -> TrainingConfig:
    if isinstance(raw, TrainingConfig):
        return raw
    fieldVals = {f.name: getattr(raw, f.name) for f in dc_fields(raw)}
    return TrainingConfig(**fieldVals)


def main() -> None:
    args = parseArgs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("infer_dual")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint!r}")

    if not os.path.exists(args.rvqCheckpoint):
        raise FileNotFoundError(f"rvq checkpoint not found: {args.rvqCheckpoint!r}")

    device = torch.device(args.device)
    log.info("loading model: %s", args.checkpoint)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Verify the checkpoint has a multi-actor head, in either layout:
    #   - legacy V1 dual-person: `decoder_p2.*`
    #   - V3 / new numActors: `decoder_actors.0.*` (and possibly .1.*, .2.* for K>2)
    sdKeys = list(ck["model_state_dict"].keys())
    isLegacyV1 = any(k.startswith("decoder_p2.") for k in sdKeys)
    isV3 = any(k.startswith("decoder_actors.") for k in sdKeys)

    if not (isLegacyV1 or isV3):
        raise RuntimeError(
            "Checkpoint has no multi-actor head (no decoder_p2.* nor "
            "decoder_actors.*). Use main.py for single-person models."
        )

    cfg = coerceConfig(ck["config"])

    # Resolve numActors from the saved config OR from the key layout if the
    # config field wasn't present (very early dual-person checkpoints).
    if isV3:
        # Infer K from the highest decoder_actors index in the state_dict
        maxIdx = max(
            int(k.split(".")[1]) for k in sdKeys if k.startswith("decoder_actors.")
        )
        cfg.numActors = max(getattr(cfg, "numActors", 1), maxIdx + 2)
    else:
        # Legacy: dual person had decoder_p2 only -> K = 2
        cfg.numActors = 2
    log.info("inferred numActors=%d (legacy_v1=%s, v3=%s)",
             cfg.numActors, isLegacyV1, isV3)
    vocab: dict = ck["vocab"]
    motionStats = ck["motion_stats"]
    log.info("motion_stats loaded: mean.shape=%s std.shape=%s",
             motionStats.mean.shape, motionStats.std.shape)

    model = TextToMotionSSM(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    log.info("loading rvq tokenizer: %s", args.rvqCheckpoint)
    rvqCk = torch.load(args.rvqCheckpoint, map_location=device, weights_only=False)
    tokenizer = MotionRVQTokenizer(
        motionDim=cfg.motionDim,
        latentDim=cfg.rvqLatentDim,
        nCodebooks=cfg.rvqNCodebooks,
        codebookSize=cfg.rvqCodebookSize,
        downT=cfg.rvqDownT,
    ).to(device)
    tokenizer.load_state_dict(rvqCk["model_state_dict"])
    tokenizer.eval()

    numFrames = int(args.duration * args.fps)
    log.info("prompt='%s'  numFrames=%d", args.prompt, numFrames)

    if cfg.useSbert:
        inputs = [args.prompt]
    else:
        tokIds = torch.tensor(tokenize(args.prompt, vocab),
                              dtype=torch.long).unsqueeze(0).to(device)
        inputs = tokIds

    with torch.no_grad():
        out, _ = model(inputs, numFrames)

        if not isinstance(out, tuple) or len(out) < 2:
            raise RuntimeError(
                f"Expected multi-actor logits tuple of len>=2, got {type(out)}."
            )
        logitsAll = list(out)

        motions = []

        for k, logitsK in enumerate(logitsAll):
            idxK = sampleIndices(logitsK, args.temperature, args.topP)
            motionK = tokenizer.decode(idxK).cpu().numpy()[0]  # (T, 168)
            motionK = motionK[:numFrames]
            # Reverse the z-norm applied during training so axis-angle
            # channels land back in their natural range for the renderer.
            motionK = denormalize(motionK, motionStats)
            motions.append(motionK)
            log.info("decoded actor %d shape=%s", k, motionK.shape)

    os.makedirs(args.outputDir, exist_ok=True)
    paths = []

    for k, motionK in enumerate(motions):
        # Keep the legacy _p1 / _p2 suffix for K=2 so existing thesis figure
        # references don't have to be updated.
        suffix = f"_p{k + 1}" if cfg.numActors == 2 else f"_actor{k}"
        outPath = os.path.join(args.outputDir, f"{args.name}{suffix}.mp4")
        log.info("rendering actor %d -> %s", k, outPath)
        renderSmplx2Video(motionK, outPath, fps=args.fps)
        paths.append(outPath)

    log.info("done")

    for p in paths:
        print(f"video -> {p}")


if __name__ == "__main__":
    main()
