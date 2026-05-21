import argparse
import logging

import torch

from src.pipeline import Pipeline
from src.shared.config import PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

def parseArgs() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Physics-Constrained Video Generation")
    defaultDevice = "cuda" if torch.cuda.is_available() else "cpu"

    p.add_argument("prompt", nargs="?", default="a person walks forward")

    p.add_argument("--name", dest="outputName", default="output")
    p.add_argument("--output-dir", dest="outputDir", default="outputs")

    p.add_argument("--duration", type=float, default=PipelineConfig.duration)
    p.add_argument("--fps", type=int, default=PipelineConfig.fps)

    p.add_argument("--device", default=defaultDevice, choices=["cuda", "cpu"])
    p.add_argument("--no-layout-opt", dest="useRandomLayout", action="store_true")

    p.add_argument("--temperature", type=float, default=1.0,
                   help="Motion sampling : "
                   "1.0=greedy/deterministic. "
                   ">1.0 produces more varied motions; "
                   "<1.0 is more conservative.",
    )
    p.add_argument(
        "--top-p", type=float, default=1.0, dest="topP",
        help="Nucleus sampling threshold (0.0-1.0) :"
             "1.0=off. "
             "0.9 keeps top tokens whose cumulative probability sums to 0.9.",
    )

    return p.parse_args()

def main() -> None:
    args = parseArgs()

    config = PipelineConfig(
        outputDir=args.outputDir,
        duration=args.duration,
        fps=args.fps,
        device=args.device,
    )
    config.planner.randomLayout = args.useRandomLayout
    config.motion.temperature = args.temperature
    config.motion.topP = args.topP

    result = Pipeline(config).run(args.prompt, outputName=args.outputName)

    video = result.get("video_path", "")
    parsed = result.get("parsed_scene")
    print(f"\nvideo -> {video}")

    if parsed:
        print(f"entities: {[e.name for e in parsed.entities]}")
        print(f"actions : {[a.actionType for a in parsed.actions]}")

if __name__ == "__main__":
    main()
