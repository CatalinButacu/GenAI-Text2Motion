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
log = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Physics-Constrained Video Generation")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

    p.add_argument("prompt", nargs="?", default="a person walks forward")

    p.add_argument("--name", dest="output_name", default="output")
    p.add_argument("--output-dir", dest="output_dir", default="outputs")

    p.add_argument("--duration", type=float, default=PipelineConfig.duration)
    p.add_argument("--fps", type=int, default=PipelineConfig.fps)

    p.add_argument("--device", default=default_device, choices=["cuda", "cpu"])
    p.add_argument("--no-layout-opt", dest="use_random_layout", action="store_true")

    p.add_argument("--temperature", type=float, default=1.0,
                   help="Motion sampling : "
                   "1.0=greedy/deterministic. "
                   ">1.0 produces more varied motions; "
                   "<1.0 is more conservative.",
    )
    p.add_argument(
        "--top-p", type=float, default=1.0, dest="top_p",
        help="Nucleus sampling threshold (0.0-1.0) :"
             "1.0=off. "
             "0.9 keeps top tokens whose cumulative probability sums to 0.9.",
    )
    p.add_argument(
        "--cfg-scale", type=float, default=1.0, dest="cfg_scale",
        help="Classifier-free guidance scale. 1.0=off (vanilla conditional). "
             "2-4 typical; higher = stronger text adherence, less diversity. "
             "Requires the SSM trained with cfg_dropout_prob > 0 and use_sbert=True.",
    )
    p.add_argument(
        "--stream", action="store_true",
        help="Print a live stage-by-stage breakdown with timings to stdout.",
    )
    p.add_argument(
        "--viewer", action="store_true",
        help="Open interactive aitviewer window instead of saving to MP4.",
    )
    p.add_argument(
        "--chat", action="store_true",
        help="Open the chat viewer: a window with a prompt bar that re-runs the "
             "pipeline live whenever you type a new prompt.",
    )
    p.add_argument(
        "--ssm-checkpoint", dest="ssm_checkpoint", default=None,
        help="Path to a MotionSSM best_model.pt (overrides the default in MotionConfig).",
    )

    return p.parse_args()

def main() -> None:
    args = parse_args()

    config = PipelineConfig(
        output_dir=args.output_dir,
        duration=args.duration,
        fps=args.fps,
        device=args.device,
    )
    config.planner.random_layout = args.use_random_layout
    config.motion.temperature = args.temperature
    config.motion.top_p = args.top_p
    config.motion.cfg_scale = args.cfg_scale
    if args.ssm_checkpoint:
        config.motion.checkpoint_path = args.ssm_checkpoint

    if args.chat:
        run_chat(config, initial_prompt=args.prompt, stream=args.stream)
        return

    result = Pipeline(config).run(
        args.prompt,
        output_name=args.output_name,
        stream=args.stream,
        viewer=args.viewer,
    )

    video = result.get("video_path", "")
    parsed = result.get("parsed_scene")
    print(f"\nvideo -> {video}")

    if parsed:
        print(f"entities: {[e.name for e in parsed.entities]}")
        print(f"actions : {[a.action_type for a in parsed.actions]}")


def run_chat(config: PipelineConfig, initial_prompt: str, stream: bool) -> None:
    """Build the pipeline once, then open the chat viewer.

    Stages 1-3 (understanding → planner → motion) run to get the initial clip.
    No headless render is done before opening the window, avoiding GL context conflicts.
    """
    from src.modules import motion, planner, understanding
    from src.modules.render.chat_viewer import ChatViewer

    def run_stages(prompt: str) -> dict | None:
        parsed = understanding.invoke(prompt, config.understanding)
        planned = planner.invoke(parsed, config.planner)
        clips = motion.invoke(planned, config.motion)
        if not clips:
            return None
        clip = next(iter(clips.values()))
        return {
            "smplx_params": clip.smplx_params,
            "betas": clip.betas,
            "gender": config.render.gender,
            "input_coord_system": config.render.input_coord_system,
        }

    initial_clip = None
    if initial_prompt:
        log.info("[chat] generating initial motion for %r ...", initial_prompt)
        initial_clip = run_stages(initial_prompt)

    log.info("[chat] opening window. Type prompts at the bottom of the screen.")
    viewer = ChatViewer(
        pipeline_runner=run_stages,
        initial_clip=initial_clip,
        fps=config.fps,
    )
    viewer.run()

if __name__ == "__main__":
    main()
