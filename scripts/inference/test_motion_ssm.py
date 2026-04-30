"""
SSM Motion Inference
====================
Test the trained TextToMotionSSM model with various prompts.

Usage:
    python scripts/test_motion_ssm.py "A person walks forward"
    python scripts/test_motion_ssm.py --interactive
"""

import argparse
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from src.modules.motion.config import TrainingConfig
from src.modules.motion.nn_models import TextToMotionSSM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MotionInference:
    """Inference wrapper for trained TextToMotionSSM."""

    def __init__(self, checkpointPath: str = "checkpoints/motion_ssm/best_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {checkpointPath}")

        checkpoint = torch.load(checkpointPath, map_location=self.device, weights_only=False)

        self.config = checkpoint["config"]
        self.vocab = checkpoint["vocab"]
        self.valLoss = checkpoint.get("valLoss", "unknown")

        # Rebuild model
        self.model = TextToMotionSSM(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        logger.info(f"Model loaded! Val loss: {self.valLoss:.4f}")
        logger.info(f"Vocab size: {len(self.vocab)}, Device: {self.device}")

    def tokenize(self, text: str, maxLength: int = 64) -> torch.Tensor:
        """Convert text to token IDs."""
        words = text.lower().split()
        tokens = [self.vocab.get("<BOS>", 2)]
        for word in words[:maxLength-2]:
            tokens.append(self.vocab.get(word, self.vocab.get("<UNK>", 1)))
        tokens.append(self.vocab.get("<EOS>", 3))

        # Pad
        if len(tokens) < maxLength:
            tokens.extend([0] * (maxLength - len(tokens)))

        return torch.tensor(tokens[:maxLength], dtype=torch.long).unsqueeze(0)

    @torch.no_grad()
    def generate(self, text: str, numFrames: int = 100) -> dict:
        """Generate motion from text."""
        logger.info(f"Generating motion for: '{text}'")

        tokenIds = self.tokenize(text).to(self.device)

        # Generate
        motion, lengthPred = self.model(tokenIds, numFrames)

        motion = motion.cpu().numpy()[0]  # (frames, 168)
        lengthPred = int(lengthPred.cpu().numpy()[0])

        motionDenorm = motion

        return {
            "text": text,
            "motion": motion,
            "motion_denorm": motionDenorm,
            "predicted_length": lengthPred,
            "numFrames": numFrames
        }

    def analyzeMotion(self, result: dict) -> dict:
        """Analyze generated motion statistics."""
        motion = result["motion"]

        # Compute statistics
        stats = {
            "shape": motion.shape,
            "mean": float(np.mean(motion)),
            "std": float(np.std(motion)),
            "min": float(np.min(motion)),
            "max": float(np.max(motion)),
            "variance_per_frame": float(np.mean(np.var(motion, axis=1))),
            "smoothness": float(np.mean(np.abs(np.diff(motion, axis=0)))),
        }

        return stats

    def compareMotions(self, texts: list) -> None:
        """Generate and compare motions for multiple texts."""
        print("\n" + "=" * 70)
        print("MOTION COMPARISON")
        print("=" * 70)

        for text in texts:
            result = self.generate(text, numFrames=60)
            stats = self.analyzeMotion(result)

            print(f"\n[{text}]")
            print(f"  Predicted length: {result['predicted_length']} frames")
            print(f"  Motion stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            print(f"  Smoothness: {stats['smoothness']:.4f}")
            print(f"  Variance: {stats['variance_per_frame']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test Motion SSM")
    parser.add_argument("prompt", nargs="?", default=None, help="Text prompt for motion")
    parser.add_argument(
        "--checkpoint", default="checkpoints/motion_ssm/best_model.pt", help="Model path",
    )
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to generate")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--compare", action="store_true", help="Compare multiple prompts")
    args = parser.parse_args()

    print("=" * 60)
    print("  Motion SSM Inference")
    print("=" * 60)

    inference = MotionInference(args.checkpoint)

    if args.compare:
        testPrompts = [
            "A person walks forward",
            "A person runs fast",
            "A person kicks a ball",
            "A person jumps up",
            "A person waves hand",
            "A person turns around",
        ]
        inference.compareMotions(testPrompts)
        return

    if args.interactive:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            text = input("Enter prompt: ").strip()
            if text.lower() in ["quit", "exit", "q"]:
                break
            if not text:
                continue

            result = inference.generate(text, args.frames)
            stats = inference.analyzeMotion(result)

            print(f"\n  Generated {result['num_frames']} frames")
            print(f"  Predicted duration: {result['predicted_length']/30:.1f}s")
            print(f"  Motion shape: {stats['shape']}")
            print(f"  Smoothness: {stats['smoothness']:.4f}")
            print()
    else:
        prompt = args.prompt or "A person walks forward"
        result = inference.generate(prompt, args.frames)
        stats = inference.analyzeMotion(result)

        print(f"\n  Prompt: {prompt}")
        print(f"  Generated: {result['num_frames']} frames")
        predicted = result["predicted_length"]
        print(f"  Predicted length: {predicted} frames ({predicted / 30:.1f}s)")
        print(f"  Motion shape: {stats['shape']}")
        print(f"  Motion mean: {stats['mean']:.4f}")
        print(f"  Motion std: {stats['std']:.4f}")
        print(f"  Smoothness (lower=smoother): {stats['smoothness']:.4f}")
        print()

        # Save motion
        outputPath = "outputs/generated_motion.npy"
        os.makedirs("outputs", exist_ok=True)
        np.save(outputPath, result["motion_denorm"])
        print(f"  Motion saved to: {outputPath}")


if __name__ == "__main__":
    main()
