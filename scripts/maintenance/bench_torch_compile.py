"""Measure torch.compile throughput on our two training hot paths.

What this answers: is `torch.compile(model, mode="reduce-overhead")` actually
worth it for our RVQ tokenizer and TextToMotionSSM forward+backward, on the
hardware we currently train on?

What it does NOT answer: end-to-end epoch wallclock (which is dominated by
data loading on CPU); compile cost is overhead that amortizes over many steps.

Run:
    python scripts/maintenance/bench_torch_compile.py
    python scripts/maintenance/bench_torch_compile.py --steps 50 --batch 32
    python scripts/maintenance/bench_torch_compile.py --device cuda  # if available

Output: a table of (model, compile mode, steps/sec, vs baseline %).
"""

from __future__ import annotations

import argparse
import logging
import time

import torch
import torch.nn.functional as F

from src.modules.motion.config import TrainingConfig
from src.modules.motion.nn_models import TextToMotionSSM
from src.modules.motion.rvq_tokenizer import MotionRVQTokenizer

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def buildTinyRvq() -> MotionRVQTokenizer:
    return MotionRVQTokenizer(
        motionDim=168, latentDim=64, nCodebooks=4, codebookSize=128, downT=4,
    )


def buildTinySsm() -> TextToMotionSSM:
    cfg = TrainingConfig(
        dModel=128,
        dState=16,
        nLayers=2,
        useSbert=False,
        useFilm=False,
        bidirectional=False,
        gradientCheckpointing=False,
        maxMotionLength=120,
        rvqDownT=4,
        rvqLatentDim=64,
        rvqNCodebooks=4,
        rvqCodebookSize=128,
        vocabSize=512,
        useAmp=False,
    )
    return TextToMotionSSM(cfg)


def timeIt(stepFn, warmupSteps: int, measureSteps: int) -> float:
    for _ in range(warmupSteps):
        stepFn()
    t0 = time.perf_counter()
    for _ in range(measureSteps):
        stepFn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def benchRvq(device: str, steps: int, batch: int) -> dict[str, float]:
    motion = torch.randn(batch, 64, 168, device=device)

    def makeRvqStep(model):  # accepts nn.Module OR torch.compile output
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def step() -> None:
            opt.zero_grad(set_to_none=True)
            recon, _, commitLoss = model(motion)
            loss = F.mse_loss(recon, motion) + 0.25 * commitLoss
            loss.backward()
            opt.step()
        return step

    baseline = buildTinyRvq().to(device)
    baseStep = makeRvqStep(baseline)
    baseSec = timeIt(baseStep, warmupSteps=3, measureSteps=steps)

    compiled = buildTinyRvq().to(device)
    compiled = torch.compile(compiled, mode="reduce-overhead", dynamic=False)
    compStep = makeRvqStep(compiled)
    # Compile mode needs more warmup (first-batch compile + reduce-overhead caches).
    compSec = timeIt(compStep, warmupSteps=5, measureSteps=steps)

    return {
        "baseline_steps_per_sec": steps / baseSec,
        "compile_steps_per_sec": steps / compSec,
        "speedup": baseSec / compSec,
    }


def benchSsm(device: str, steps: int, batch: int) -> dict[str, float]:
    tokens = torch.randint(0, 512, (batch, 32), device=device)
    target = torch.randint(0, 128, (batch, 30, 4), device=device)

    def makeSsmStep(model):  # accepts nn.Module OR torch.compile output
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def step() -> None:
            opt.zero_grad(set_to_none=True)
            logits, _ = model(tokens, motionLength=120)
            tLen = min(logits.shape[1], target.shape[1])
            loss = F.cross_entropy(
                logits[:, :tLen].reshape(-1, logits.shape[-1]),
                target[:, :tLen].reshape(-1),
            )
            loss.backward()
            opt.step()
        return step

    baseline = buildTinySsm().to(device)
    baseStep = makeSsmStep(baseline)
    baseSec = timeIt(baseStep, warmupSteps=3, measureSteps=steps)

    compiled = buildTinySsm().to(device)
    compiled = torch.compile(compiled, mode="reduce-overhead", dynamic=False)
    compStep = makeSsmStep(compiled)
    compSec = timeIt(compStep, warmupSteps=5, measureSteps=steps)

    return {
        "baseline_steps_per_sec": steps / baseSec,
        "compile_steps_per_sec": steps / compSec,
        "speedup": baseSec / compSec,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (defaults to cuda if available)")
    parser.add_argument("--steps", type=int, default=20,
                        help="measured steps per run (after warmup)")
    parser.add_argument("--batch", type=int, default=8,
                        help="batch size; bigger = compile shows more")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[bench] device=%s steps=%d batch=%d", device, args.steps, args.batch)

    if device == "cpu":
        log.warning("[bench] CPU benchmark: torch.compile often hurts on CPU due "
                    "to dispatch overhead. Results inform but don't decide.")

    rows: list[tuple[str, dict[str, float]]] = []
    log.info("[bench] RVQ tokenizer ...")
    rows.append(("MotionRVQTokenizer", benchRvq(device, args.steps, args.batch)))
    log.info("[bench] TextToMotionSSM ...")
    rows.append(("TextToMotionSSM", benchSsm(device, args.steps, args.batch)))

    print()
    print(f"{'model':<22}  {'baseline (steps/s)':<20}  "
          f"{'compile (steps/s)':<20}  {'speedup':<8}")
    print("-" * 76)
    for name, m in rows:
        print(
            f"{name:<22}  {m['baseline_steps_per_sec']:<20.2f}  "
            f"{m['compile_steps_per_sec']:<20.2f}  {m['speedup']:<8.2f}x"
        )

    print()
    print("How to apply (if compile wins on GPU):")
    print("  In the trainer, after model construction:")
    print("    model = torch.compile(model, mode='reduce-overhead', dynamic=False)")
    print("  Use dynamic=False with bucketed/padded batches (we already use maxMotionLength).")
    print("  First-batch compile takes 30-90s; amortizes after ~3 epochs.")


if __name__ == "__main__":
    main()
