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

from src.architecture.nn_models import TextToMotionSSM
from src.architecture.rvq_tokenizer import MotionRVQTokenizer
from src.shared.config import TrainingConfig

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def build_tiny_rvq() -> MotionRVQTokenizer:
    return MotionRVQTokenizer(
        motion_dim=168, latent_dim=64, n_codebooks=4, codebook_size=128, down_t=4,
    )


def build_tiny_ssm() -> TextToMotionSSM:
    cfg = TrainingConfig(
        d_model=128,
        d_state=16,
        n_layers=2,
        use_sbert=False,
        use_film=False,
        bidirectional=False,
        gradient_checkpointing=False,
        max_motion_length=120,
        rvq_down_t=4,
        rvq_latent_dim=64,
        rvq_n_codebooks=4,
        rvq_codebook_size=128,
        vocab_size=512,
        use_amp=False,
    )
    return TextToMotionSSM(cfg)


def time_it(step_fn, warmup_steps: int, measure_steps: int) -> float:
    for _ in range(warmup_steps):
        step_fn()
    t0 = time.perf_counter()
    for _ in range(measure_steps):
        step_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_rvq(device: str, steps: int, batch: int) -> dict[str, float]:
    motion = torch.randn(batch, 64, 168, device=device)

    def make_rvq_step(model):  # accepts nn.Module OR torch.compile output
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def step() -> None:
            opt.zero_grad(set_to_none=True)
            recon, _, commit_loss = model(motion)
            loss = F.mse_loss(recon, motion) + 0.25 * commit_loss
            loss.backward()
            opt.step()
        return step

    baseline = build_tiny_rvq().to(device)
    base_step = make_rvq_step(baseline)
    base_sec = time_it(base_step, warmup_steps=3, measure_steps=steps)

    compiled = build_tiny_rvq().to(device)
    compiled = torch.compile(compiled, mode="reduce-overhead", dynamic=False)
    comp_step = make_rvq_step(compiled)
    # Compile mode needs more warmup (first-batch compile + reduce-overhead caches).
    comp_sec = time_it(comp_step, warmup_steps=5, measure_steps=steps)

    return {
        "baseline_steps_per_sec": steps / base_sec,
        "compile_steps_per_sec": steps / comp_sec,
        "speedup": base_sec / comp_sec,
    }


def bench_ssm(device: str, steps: int, batch: int) -> dict[str, float]:
    tokens = torch.randint(0, 512, (batch, 32), device=device)
    target = torch.randint(0, 128, (batch, 30, 4), device=device)

    def make_ssm_step(model):  # accepts nn.Module OR torch.compile output
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def step() -> None:
            opt.zero_grad(set_to_none=True)
            logits, _ = model(tokens, motion_length=120)
            t_len = min(logits.shape[1], target.shape[1])
            loss = F.cross_entropy(
                logits[:, :t_len].reshape(-1, logits.shape[-1]),
                target[:, :t_len].reshape(-1),
            )
            loss.backward()
            opt.step()
        return step

    baseline = build_tiny_ssm().to(device)
    base_step = make_ssm_step(baseline)
    base_sec = time_it(base_step, warmup_steps=3, measure_steps=steps)

    compiled = build_tiny_ssm().to(device)
    compiled = torch.compile(compiled, mode="reduce-overhead", dynamic=False)
    comp_step = make_ssm_step(compiled)
    comp_sec = time_it(comp_step, warmup_steps=5, measure_steps=steps)

    return {
        "baseline_steps_per_sec": steps / base_sec,
        "compile_steps_per_sec": steps / comp_sec,
        "speedup": base_sec / comp_sec,
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
    rows.append(("MotionRVQTokenizer", bench_rvq(device, args.steps, args.batch)))
    log.info("[bench] TextToMotionSSM ...")
    rows.append(("TextToMotionSSM", bench_ssm(device, args.steps, args.batch)))

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
    print("  Use dynamic=False with bucketed/padded batches (we already use max_motion_length).")
    print("  First-batch compile takes 30-90s; amortizes after ~3 epochs.")


if __name__ == "__main__":
    main()
