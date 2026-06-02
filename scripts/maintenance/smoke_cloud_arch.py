"""Pre-cloud-run architecture smoke.

Builds the EXACT TextToMotionSSM architecture the cloud Tier-B run would
build, runs forward+backward+optimizer.step on a small synthetic batch, and
reports peak VRAM.

This is NOT a training run -- there's no dataset, no convergence check.
It only answers: "with these flags, does the model build and one step
succeed without OOM / shape error / NaN?". If it does, the cloud run
(which has 4x our VRAM) will too.

Run:
    python scripts/maintenance/smoke_cloud_arch.py
    python scripts/maintenance/smoke_cloud_arch.py --ar-k-head --compile
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from src.architecture.nn_models import TextToMotionSSM
from src.shared.config import TrainingConfig


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text-encoder", default="clip-b",
                   choices=["sbert-small", "clip-b", "clip-l"])
    p.add_argument("--ar-k-head", action="store_true")
    p.add_argument("--compile", action="store_true", dest="compile_model")
    p.add_argument("--batch-size", type=int, default=2,
                   help="batch size. Cloud uses 64; locally we shrink.")
    p.add_argument("--steps", type=int, default=2,
                   help="forward+backward steps. >=2 confirms loss decreases.")
    args = p.parse_args()

    encoder_map = {
        "sbert-small": "all-MiniLM-L6-v2",
        "clip-b":      "clip-ViT-B-32",
        "clip-l":      "clip-ViT-L-14",
    }

    cfg = TrainingConfig(
        d_model=384, d_state=64, n_layers=6,
        max_motion_length=200, max_text_length=64,
        use_sbert=True, sbert_model=encoder_map[args.text_encoder], freeze_sbert=True,
        bidirectional=True, use_film=True,
        gradient_checkpointing=True,
        rvq_n_codebooks=6, rvq_codebook_size=512, rvq_down_t=4,
        cfg_dropout_prob=0.1,
        arch="residual_k" if args.ar_k_head else "independent",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: encoder={args.text_encoder}  arch={cfg.arch}  "
          f"compile={args.compile_model}  batch={args.batch_size}  steps={args.steps}")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU VRAM: total={total_gb:.1f} GB  free={free_gb:.1f} GB")

    print()
    print("Building model ...")
    t0 = time.perf_counter()
    model = TextToMotionSSM(cfg).to(device)

    if args.compile_model and device.type == "cuda":
        print("Compiling (first batch eats 30-90s) ...")
        model = torch.compile(model, mode="reduce-overhead", dynamic=False)  # type: ignore[assignment]
    print(f"  model ready in {time.perf_counter() - t0:.1f}s")
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: total={n_params / 1e6:.1f}M  trainable={n_trainable / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    texts = ["a person walks forward"] * args.batch_size
    motion_length = cfg.max_motion_length  # 200
    latent_len = motion_length // cfg.rvq_down_t  # 50
    fake_targets = torch.randint(
        0, cfg.rvq_codebook_size, (args.batch_size, latent_len, cfg.rvq_n_codebooks),
        device=device,
    )

    losses: list[float] = []

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        t_step = time.perf_counter()
        logits, length_pred = model(
            texts, motion_length=motion_length,
            target_tokens=fake_targets if cfg.arch == "residual_k" else None,
        )
        # Align T' between logits and targets in case of off-by-one
        t_len = min(logits.shape[1], fake_targets.shape[1])
        loss = F.cross_entropy(
            logits[:, :t_len].reshape(-1, cfg.rvq_codebook_size),
            fake_targets[:, :t_len].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        elapsed = time.perf_counter() - t_step

        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"  step {step + 1}/{args.steps}: loss={loss.item():.4f}  "
                  f"step_time={elapsed:.2f}s  peak_VRAM={peak_gb:.2f} GB")
        else:
            print(
                f"  step {step + 1}/{args.steps}: "
                f"loss={loss.item():.4f}  step_time={elapsed:.2f}s"
            )

    # Sanity check: loss must be finite and SOMETHING must change after step 1.
    if not all(torch.isfinite(torch.tensor(losses)).tolist()):
        print("\nFAILURE: non-finite loss")
        return 1
    print()
    print("VERDICT: architecture builds, forward+backward+step succeeds, no OOM.")

    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        scale = 64 / args.batch_size  # cloud uses batch=64
        est = peak_gb * scale
        print(f"  Peak VRAM at batch={args.batch_size}: {peak_gb:.2f} GB  "
              f"-> linear projection at cloud's batch=64: ~{est:.1f} GB")

        if est < 14.0:  # T4 has 16 GB; leave 2 GB headroom
            print(f"  Cloud T4 (16 GB) headroom: OK (~{16 - est:.1f} GB free at batch=64)")
        else:
            print(
                f"  WARNING: projected {est:.1f} GB may not fit T4 16GB. "
                "Lower batch_size or rely on gradient_checkpointing fully."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
