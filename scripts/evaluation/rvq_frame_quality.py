"""Per-frame RVQ reconstruction quality buckets on the test set."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.evaluation.eval_rvq import buildTestDataset, loadCheckpoint

log = logging.getLogger(__name__)

BUCKETS = [
    ("excellent", 0.05),
    ("good", 0.10),
    ("acceptable", 0.30),
    ("rough", 1.00),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32, dest="batchSize")
    parser.add_argument("--stats-path", default="data/stats/amass_full.npz", dest="statsPath")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    device = torch.device(args.device)
    model, cfg = loadCheckpoint(args.checkpoint, device)
    statsPath = args.statsPath if os.path.exists(args.statsPath) else None
    dataset, src = buildTestDataset(cfg, statsPath)
    log.info("[frame-qual] test set: %d clips, source=%s", len(dataset), src)

    loader = DataLoader(dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)
    perFrameMse: list[float] = []
    perClipMse: list[float] = []
    totalFramesSeen = 0

    with torch.no_grad():
        for batch in loader:
            motion = batch["motion"].to(device)
            mask = batch["motion_mask"].to(device)  # (B, T), 1=real frame
            x = motion.transpose(1, 2)
            z = model.encoder(x).transpose(1, 2)
            quantized, _, _ = model.rvq(z)
            recon = model.decoder(quantized.transpose(1, 2)).transpose(1, 2)
            diffSq = (recon - motion) ** 2  # (B, T, D)
            # Per-frame MSE: average over channels D (the renderer cares about
            # the average error magnitude visible in any single frame).
            frameMse = diffSq.mean(dim=2)  # (B, T)

            for b in range(motion.shape[0]):
                validIdx = mask[b].bool()
                fmse = frameMse[b][validIdx].cpu().numpy()
                perFrameMse.extend(fmse.tolist())
                totalFramesSeen += int(validIdx.sum().item())
                clipMse = float((diffSq[b] * mask[b].unsqueeze(-1)).sum().item()
                                / max(validIdx.sum().item() * motion.shape[2], 1))
                perClipMse.append(clipMse)

    framesArr = np.array(perFrameMse)
    clipsArr = np.array(perClipMse)

    log.info("[frame-qual] %d total frames across %d clips", framesArr.size, clipsArr.size)
    print("\n=== Per-FRAME quality breakdown ===")
    print(f"{'tier':<12} {'threshold':>12} {'frames':>10} {'pct':>8}  {'cumul pct':>10}")
    cumul = 0

    for name, thr in BUCKETS:
        cnt = int(((framesArr <= thr) & (framesArr > (0 if name == "excellent" else BUCKETS[
            [b[0] for b in BUCKETS].index(name) - 1
        ][1]))).sum()) if name != "excellent" else int((framesArr <= thr).sum())
        pct = 100.0 * cnt / framesArr.size
        cumul += pct
        print(f"{name:<12} {'<= ' + str(thr):>12} {cnt:>10d} {pct:>7.2f}% {cumul:>9.2f}%")
    broken = int((framesArr > BUCKETS[-1][1]).sum())
    bpct = 100.0 * broken / framesArr.size
    print(f"{'broken':<12} {'>  ' + str(BUCKETS[-1][1]):>12} {broken:>10d} {bpct:>7.2f}%   100.00%")

    print("\n=== Per-CLIP quality breakdown ===")
    print(f"{'tier':<12} {'threshold':>12} {'clips':>10} {'pct':>8}")
    cumulC = 0

    for name, thr in BUCKETS:
        cnt = int(((clipsArr <= thr) & (clipsArr > (0 if name == "excellent" else BUCKETS[
            [b[0] for b in BUCKETS].index(name) - 1
        ][1]))).sum()) if name != "excellent" else int((clipsArr <= thr).sum())
        pct = 100.0 * cnt / clipsArr.size
        cumulC += pct
        print(f"{name:<12} {'<= ' + str(thr):>12} {cnt:>10d} {pct:>7.2f}%")
    brokenC = int((clipsArr > BUCKETS[-1][1]).sum())
    print(f"{'broken':<12} {'>  ' + str(BUCKETS[-1][1]):>12} {brokenC:>10d} "
          f"{100.0 * brokenC / clipsArr.size:>7.2f}%")

    print("\n=== Distribution stats ===")
    print(f"per-frame MSE:  mean={framesArr.mean():.4f}  median={np.median(framesArr):.4f}  "
          f"max={framesArr.max():.4f}")
    print(f"per-clip MSE:   mean={clipsArr.mean():.4f}  median={np.median(clipsArr):.4f}  "
          f"max={clipsArr.max():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
