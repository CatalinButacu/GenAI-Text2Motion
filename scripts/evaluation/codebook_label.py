"""Surface RVQ codebook prototypes for human labeling (option A from methodology).

The RVQ tokenizer is unsupervised — it learns motion atoms with no text labels.
This script renders short SMPL-X GIFs for the most-used entries of a codebook,
so a human can watch each prototype and assign a verb-phrase label. Together,
the labels form the atomic motion vocabulary discovered by the tokenizer.

Compound actions ("walk and then sit") emerge naturally as sequences of these
atomic tokens — see scripts/evaluation/vocab_compose.py + CodebookVocab for
inference-time decomposition.

Usage:
    python scripts/evaluation/codebook_label.py \\
        --checkpoint checkpoints/rvq_tokenizer/<run_id>/best_model.pt \\
        --codebook 0 --top-k 64 --n-prototypes 3
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluation.eval_rvq import build_test_dataset, load_checkpoint
from scripts.evaluation.render_rvq_samples import render_smplx_to_gif
from src.data.motion_normalize import MotionStats, denormalize

log = logging.getLogger(__name__)

@torch.no_grad()
def encode_all(model, dataset, device, batch_size: int):
    """Encode the whole dataset once. Cache indices, motion, latent mask, texts."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    indices_list: list[torch.Tensor] = []
    motion_list: list[torch.Tensor] = []
    mask_lat_list: list[torch.Tensor] = []
    texts: list[str] = []
    down_t = model.down_t

    for batch in loader:
        motion = batch["motion"].to(device)
        mask = batch["motion_mask"].to(device)
        x = motion.transpose(1, 2)
        z = model.encoder(x).transpose(1, 2)
        _, indices, _ = model.rvq(z)
        B, Tpad = mask.shape
        Tlat = indices.shape[1]
        lat_mask = torch.zeros(B, Tlat, dtype=torch.bool, device=mask.device)

        for tl in range(Tlat):
            start = tl * down_t
            end = min(start + down_t, Tpad)
            lat_mask[:, tl] = mask[:, start:end].any(dim=1)
        indices_list.append(indices.cpu())
        motion_list.append(motion.cpu())
        mask_lat_list.append(lat_mask.cpu())
        raw_texts = batch.get("texts", [""] * B)

        for t in raw_texts:
            texts.append(t if isinstance(t, str) else "")

    return (
        torch.cat(indices_list, dim=0),
        torch.cat(motion_list, dim=0),
        torch.cat(mask_lat_list, dim=0),
        texts,
    )

def top_entries_by_frequency(indices: torch.Tensor, mask: torch.Tensor,
                          codebook: int, top_k: int) -> list[tuple[int, int]]:
    flat_idx = indices[..., codebook]
    flat_vals = flat_idx[mask].numpy()
    bins = np.bincount(flat_vals)
    order = np.argsort(bins)[::-1]

    return [(int(e), int(bins[e])) for e in order[:top_k] if bins[e] > 0]

def pick_prototypes(indices: torch.Tensor, mask: torch.Tensor, codebook: int,
                   entry: int, n_proto: int) -> list[tuple[int, int]]:
    # One exemplar per distinct clip — gives diversity in body shape/context
    # without needing a second forward pass to compute exact distances.
    flat_idx = indices[..., codebook]
    matches = ((flat_idx == entry) & mask).nonzero(as_tuple=False)

    if matches.numel() == 0:
        return []
    seen: set[int] = set()
    out: list[tuple[int, int]] = []

    for nt in matches:
        clip_idx = int(nt[0])
        frame_lat = int(nt[1])

        if clip_idx in seen:
            continue
        seen.add(clip_idx)
        out.append((clip_idx, frame_lat))

        if len(out) >= n_proto:
            break

    return out

def render_prototype(motion: torch.Tensor, frame_lat: int, down_t: int,
                    window_frames: int, stats: MotionStats, out_path: str,
                    fps: int) -> int:
    center = frame_lat * down_t + down_t // 2
    half = window_frames // 2
    Tpad = motion.shape[0]
    start = max(0, center - half)
    end = min(Tpad, center + half)
    clip = motion[start:end].cpu().numpy()
    smplx = denormalize(clip, stats)
    render_smplx_to_gif(smplx, out_path, fps=fps)

    return end - start

def top_text_snippets(indices: torch.Tensor, mask: torch.Tensor, codebook: int,
                    entry: int, texts: list[str], max_snippets: int = 5) -> list[str]:
    flat_idx = indices[..., codebook]
    matches = ((flat_idx == entry) & mask).nonzero(as_tuple=False)
    seen: set[int] = set()
    out: list[str] = []

    for nt in matches:
        clip_idx = int(nt[0])

        if clip_idx in seen or clip_idx >= len(texts):
            continue
        seen.add(clip_idx)
        snippet = texts[clip_idx].strip().replace("\n", " ")[:80]

        if snippet:
            out.append(snippet)

        if len(out) >= max_snippets:
            break

    return out

def write_readme(out_dir: str, run_id: str, codebook: int, n_entries: int) -> None:
    text = f"""# Codebook labeling — `{run_id}` codebook {codebook}

This directory holds the prototypes for the {n_entries} most-frequent entries of
RVQ codebook {codebook}.

## How to label

Open `labels.csv` in a spreadsheet. For each row, watch the prototype GIFs
listed in `prototype_paths` and write a short verb-phrase in the `label`
column. Examples: `walk_forward`, `arms_above_head`, `sit_down`, `idle`.

## Conventions

- One verb per label (no compounds — those emerge as token sequences).
- Lowercase, snake_case.
- Use `idle` for ambiguous static poses.
- Use `transition` if it's clearly a between-state movement.
- Use `noise` if the prototype looks broken/garbled (these entries get
  dropped from the vocab).

## What happens after

Run `python scripts/evaluation/vocab_compose.py --labels labels.csv ...`
to test compound-text decomposition (e.g. "walk then sit" -> token sequence).
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    parser.add_argument("--codebook", type=int, default=0,
                        help="Which RVQ layer to label (0 = coarsest, recommended)")
    parser.add_argument("--top-k", type=int, default=64, dest="top_k",
                        help="Label only the K most-used entries (rest are rare/dead)")
    parser.add_argument("--n-prototypes", type=int, default=3, dest="n_prototypes")
    parser.add_argument("--window-frames", type=int, default=32, dest="window_frames",
                        help="Length of each prototype clip in motion frames (~1s @30fps)")
    parser.add_argument("--stats-path", default="data/stats/amass_full.npz", dest="stats_path")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S")

    run_id = os.path.basename(os.path.dirname(args.checkpoint))

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.checkpoint),
                                       "eval", "codebook_labels")
    os.makedirs(args.output_dir, exist_ok=True)
    log.info("[label] outputs -> %s", args.output_dir)

    if not os.path.exists(args.stats_path):
        log.error("[label] stats not found: %s", args.stats_path)

        return 1
    s = np.load(args.stats_path)
    stats = MotionStats(mean=s["mean"], std=s["std"])

    device = torch.device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    dataset, src = build_test_dataset(cfg, args.stats_path)
    log.info("[label] test set: %d samples (source=%s)", len(dataset), src)

    log.info("[label] encoding test set ...")
    indices, motion, lat_mask, texts = encode_all(model, dataset, device, args.batch_size)
    log.info("[label] encoded N=%d  Tlat=%d  K=%d",
             indices.shape[0], indices.shape[1], indices.shape[2])

    top_entries = top_entries_by_frequency(indices, lat_mask, args.codebook, args.top_k)
    log.info("[label] codebook %d: %d entries to label (covers %d/%d steps)",
             args.codebook, len(top_entries),
             sum(c for _, c in top_entries),
             int(lat_mask.sum().item()))

    csv_path = os.path.join(args.output_dir, "labels.csv")
    rows: list[dict] = []

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "codebook_idx", "entry_idx", "n_assigned", "prototype_paths",
            "top_texts", "label", "notes",
        ])
        writer.writeheader()

        for rank, (entry, count) in enumerate(top_entries):
            entry_dir = os.path.join(args.output_dir, f"cb{args.codebook}_e{entry:04d}")
            os.makedirs(entry_dir, exist_ok=True)
            picks = pick_prototypes(indices, lat_mask, args.codebook,
                                    entry, args.n_prototypes)
            paths: list[str] = []

            for k, (clip_idx, frame_lat) in enumerate(picks):
                out_path = os.path.join(entry_dir, f"proto_{k}.gif")

                try:
                    Tactual = render_prototype(
                        motion[clip_idx], frame_lat, model.down_t, args.window_frames,
                        stats, out_path, args.fps,
                    )
                except (RuntimeError, OSError) as exc:
                    log.warning("[label] cb%d e%d proto%d: render failed: %s",
                                args.codebook, entry, k, exc)
                    continue
                rel = os.path.relpath(out_path, args.output_dir)
                paths.append(rel)
                log.info("[label] %3d/%d cb%d e%d  proto%d  T=%d  -> %s",
                         rank + 1, len(top_entries), args.codebook, entry, k, Tactual, rel)

            snippets = top_text_snippets(indices, lat_mask, args.codebook, entry, texts)
            row = {
                "codebook_idx": args.codebook,
                "entry_idx": entry,
                "n_assigned": count,
                "prototype_paths": ";".join(paths),
                "top_texts": " | ".join(snippets),
                "label": "",
                "notes": "",
            }
            writer.writerow(row)
            rows.append(row)

    write_readme(args.output_dir, run_id, args.codebook, len(top_entries))
    log.info("[label] DONE  csv=%s  prototypes=%d  see README.md for instructions",
             csv_path, len(rows))

    return 0

if __name__ == "__main__":
    sys.exit(main())
