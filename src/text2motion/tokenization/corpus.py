from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from text2motion.motion.dataset import MotionScaler
from text2motion.motion.model import MotionClip
from text2motion.tokenization.model import MotionTokenizer


@dataclass(frozen=True)
class CorpusRequest:
    features_dir: Path
    out_path: Path
    segment_frames: int = 196
    stride: int = 196


@torch.no_grad()
def tokenize_corpus(
    tokenizer: MotionTokenizer,
    scaler: MotionScaler,
    request: CorpusRequest,
) -> dict[str, float]:
    features = sorted(Path(request.features_dir).glob("*.npy"))
    if not features:
        raise FileNotFoundError(
            f"no features under {request.features_dir} -- build the pretraining corpus first"
        )

    pack: dict[str, np.ndarray] = {}
    segments = 0
    token_steps = 0

    for path in tqdm(features, desc="encode"):
        feat = scaler.normalize(np.load(path).astype(np.float32))
        for start in range(0, feat.shape[0] - request.segment_frames + 1, request.stride):
            window = feat[start : start + request.segment_frames]
            clip = MotionClip(
                features=torch.from_numpy(window),
                frame_count=window.shape[0],
                clip_id=f"{path.stem}__{start}",
            )
            tokens = tokenizer.encode(clip)
            indices = tokens.indices.cpu().numpy().astype(np.int16)
            pack[clip.clip_id] = indices
            segments += 1
            token_steps += tokens.token_count

    out_path = Path(request.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **pack)

    return {
        "sequences": len(features),
        "segments": segments,
        "token_steps": token_steps,
        "pack_mb": round(out_path.stat().st_size / 1e6, 1),
    }
