from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from text2motion.motion.model import MotionClip
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.storage import sha256_file
from text2motion.tokenization.contracts import TokenizeRequest
from text2motion.tokenization.nn_model import FrozenMotionTokenizer

PACK_IDENTITY_KEY = "__identity__"
PACK_FORMAT = "text2motion.token_pack"
PACK_VERSION = 1


def window_starts(frames: int, segment: int, stride: int) -> list[int]:
    if frames < segment:
        return []
    starts = list(range(0, frames - segment + 1, stride))
    if starts[-1] != frames - segment:
        starts.append(frames - segment)
    return starts


def source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(f"{path.name}:{sha256_file(path)}\n".encode())
    return digest.hexdigest()


def pack_clip_keys(pack: np.lib.npyio.NpzFile) -> list[str]:
    return [key for key in pack.files if key != PACK_IDENTITY_KEY]


def read_pack_identity(path: str | Path) -> dict[str, object]:
    with np.load(path) as pack:
        if PACK_IDENTITY_KEY not in pack.files:
            raise ValueError(
                f"{path} is a token pack without an identity record (legacy); "
                "rebuild it with the tokenize command"
            )
        identity = json.loads(str(pack[PACK_IDENTITY_KEY]))
    if identity.get("format") != PACK_FORMAT or identity.get("version") != PACK_VERSION:
        raise ValueError(f"{path}: unsupported token pack identity {identity.get('format')!r}")
    return identity


@torch.no_grad()
def build_token_pack(
    tokenizer: FrozenMotionTokenizer,
    scaler: MotionScaler,
    request: TokenizeRequest,
    excluded: set[str],
    identity: dict[str, object],
) -> dict[str, float]:
    feature_paths = sorted(Path(request.features_dir).glob("*.npy"))
    if not feature_paths:
        raise FileNotFoundError(
            f"no features under {request.features_dir} -- build the pretraining corpus first"
        )
    kept = [path for path in feature_paths if path.name not in excluded]

    pack: dict[str, np.ndarray] = {}
    token_steps = 0
    for path in tqdm(kept, desc="encode"):
        normalized = scaler.normalize(np.load(path).astype(np.float32))
        for start in window_starts(normalized.shape[0], request.segment_frames, request.stride):
            window = normalized[start : start + request.segment_frames]
            clip_id = f"{path.stem}__{start}"
            clip = MotionClip(
                features=torch.from_numpy(window),
                frame_count=window.shape[0],
                clip_id=clip_id,
            )
            tokens = tokenizer.encode(clip)
            pack[clip_id] = tokens.indices.cpu().numpy().astype(np.int16)
            token_steps += tokens.token_count
    if not pack:
        raise ValueError(
            f"no clip of {len(kept)} kept clips ({len(feature_paths) - len(kept)} excluded) has "
            f"{request.segment_frames} frames; the token pack would be empty"
        )

    record = {
        **identity,
        "format": PACK_FORMAT,
        "version": PACK_VERSION,
        "source_features": len(kept),
        "source_sha256": source_digest(kept),
        "excluded_present": len(feature_paths) - len(kept),
        "excluded_names": len(excluded),
        "segment_frames": request.segment_frames,
        "stride": request.stride,
        "tail": "end_aligned",
    }
    pack[PACK_IDENTITY_KEY] = np.array(json.dumps(record, sort_keys=True))

    out_path = Path(request.token_pack)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, allow_pickle=False, **pack)
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "sequences": len(kept),
        "excluded": len(feature_paths) - len(kept),
        "segments": len(pack) - 1,
        "token_steps": token_steps,
        "pack_mb": round(out_path.stat().st_size / 1e6, 1),
    }
