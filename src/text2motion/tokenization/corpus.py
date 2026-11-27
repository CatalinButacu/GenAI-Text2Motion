from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from text2motion.motion.contracts import MotionClip
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.storage import sha256_file
from text2motion.tokenization.contracts import TokenPackPlan
from text2motion.tokenization.network.frozen import FrozenMotionTokenizer

PACK_IDENTITY_KEY = "__identity__"
PACK_FORMAT = "text2motion.token_pack"
PACK_VERSION = 1
INT16_CODES = 32768


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


def check_pack_precision(identity: dict[str, object], precision: str) -> None:
    encoded = identity.get("precision")
    if encoded != precision:
        raise ValueError(
            f"the token pack was encoded with precision {encoded!r}, but this run uses "
            f"{precision!r}; the two settings give different tokens, so rebuild the pack with "
            "the tokenize command or set the same precision in the config"
        )


def feature_paths_for(plan: TokenPackPlan) -> list[Path]:
    feature_paths = sorted(Path(plan.features_dir).glob("*.npy"))
    if not feature_paths:
        raise FileNotFoundError(
            f"no features under {plan.features_dir} -- build the pretraining corpus first"
        )
    return feature_paths


def check_pack_geometry(tokenizer: FrozenMotionTokenizer, plan: TokenPackPlan) -> None:
    if tokenizer.codebook_size > INT16_CODES:
        raise ValueError(
            f"{tokenizer.codebook_size} codes do not fit the int16 token pack (at most {INT16_CODES})"
        )
    if plan.segment_frames % tokenizer.downsample:
        raise ValueError(
            f"segment_frames {plan.segment_frames} is not a multiple of the downsample "
            f"{tokenizer.downsample}; the encoder would cut each window"
        )


def encode_clip_file(
    path: Path,
    tokenizer: FrozenMotionTokenizer,
    scaler: MotionScaler,
    plan: TokenPackPlan,
    pack: dict[str, np.ndarray],
) -> int:
    token_steps = 0
    normalized = scaler.normalize(np.load(path).astype(np.float32))
    for start in window_starts(normalized.shape[0], plan.segment_frames, plan.stride):
        window = normalized[start : start + plan.segment_frames]
        clip_id = f"{path.stem}__{start}"
        clip = MotionClip(
            features=torch.from_numpy(window),
            frame_count=window.shape[0],
            clip_id=clip_id,
        )
        tokens = tokenizer.encode(clip)
        pack[clip_id] = tokens.indices.cpu().numpy().astype(np.int16)
        token_steps += tokens.token_count
    return token_steps


def encode_kept_clips(
    kept: list[Path],
    tokenizer: FrozenMotionTokenizer,
    scaler: MotionScaler,
    plan: TokenPackPlan,
) -> tuple[dict[str, np.ndarray], int]:
    pack: dict[str, np.ndarray] = {}
    token_steps = 0
    for path in tqdm(kept, desc="encode"):
        token_steps += encode_clip_file(path, tokenizer, scaler, plan, pack)
    return pack, token_steps


def check_pack_not_empty(
    pack: dict[str, np.ndarray],
    kept: list[Path],
    excluded_present: int,
    plan: TokenPackPlan,
) -> None:
    if not pack:
        raise ValueError(
            f"no clip of {len(kept)} kept clips ({excluded_present} excluded) has "
            f"{plan.segment_frames} frames; the token pack would be empty"
        )


def make_pack_record(
    identity: dict[str, object],
    kept: list[Path],
    excluded_present: int,
    excluded: set[str],
    plan: TokenPackPlan,
) -> dict[str, object]:
    return {
        **identity,
        "format": PACK_FORMAT,
        "version": PACK_VERSION,
        "source_features": len(kept),
        "source_sha256": source_digest(kept),
        "excluded_present": excluded_present,
        "excluded_names": len(excluded),
        "segment_frames": plan.segment_frames,
        "stride": plan.stride,
        "tail": "end_aligned",
    }


def write_token_pack(pack: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, allow_pickle=False, **pack)
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)


def pack_summary(
    kept: list[Path],
    excluded_present: int,
    pack: dict[str, np.ndarray],
    token_steps: int,
    out_path: Path,
) -> dict[str, float]:
    return {
        "sequences": len(kept),
        "excluded": excluded_present,
        "segments": len(pack) - 1,
        "token_steps": token_steps,
        "pack_mb": round(out_path.stat().st_size / 1e6, 1),
    }


@torch.no_grad()
def build_token_pack(
    tokenizer: FrozenMotionTokenizer,
    scaler: MotionScaler,
    plan: TokenPackPlan,
    excluded: set[str],
    identity: dict[str, object],
) -> dict[str, float]:
    feature_paths = feature_paths_for(plan)
    kept = [path for path in feature_paths if path.name not in excluded]
    excluded_present = len(feature_paths) - len(kept)
    check_pack_geometry(tokenizer, plan)
    pack, token_steps = encode_kept_clips(kept, tokenizer, scaler, plan)
    check_pack_not_empty(pack, kept, excluded_present, plan)
    record = make_pack_record(identity, kept, excluded_present, excluded, plan)
    pack[PACK_IDENTITY_KEY] = np.array(json.dumps(record, sort_keys=True))
    out_path = Path(plan.token_pack)
    write_token_pack(pack, out_path)
    return pack_summary(kept, excluded_present, pack, token_steps, out_path)
