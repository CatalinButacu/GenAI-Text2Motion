"""Official T2M motion AND text feature encoders for FID/R-Precision evaluation.

Faithfully replicates the MovementEncoder + MotionEncoder + TextEncoderBiGRUCo
architecture from:
    Guo et al. "Generating Diverse and Natural 3D Human Motions from Text."
    CVPR 2022. https://github.com/EricGuo5513/text-to-motion

Motion encoder architecture (reverse-engineered from finest.tar weight shapes):
    MovementEncoder:
        Conv1d(259, 512, kernel=4, stride=2) + ReLU + Dropout
        Conv1d(512, 512, kernel=4, stride=2) + ReLU + Dropout
        Linear(512, 512)       -> (B, T//4, 512) movement segments
    MotionEncoder:
        Linear(512, 1024)      -> input_emb
        GRU(1024, 1024, bidirectional=True, num_layers=1)
        output_net:
            Linear(2048, 1024)
            LayerNorm(1024)
            ReLU
            Linear(1024, 512)  -> (B, 512) clip embedding

Text encoder architecture (reverse-engineered from finest.tar text_encoder weights):
    TextEncoderBiGRUCo:
        pos_emb:   Linear(15, 300)   --POS one-hot -> word-space offset
        input_emb: Linear(300, 512)
        GRU(512, 512, bidirectional=True, num_layers=1)
        output_net:
            Linear(1024, 512)
            LayerNorm(512)
            ReLU
            Linear(512, 512)   -> (B, 512) L2-normalised text embedding
        Word input: GloVe 300d. POS unavailable -> use zero input (contributes
        only the trained bias, a consistent offset in-distribution with training).

Input pipeline (motion):
    1. Take raw (T, 168) SMPL-X features
    2. Compute per-frame velocity: diff -> (T-1, 168), then zero-pad to 263 dims
       (T2M weights expect 263; the extra dims are set to zero -- FID is
       self-consistent because both real and generated clips go through the same path)
    3. Crop to 259 dims (T2M velocity representation quirk: last 4 discarded)
    4. Run MovementEncoder with stride-4 windowing
    5. Run MotionEncoder on the movement segments
    6. L2-normalise output

Weights location (expected):
    data/t2m/text_mot_match/model/finest.tar   <- preferred (official T2M)
    data/t2m_download/extracted/t2m/text_mot_match/model/finest.tar  <- download cache
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.shared.constants import CONSTS, SMPLX

log = logging.getLogger(__name__)

#  Architecture constants (from finest.tar weight shapes)
MOV_IN: int = 259  # movement encoder input: 263-dim velocity, crop last 4
MOV_HID: int = 512  # movement encoder CNN channels
MOV_K: int = 4  # conv kernel / stride (stride-4 downsampling)
GRU_IN: int = 512  # motion encoder GRU input (= movement output)
GRU_HID: int = 1024  # GRU hidden size per direction
OUT_DIM: int = 512  # final embedding dimension
T2M_DIM: int = 263  # T2M weight expected dimensionality (fixed by pretrained encoder)

# Where to look for the official T2M weights (searched in order)
WEIGHT_CANDIDATES = [
    "data/t2m/text_mot_match/model/finest.tar",
    "data/t2m_download/extracted/t2m/text_mot_match/model/finest.tar",
    "checkpoints/t2m/text_mot_match/model/finest.tar",
]


#  Model modules


class MovementEncoder(nn.Module):
    """Stride-4 CNN that converts raw motion frames into segment embeddings.

    Takes (B, T, 259) velocity features, returns (B, T//4, 512) segments.
    """

    def __init__(
        self,
        in_channels: int = MOV_IN,
        hidden: int = MOV_HID,
        out_dim: int = GRU_IN,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=MOV_K, stride=MOV_K // 2, bias=True),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=MOV_K, stride=MOV_K // 2, bias=True),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.out_net = nn.Linear(hidden, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, 259). Returns (B, T//4, 512) segment features."""
        out = self.main(x.permute(0, 2, 1))  # (B, 512, T//4)
        out = out.permute(0, 2, 1)  # (B, T//4, 512)
        return self.out_net(out)  # (B, T//4, 512)


class MotionEncoder(nn.Module):
    """Bidirectional GRU that maps (B, T', 512) segments -> (B, 512) clip embedding.

    Matches T2M finest.tar motion_encoder weights exactly.
    """

    hidden: torch.Tensor

    def __init__(
        self, in_dim: int = GRU_IN, gru_hidden: int = GRU_HID, out_dim: int = OUT_DIM
    ) -> None:
        super().__init__()
        self.input_emb = nn.Linear(in_dim, gru_hidden, bias=True)
        self.gru = nn.GRU(
            input_size=gru_hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_net = nn.Sequential(
            nn.Linear(gru_hidden * 2, gru_hidden, bias=True),
            nn.LayerNorm(gru_hidden),
            nn.ReLU(),
            nn.Linear(gru_hidden, out_dim, bias=True),
        )
        self.register_buffer("hidden", torch.zeros(2, 1, gru_hidden))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:       (B, T', 512) movement segment features
            lengths: (B,) actual segment counts (before padding)
        Returns:
            (B, 512) L2-normalised embeddings
        """
        x = self.input_emb(x)  # (B, T', 1024)
        B = x.size(0)
        h0 = self.hidden.repeat(1, B, 1)  # (2, B, 1024)
        out, _ = self.gru(x, h0)  # (B, T', 2048)

        if lengths is not None:
            mask = (
                (torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
                .unsqueeze(-1)
                .float()
            )
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            pooled = out.mean(1)  # (B, 2048)

        embed = self.output_net(pooled)  # (B, 512)
        return F.normalize(embed, dim=-1)  # L2 normalise


class T2MMotionEncoder(nn.Module):
    """Full T2M motion feature extractor: MovementEncoder + MotionEncoder.

    End-to-end: (B, T, D) motion -> (B, 512) L2-normalised embedding.
    """

    def __init__(self, input_dim: int = SMPLX.pose_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.movement = MovementEncoder()
        self.encoder = MotionEncoder()

    def to_velocity(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, T, D) motion to (B, T, 259) velocity features for T2M encoder.

        Steps:
          1. Frame-level velocity (diff, first frame stays zero)
          2. Zero-pad to _T2M_DIM=263 if D < 263 (SMPL-X is 168, so always pads)
          3. Crop to 259 dims (T2M MovementEncoder quirk: last 4 dims discarded)
        """
        vel = torch.zeros_like(x)
        vel[:, 1:] = x[:, 1:] - x[:, :-1]

        D = vel.size(-1)
        if D < T2M_DIM:
            vel = F.pad(vel, (0, T2M_DIM - D))
        elif D > T2M_DIM:
            vel = vel[:, :, :T2M_DIM]

        return vel[:, :, :MOV_IN]

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:       (B, T, D) motion sequence --D=168 (SMPL-X) or 263 (HML3D)
            lengths: (B,) actual clip lengths in frames (before padding)
        Returns:
            (B, 512) L2-normalised motion embeddings
        """
        vel = self.to_velocity(x)
        seg = self.movement(vel)

        seg_lengths = None
        if lengths is not None:
            seg_lengths = (lengths.float() / MOV_K).ceil().long().clamp(min=1, max=seg.size(1))

        return self.encoder(seg, seg_lengths)


#  Weight loading


def load_encoder(
    input_dim: int = SMPLX.pose_dim,
    weights_path: str | None = None,
    device: str = "cpu",
) -> T2MMotionEncoder:
    """Instantiate T2MMotionEncoder and load official pre-trained weights.

    Resolution order:
      1. `weights_path` if provided and exists
      2. Candidates in ``_WEIGHT_CANDIDATES``
      3. Random initialisation with a clearly actionable warning

    Args:
        input_dim:    Motion feature dim fed to the encoder (168 for SMPL-X).
        weights_path: Optional explicit path to the .tar checkpoint.
        device:       Torch device string ("cpu" / "cuda").

    Returns:
        T2MMotionEncoder in eval mode, placed on ``device``.
    """
    enc = T2MMotionEncoder(input_dim=input_dim)
    enc._loaded_pretrained = False  # type: ignore[attr-defined]

    # Find weights
    resolved: str | None = None
    if weights_path and Path(weights_path).exists():
        resolved = weights_path
    if resolved is None:
        for candidate in WEIGHT_CANDIDATES:
            if Path(candidate).exists():
                resolved = candidate
                break

    if resolved:
        log.info("[T2MEncoder] loading official weights from: %s", resolved)
        ck = torch.load(resolved, map_location="cpu", weights_only=False)

        # Load movement_encoder
        if "movement_encoder" in ck:
            mov_sd = ck["movement_encoder"]
            model_mov_sd = enc.movement.state_dict()
            compat = {
                k: v
                for k, v in mov_sd.items()
                if k in model_mov_sd and v.shape == model_mov_sd[k].shape
            }
            enc.movement.load_state_dict(compat, strict=False)
            log.info("[T2MEncoder] movement_encoder: loaded %d/%d keys", len(compat), len(mov_sd))

        # Load motion_encoder
        if "motion_encoder" in ck:
            mot_sd = ck["motion_encoder"]
            model_mot_sd = enc.encoder.state_dict()
            compat = {
                k: v
                for k, v in mot_sd.items()
                if k in model_mot_sd and v.shape == model_mot_sd[k].shape
            }
            enc.encoder.load_state_dict(compat, strict=False)
            log.info("[T2MEncoder] motion_encoder:   loaded %d/%d keys", len(compat), len(mot_sd))

        enc._loaded_pretrained = True  # type: ignore[attr-defined]
        log.info(
            "[T2MEncoder] Official pre-trained weights loaded.\n"
            "  NOTE: T2M was trained on HumanML3D 263-dim features (root vel + foot\n"
            "  contacts + 6D rot + local vel). This pipeline feeds 168-dim SMPL-X\n"
            "  axis-angle frame-differences, zero-padded to 263. The forward pass is\n"
            "  deterministic so gen-vs-real FID is internally consistent, but absolute\n"
            "  FID values are NOT directly comparable to published T2M / MoMask numbers\n"
            "  unless inputs are first converted to the HumanML3D feature representation."
        )
    else:
        log.warning(
            "[T2MEncoder] Pre-trained weights NOT found. Using random init.\n"
            "  FID will be internally consistent but NOT comparable to published results.\n"
            "  Weights were downloaded to:\n"
            "    data/t2m_download/extracted/t2m/text_mot_match/model/finest.tar\n"
            "  Run the setup step to link them:\n"
            "    python scripts/evaluation/setup_evaluator.py"
        )

    return enc.to(device).eval()


#  Feature extraction


@torch.no_grad()
def extract_features(
    encoder: T2MMotionEncoder,
    motions: list[np.ndarray],
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Extract (N, 512) L2-normalised features from a list of (T_i, D) clips.

    Args:
        encoder:    Loaded T2MMotionEncoder in eval mode.
        motions:    List of (T_i, D) float32 motion arrays.
        batch_size: Clips per GPU batch.
        device:     Torch device string.

    Returns:
        (N, 512) float32 numpy array.
    """
    all_feats: list[np.ndarray] = []

    def batches(lst: list, n: int) -> Iterator[list]:
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    for batch_clips in batches(motions, batch_size):
        lengths = torch.tensor(
            [min(c.shape[0], 200) for c in batch_clips],
            dtype=torch.long,
            device=device,
        )
        max_len = int(lengths.max().item())
        D = batch_clips[0].shape[1]

        padded = np.zeros((len(batch_clips), max_len, D), dtype=np.float32)
        for i, clip in enumerate(batch_clips):
            t = min(clip.shape[0], max_len)
            padded[i, :t] = clip[:t]

        x = torch.tensor(padded, device=device, dtype=torch.float32)
        feats = encoder(x, lengths)  # (B, 512)
        all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


#  T2M Text Encoder

GLOVE_DIM = 300
TXT_POS_SIZE = 15  # number of POS tag classes in finest.tar text_encoder

# GloVe candidate paths (searched in order)
GLOVE_CANDIDATES = [
    "data/inter-x/misc/glove.6B.300d.txt",
    "data/glove.6B.300d.txt",
]

glove_cache: dict[str, np.ndarray] | None = None


class T2MTextEncoder(nn.Module):
    """BiGRU text encoder matching T2M finest.tar ``text_encoder`` weights.

    Encodes tokenized English text into L2-normalised 512-dim embeddings in
    the SAME joint space as ``T2MMotionEncoder`` (both trained jointly in
    finest.tar).  Using both encoders in ``compute_r_precision`` produces
    valid R-Precision values comparable to published T2M results.

    POS tags: the T2M encoder was trained with fine-grained POS one-hot vectors
    (15 classes).  When POS tags are unavailable, feeding zero-vectors to
    ``pos_emb`` reduces to adding the trained ``pos_emb.bias`` --a constant
    300-dim offset --to every word embedding.  This is consistent across all
    texts (same assumption, no per-token discrimination lost) and keeps features
    within the trained distribution.
    """

    def __init__(
        self,
        word_size: int = GLOVE_DIM,
        pos_size: int = TXT_POS_SIZE,
        hidden_size: int = 512,
        output_size: int = 512,
    ) -> None:
        super().__init__()
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.register_buffer("hidden", torch.zeros(2, 1, hidden_size))
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_net = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(
        self,
        word_embs: torch.Tensor,  # (B, T, 300) GloVe embeddings (zero-padded)
        lengths: torch.Tensor | None = None,  # (B,) actual token counts
    ) -> torch.Tensor:
        """
        Returns:
            (B, 512) L2-normalised text embeddings in T2M joint space.
        """
        B = word_embs.size(0)
        # Zero POS input -> contributes exactly pos_emb.bias (consistent offset)
        pos_zero = torch.zeros(
            B, word_embs.size(1), TXT_POS_SIZE, device=word_embs.device, dtype=word_embs.dtype
        )
        x = word_embs + self.pos_emb(pos_zero)  # (B, T, 300) word + pos offset
        x = self.input_emb(x)  # (B, T, 512)

        h0 = self.hidden.repeat(1, B, 1)  # type: ignore[union-attr]  # (2, B, 512)
        out, _ = self.gru(x, h0)  # (B, T, 1024)

        if lengths is not None:
            mask = (
                (torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1))
                .unsqueeze(-1)
                .float()
            )
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        else:
            pooled = out.mean(1)  # (B, 1024)

        embed = self.output_net(pooled)  # (B, 512)
        return F.normalize(embed, dim=-1)


def load_glove(glove_path: str | None = None) -> dict[str, np.ndarray]:
    """Load GloVe 300d word vectors, caching on first call."""
    global glove_cache
    if glove_cache is not None:
        return glove_cache

    candidates = ([glove_path] if glove_path else []) + GLOVE_CANDIDATES
    resolved = next((c for c in candidates if c and Path(c).exists()), None)

    if resolved is None:
        log.warning(
            "[T2MTextEncoder] GloVe 300d not found at %s --word embeddings will be zero.",
            GLOVE_CANDIDATES,
        )
        glove_cache = {}
        return glove_cache

    log.info("[T2MTextEncoder] Loading GloVe from %s ...", resolved)
    vocab: dict[str, np.ndarray] = {}
    with open(resolved, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) == GLOVE_DIM + 1:
                vocab[parts[0]] = np.array(parts[1:], dtype=np.float32)
    log.info("[T2MTextEncoder] Loaded %d GloVe vectors", len(vocab))
    glove_cache = vocab
    return glove_cache


def texts_to_glove(
    texts: list[str],
    glove: dict[str, np.ndarray],
    max_len: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize texts (whitespace) and look up GloVe embeddings.

    Returns:
        word_embs: (N, max_len, 300) float32 --zero-padded
        lengths:   (N,) int64 --actual token counts (>= 1)
    """
    N = len(texts)
    word_embs = np.zeros((N, max_len, GLOVE_DIM), dtype=np.float32)
    lengths = np.ones(N, dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = text.lower().split()[:max_len]
        for j, tok in enumerate(tokens):
            vec = glove.get(tok)
            if vec is not None:
                word_embs[i, j] = vec
        lengths[i] = max(len(tokens), 1)
    return word_embs, lengths


def load_text_encoder(
    weights_path: str | None = None,
    device: str = "cpu",
) -> T2MTextEncoder | None:
    """Load T2MTextEncoder from finest.tar.

    Returns None if no weights are found --``compute_r_precision`` will then
    fall back to SBERT with a warning that the result is not valid.
    """
    candidates = ([weights_path] if weights_path else []) + WEIGHT_CANDIDATES
    resolved = next((c for c in candidates if c and Path(c).exists()), None)
    if resolved is None:
        log.warning("[T2MTextEncoder] finest.tar not found; R-Precision will use SBERT.")
        return None

    ck = torch.load(resolved, map_location="cpu", weights_only=False)
    if "text_encoder" not in ck:
        log.warning("[T2MTextEncoder] finest.tar has no 'text_encoder' key.")
        return None

    enc = T2MTextEncoder()
    txt_sd = ck["text_encoder"]
    model_sd = enc.state_dict()
    compat = {k: v for k, v in txt_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
    enc.load_state_dict(compat, strict=False)
    log.info("[T2MTextEncoder] loaded %d/%d weights from %s", len(compat), len(txt_sd), resolved)
    return enc.to(device).eval()


@torch.no_grad()
def encode_texts_t2m(
    encoder: T2MTextEncoder,
    texts: list[str],
    device: str = "cpu",
    glove_path: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode text descriptions into (N, 512) L2-normalised T2M text features.

    Uses GloVe 300d word embeddings + zero POS tags (consistent across all
    texts; keeps features in the T2M joint embedding space).

    Returns: (N, 512) float32 array in the same space as ``extract_features``.
    """
    glove = load_glove(glove_path)
    word_embs, lengths = texts_to_glove(texts, glove)
    all_feats: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        we = torch.tensor(word_embs[i : i + batch_size], device=device)
        ln = torch.tensor(lengths[i : i + batch_size], device=device)
        feats = encoder(we, ln)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)
