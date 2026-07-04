"""Guo et al. text+motion evaluator (the fixed FID / R-precision matcher) -- reused UNMODIFIED.

Architecture is a faithful replica of the encoders in `finest.tar` from
    Guo et al. "Generating Diverse and Natural 3D Human Motions from Text", CVPR 2022,
    github.com/EricGuo5513/text-to-motion
so the pretrained weights load exactly.

STANDARD-263 input path (this is the citable one; NOT the donor's 168->259 diff-hack):
  raw 263 HumanML3D feature -> normalize with the evaluator's own 263 mean/std
  -> drop the last 4 foot-contact dims (feed [..., :259]) -> MovementEncoder -> MotionEncoder.
The movement encoder's first conv is (512, 259, 4) in finest.tar, confirming the 259 input.

All metrics live in the matcher's joint embedding space; never retrain it (ADR 0001).
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

MOV_IN = 259  # 263 feature minus 4 foot-contact dims
MOV_HID = 512
MOV_K = 4
GRU_IN = 512
GRU_HID = 1024
OUT_DIM = 512
FEATURE_DIM = 263
GLOVE_DIM = 300
POS_SIZE = 15


def _bigru_final_state(
    gru: nn.GRU, x: torch.Tensor, hidden: torch.Tensor, lengths: torch.Tensor | None
) -> torch.Tensor:
    """Run a 1-layer BiGRU and return the concatenated forward+backward FINAL hidden state
    (B, 2*hidden) -- Guo et al.'s BiGRUCo pooling (NOT mean-pooling). Lengths, when given, are honoured
    via pack_padded_sequence so padding frames never enter the final state."""
    h0 = hidden.repeat(1, x.size(0), 1)

    if lengths is None:
        _, h_n = gru(x, h0)
    else:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, h_n = gru(packed, h0)

    return torch.cat([h_n[0], h_n[1]], dim=-1)


class MovementEncoder(nn.Module):
    """(B, T, 259) -> (B, T//4, 512) stride-4 conv segments."""

    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(MOV_IN, MOV_HID, kernel_size=MOV_K, stride=MOV_K // 2, padding=1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(MOV_HID, MOV_HID, kernel_size=MOV_K, stride=MOV_K // 2, padding=1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(MOV_HID, GRU_IN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.out_net(out)


class MotionEncoder(nn.Module):
    """(B, T', 512) movement segments -> (B, 512) L2-normalised clip embedding."""

    def __init__(self) -> None:
        super().__init__()
        self.input_emb = nn.Linear(GRU_IN, GRU_HID)
        self.gru = nn.GRU(GRU_HID, GRU_HID, num_layers=1, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(GRU_HID * 2, GRU_HID),
            nn.LayerNorm(GRU_HID),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(GRU_HID, OUT_DIM),
        )
        self.register_buffer("hidden", torch.zeros(2, 1, GRU_HID))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        final = _bigru_final_state(self.gru, self.input_emb(x), self.hidden, lengths)
        return self.output_net(final)


class MotionMatcher(nn.Module):
    """Full motion feature extractor: standard 263 feature -> (B, 512) embedding.

    Expects an ALREADY-NORMALISED 263 feature (normalise with `load_eval_stats`). Drops the last
    4 foot-contact dims internally -- no velocity diff, no padding (that was the 168-track hack).
    """

    def __init__(self) -> None:
        super().__init__()
        self.movement = MovementEncoder()
        self.encoder = MotionEncoder()

    def forward(self, feat263: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if feat263.size(-1) != FEATURE_DIM:
            raise ValueError(
                f"expected last dim {FEATURE_DIM} (standard HumanML3D), got {feat263.size(-1)}"
            )

        seg = self.movement(feat263[..., :MOV_IN])
        seg_lengths = None

        if lengths is not None:
            seg_lengths = (lengths // MOV_K).clamp(min=1, max=seg.size(1))

        return self.encoder(seg, seg_lengths)


class TextMatcher(nn.Module):
    """BiGRU text encoder matching finest.tar `text_encoder`; maps text -> (B, 512) in the same
    joint space as MotionMatcher. POS is fed as zeros (consistent constant offset; see Guo et al.)."""

    def __init__(self) -> None:
        super().__init__()
        self.pos_emb = nn.Linear(POS_SIZE, GLOVE_DIM)
        self.input_emb = nn.Linear(GLOVE_DIM, 512)
        self.gru = nn.GRU(512, 512, num_layers=1, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
        )
        self.register_buffer("hidden", torch.zeros(2, 1, 512))

    def forward(
        self,
        word_embs: torch.Tensor,
        pos_onehots: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pos_onehots is None:
            pos_onehots = torch.zeros(*word_embs.shape[:2], POS_SIZE, device=word_embs.device)
        x = self.input_emb(word_embs + self.pos_emb(pos_onehots))
        final = _bigru_final_state(self.gru, x, self.hidden, lengths)
        return self.output_net(final)


def load_eval_stats(stats_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the canonical 263 eval mean/std (T2M Comp_v6 meta). Returns (mean, std), shape (263,)."""
    mean = np.load(Path(stats_dir) / "mean.npy")
    std = np.load(Path(stats_dir) / "std.npy")

    if mean.shape[-1] != FEATURE_DIM:
        raise ValueError(
            f"eval stats must be {FEATURE_DIM}-dim (standard HumanML3D), got {mean.shape}"
        )

    return mean.astype(np.float32), std.astype(np.float32)


def load_matchers(finest_tar: Path, device: str = "cpu") -> tuple[MotionMatcher, TextMatcher]:
    """Load motion + text matchers from finest.tar (fail loud if missing or shapes mismatch)."""
    ckpt = torch.load(Path(finest_tar), map_location="cpu", weights_only=False)

    motion = MotionMatcher()
    _load_exact(motion.movement, ckpt["movement_encoder"], "movement_encoder")
    _load_exact(motion.encoder, ckpt["motion_encoder"], "motion_encoder")

    text = TextMatcher()
    _load_exact(text, ckpt["text_encoder"], "text_encoder")

    return motion.to(device).eval(), text.to(device).eval()


def _load_exact(module: nn.Module, state: dict, name: str) -> None:
    """Load weights and FAIL LOUD if any model parameter went unfilled (no silent partial loads)."""
    missing, unexpected = module.load_state_dict(state, strict=False)

    if missing:
        raise RuntimeError(f"{name}: {len(missing)} params got no pretrained weight: {missing[:6]}")

    if unexpected:
        raise RuntimeError(f"{name}: {len(unexpected)} checkpoint keys unused: {unexpected[:6]}")
