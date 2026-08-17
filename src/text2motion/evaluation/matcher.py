from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from os.path import join as pjoin
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from text2motion.motion.dataset import MotionScaler, Split
from text2motion.motion.representation import DIM


class LengthMode(StrEnum):
    FIXED = "fixed"
    END_TOKEN = "end"


POS_enumerator = {
    "VERB": 0,
    "NOUN": 1,
    "DET": 2,
    "ADP": 3,
    "NUM": 4,
    "AUX": 5,
    "PRON": 6,
    "ADJ": 7,
    "ADV": 8,
    "Loc_VIP": 9,
    "Body_VIP": 10,
    "Obj_VIP": 11,
    "Act_VIP": 12,
    "Desc_VIP": 13,
    "OTHER": 14,
}

Loc_list = (
    "left",
    "right",
    "clockwise",
    "counterclockwise",
    "anticlockwise",
    "forward",
    "back",
    "backward",
    "up",
    "down",
    "straight",
    "curve",
)

Body_list = (
    "arm",
    "chin",
    "foot",
    "feet",
    "face",
    "hand",
    "mouth",
    "leg",
    "waist",
    "eye",
    "knee",
    "shoulder",
    "thigh",
)

Obj_List = (
    "stair",
    "dumbbell",
    "chair",
    "window",
    "floor",
    "car",
    "ball",
    "handrail",
    "baseball",
    "basketball",
)

Act_list = (
    "walk",
    "run",
    "swing",
    "pick",
    "bring",
    "kick",
    "put",
    "squat",
    "throw",
    "hop",
    "dance",
    "jump",
    "turn",
    "stumble",
    "dance",
    "stop",
    "sit",
    "lift",
    "lower",
    "raise",
    "wash",
    "stand",
    "kneel",
    "stroll",
    "rub",
    "bend",
    "balance",
    "flap",
    "jog",
    "shuffle",
    "lean",
    "rotate",
    "spin",
    "spread",
    "climb",
)

Desc_list = (
    "slowly",
    "carefully",
    "fast",
    "careful",
    "slow",
    "quickly",
    "happy",
    "angry",
    "sad",
    "happily",
    "angrily",
    "sadly",
)

VIP_dict = {
    "Loc_VIP": Loc_list,
    "Body_VIP": Body_list,
    "Obj_VIP": Obj_List,
    "Act_VIP": Act_list,
    "Desc_VIP": Desc_list,
}


class WordVectorizer:
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, "%s_data.npy" % prefix))
        words = pickle.load(open(pjoin(meta_root, "%s_words.pkl" % prefix), "rb"))
        word2idx = pickle.load(open(pjoin(meta_root, "%s_idx.pkl" % prefix), "rb"))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _get_pos_ohot(self, pos):
        pos_vec = np.zeros(len(POS_enumerator))
        if pos in POS_enumerator:
            pos_vec[POS_enumerator[pos]] = 1
        else:
            pos_vec[POS_enumerator["OTHER"]] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split("/")
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in VIP_dict.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._get_pos_ohot(vip_pos)
            else:
                pos_vec = self._get_pos_ohot(pos)
        else:
            word_vec = self.word2vec["unk"]
            pos_vec = self._get_pos_ohot("OTHER")
        return word_vec, pos_vec


MOV_IN = 259
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
    mean = np.load(Path(stats_dir) / "mean.npy")
    std = np.load(Path(stats_dir) / "std.npy")

    if mean.shape[-1] != FEATURE_DIM:
        raise ValueError(
            f"eval stats must be {FEATURE_DIM}-dim (standard HumanML3D), got {mean.shape}"
        )

    return mean.astype(np.float32), std.astype(np.float32)


def load_matchers(finest_tar: Path, device: str = "cpu") -> tuple[MotionMatcher, TextMatcher]:
    ckpt = torch.load(Path(finest_tar), map_location="cpu", weights_only=False)

    motion = MotionMatcher()
    _load_exact(motion.movement, ckpt["movement_encoder"], "movement_encoder")
    _load_exact(motion.encoder, ckpt["motion_encoder"], "motion_encoder")

    text = TextMatcher()
    _load_exact(text, ckpt["text_encoder"], "text_encoder")

    return motion.to(device).eval(), text.to(device).eval()


def _load_exact(module: nn.Module, state: dict, name: str) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False)

    if missing:
        raise RuntimeError(f"{name}: {len(missing)} params got no pretrained weight: {missing[:6]}")

    if unexpected:
        raise RuntimeError(f"{name}: {len(unexpected)} checkpoint keys unused: {unexpected[:6]}")


def make_build_text(word_vectorizer: Any, max_tokens: int = 20) -> Callable:
    def build_text(tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        items = ["sos/OTHER"] + tokens[:max_tokens] + ["eos/OTHER"]
        word_embeddings = np.stack([word_vectorizer[item][0] for item in items]).astype(np.float32)
        pos_onehots = np.stack([word_vectorizer[item][1] for item in items]).astype(np.float32)
        return word_embeddings, pos_onehots

    return build_text


@dataclass(frozen=True)
class EvaluationContext:
    device: str
    out_dir: Path
    text_dir: Path
    scaler: MotionScaler
    eval_mean: np.ndarray
    eval_std: np.ndarray
    motion_matcher: Any
    text_matcher: Any
    build_text: Callable
    downsample: int

    @classmethod
    def load(
        cls,
        out_dir: Path,
        eval_stats_dir: Path,
        eval_matcher: Path,
        vocab_dir: Path,
        downsample: int,
        text_dir: Path | None = None,
        device: str = "cpu",
    ) -> EvaluationContext:
        out_dir = Path(out_dir)
        eval_mean, eval_std = load_eval_stats(eval_stats_dir)
        motion_matcher, text_matcher = load_matchers(eval_matcher, device=device)

        return cls(
            device=device,
            out_dir=out_dir,
            text_dir=Path(text_dir) if text_dir else out_dir / "texts",
            scaler=MotionScaler.load(out_dir, dim=DIM),
            eval_mean=eval_mean,
            eval_std=eval_std,
            motion_matcher=motion_matcher,
            text_matcher=text_matcher,
            build_text=make_build_text(WordVectorizer(str(vocab_dir), "our_vab")),
            downsample=downsample,
        )

    def clip_ids(self, split: Split | str) -> list[str]:
        raw = (self.out_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        present = [name.strip() for name in raw if name.strip()]
        return list(dict.fromkeys(i[1:] if i.startswith("M") else i for i in present))

    @property
    def our_mean(self) -> np.ndarray:
        return self.scaler.mean

    @property
    def our_std(self) -> np.ndarray:
        return self.scaler.std

    def normalize(self, feats):
        return self.scaler.normalize(feats)

    def denormalize(self, feats):
        return self.scaler.denormalize(feats)
