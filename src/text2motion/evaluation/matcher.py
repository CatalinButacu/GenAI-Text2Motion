from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from os.path import join as pjoin
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from text2motion.motion.contracts import Split
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.representation import DIM, SLICES

_POS_INDEX = {
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

_VIP_WORDS = {
    "Loc_VIP": (
        "left", "right", "clockwise", "counterclockwise", "anticlockwise", "forward",
        "back", "backward", "up", "down", "straight", "curve",
    ),
    "Body_VIP": (
        "arm", "chin", "foot", "feet", "face", "hand", "mouth", "leg", "waist",
        "eye", "knee", "shoulder", "thigh",
    ),
    "Obj_VIP": (
        "stair", "dumbbell", "chair", "window", "floor", "car", "ball", "handrail",
        "baseball", "basketball",
    ),
    "Act_VIP": (
        "walk", "run", "swing", "pick", "bring", "kick", "put", "squat", "throw",
        "hop", "dance", "jump", "turn", "stumble", "stop", "sit", "lift", "lower",
        "raise", "wash", "stand", "kneel", "stroll", "rub", "bend", "balance", "flap",
        "jog", "shuffle", "lean", "rotate", "spin", "spread", "climb",
    ),
    "Desc_VIP": (
        "slowly", "carefully", "fast", "careful", "slow", "quickly", "happy", "angry",
        "sad", "happily", "angrily", "sadly",
    ),
}


class GuoWordPosVectorizer:
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, "%s_data.npy" % prefix))
        words = pickle.load(open(pjoin(meta_root, "%s_words.pkl" % prefix), "rb"))
        word2idx = pickle.load(open(pjoin(meta_root, "%s_idx.pkl" % prefix), "rb"))
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def _pos_one_hot(self, pos):
        pos_vec = np.zeros(len(_POS_INDEX))
        if pos in _POS_INDEX:
            pos_vec[_POS_INDEX[pos]] = 1
        else:
            pos_vec[_POS_INDEX["OTHER"]] = 1
        return pos_vec

    def __len__(self):
        return len(self.word2vec)

    def __getitem__(self, item):
        word, pos = item.split("/")
        if word in self.word2vec:
            word_vec = self.word2vec[word]
            vip_pos = None
            for key, values in _VIP_WORDS.items():
                if word in values:
                    vip_pos = key
                    break
            if vip_pos is not None:
                pos_vec = self._pos_one_hot(vip_pos)
            else:
                pos_vec = self._pos_one_hot(pos)
        else:
            word_vec = self.word2vec["unk"]
            pos_vec = self._pos_one_hot("OTHER")
        return word_vec, pos_vec


_MOVEMENT_INPUT = SLICES["vel"].stop
_MOVEMENT_WIDTH = 512
_MOVEMENT_KERNEL = 4
_GRU_INPUT = 512
_GRU_WIDTH = 1024
_EMBEDDING_DIM = 512
_GLOVE_DIM = 300
_POS_DIM = 15


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


class GuoMovementEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                _MOVEMENT_INPUT,
                _MOVEMENT_WIDTH,
                kernel_size=_MOVEMENT_KERNEL,
                stride=_MOVEMENT_KERNEL // 2,
                padding=1,
            ),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                _MOVEMENT_WIDTH,
                _MOVEMENT_WIDTH,
                kernel_size=_MOVEMENT_KERNEL,
                stride=_MOVEMENT_KERNEL // 2,
                padding=1,
            ),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(_MOVEMENT_WIDTH, _GRU_INPUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.out_net(out)


class GuoMotionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_emb = nn.Linear(_GRU_INPUT, _GRU_WIDTH)
        self.gru = nn.GRU(
            _GRU_WIDTH, _GRU_WIDTH, num_layers=1, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(_GRU_WIDTH * 2, _GRU_WIDTH),
            nn.LayerNorm(_GRU_WIDTH),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(_GRU_WIDTH, _EMBEDDING_DIM),
        )
        self.register_buffer("hidden", torch.zeros(2, 1, _GRU_WIDTH))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        final = _bigru_final_state(self.gru, self.input_emb(x), self.hidden, lengths)
        return self.output_net(final)


class GuoMotionEmbeddingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.movement = GuoMovementEncoder()
        self.encoder = GuoMotionEncoder()

    def forward(self, feat263: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if feat263.size(-1) != DIM:
            raise ValueError(
                f"expected last dim {DIM} (standard HumanML3D), got {feat263.size(-1)}"
            )

        seg = self.movement(feat263[..., :_MOVEMENT_INPUT])
        seg_lengths = None

        if lengths is not None:
            seg_lengths = (lengths // _MOVEMENT_KERNEL).clamp(min=1, max=seg.size(1))

        return self.encoder(seg, seg_lengths)


class GuoTextEmbeddingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_emb = nn.Linear(_POS_DIM, _GLOVE_DIM)
        self.input_emb = nn.Linear(_GLOVE_DIM, _EMBEDDING_DIM)
        self.gru = nn.GRU(
            _EMBEDDING_DIM,
            _EMBEDDING_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_net = nn.Sequential(
            nn.Linear(_EMBEDDING_DIM * 2, _EMBEDDING_DIM),
            nn.LayerNorm(_EMBEDDING_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(_EMBEDDING_DIM, _EMBEDDING_DIM),
        )
        self.register_buffer("hidden", torch.zeros(2, 1, _EMBEDDING_DIM))

    def forward(
        self,
        word_embs: torch.Tensor,
        pos_onehots: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pos_onehots is None:
            pos_onehots = torch.zeros(*word_embs.shape[:2], _POS_DIM, device=word_embs.device)
        x = self.input_emb(word_embs + self.pos_emb(pos_onehots))
        final = _bigru_final_state(self.gru, x, self.hidden, lengths)
        return self.output_net(final)


def load_guo_motion_normalization_stats(stats_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    mean = np.load(Path(stats_dir) / "mean.npy")
    std = np.load(Path(stats_dir) / "std.npy")

    if mean.shape[-1] != DIM:
        raise ValueError(
            f"eval stats must be {DIM}-dim (standard HumanML3D), got {mean.shape}"
        )

    return mean.astype(np.float32), std.astype(np.float32)


def load_frozen_guo_embedding_encoders(finest_tar: Path, device: str = "cpu") -> tuple[GuoMotionEmbeddingEncoder, GuoTextEmbeddingEncoder]:
    ckpt = torch.load(Path(finest_tar), map_location="cpu", weights_only=False)

    motion = GuoMotionEmbeddingEncoder()
    _load_exact(motion.movement, ckpt["movement_encoder"], "movement_encoder")
    _load_exact(motion.encoder, ckpt["motion_encoder"], "motion_encoder")

    text = GuoTextEmbeddingEncoder()
    _load_exact(text, ckpt["text_encoder"], "text_encoder")

    return motion.to(device).eval(), text.to(device).eval()


def _load_exact(module: nn.Module, state: dict, name: str) -> None:
    missing, unexpected = module.load_state_dict(state, strict=False)

    if missing:
        raise RuntimeError(f"{name}: {len(missing)} params got no pretrained weight: {missing[:6]}")

    if unexpected:
        raise RuntimeError(f"{name}: {len(unexpected)} checkpoint keys unused: {unexpected[:6]}")


def create_guo_text_input_builder(word_vectorizer: Any, max_tokens: int = 20) -> Callable:
    def build_text(tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        items = ["sos/OTHER"] + tokens[:max_tokens] + ["eos/OTHER"]
        word_embeddings = np.stack([word_vectorizer[item][0] for item in items]).astype(np.float32)
        pos_onehots = np.stack([word_vectorizer[item][1] for item in items]).astype(np.float32)
        return word_embeddings, pos_onehots

    return build_text


@dataclass(frozen=True)
class GuoEvaluationResources:
    device: str
    motion_dataset_dir: Path
    annotation_dir: Path
    dataset_motion_scaler: MotionScaler
    guo_motion_mean: np.ndarray
    guo_motion_std: np.ndarray
    motion_embedder: Any
    text_embedder: Any
    build_guo_text_inputs: Callable
    tokenizer_downsample_factor: int

    @classmethod
    def from_frozen_guo_assets(
        cls,
        motion_dataset_dir: Path,
        eval_stats_dir: Path,
        eval_matcher: Path,
        vocab_dir: Path,
        tokenizer_downsample_factor: int,
        annotation_dir: Path | None = None,
        device: str = "cpu",
    ) -> GuoEvaluationResources:
        motion_dataset_dir = Path(motion_dataset_dir)
        guo_motion_mean, guo_motion_std = load_guo_motion_normalization_stats(eval_stats_dir)
        motion_embedder, text_embedder = load_frozen_guo_embedding_encoders(
            eval_matcher, device=device
        )

        return cls(
            device=device,
            motion_dataset_dir=motion_dataset_dir,
            annotation_dir=(
                Path(annotation_dir) if annotation_dir else motion_dataset_dir / "texts"
            ),
            dataset_motion_scaler=MotionScaler.load(motion_dataset_dir, dim=DIM),
            guo_motion_mean=guo_motion_mean,
            guo_motion_std=guo_motion_std,
            motion_embedder=motion_embedder,
            text_embedder=text_embedder,
            build_guo_text_inputs=create_guo_text_input_builder(
                GuoWordPosVectorizer(str(vocab_dir), "our_vab")
            ),
            tokenizer_downsample_factor=tokenizer_downsample_factor,
        )

    def clip_ids(self, split: Split | str) -> list[str]:
        raw = (self.motion_dataset_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines()
        present = [name.strip() for name in raw if name.strip()]
        return list(dict.fromkeys(i[1:] if i.startswith("M") else i for i in present))

    def normalize(self, feats):
        return self.dataset_motion_scaler.normalize(feats)

    def denormalize(self, feats):
        return self.dataset_motion_scaler.denormalize(feats)
