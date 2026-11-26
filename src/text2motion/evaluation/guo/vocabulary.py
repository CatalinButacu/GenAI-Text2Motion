from __future__ import annotations

import pickle
from collections.abc import Callable
from os.path import join as pjoin
from types import MappingProxyType
from typing import Any

import numpy as np

_POS_INDEX = MappingProxyType(
    {
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
)


_VIP_WORDS = MappingProxyType(
    {
        "Loc_VIP": (
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
        ),
        "Body_VIP": (
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
        ),
        "Obj_VIP": (
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
        ),
        "Act_VIP": (
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
        ),
        "Desc_VIP": (
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
        ),
    }
)


############################################################
#       GuoWordPosVectorizer
############################################################
class GuoWordPosVectorizer:
    def __init__(self, meta_root, prefix):
        vectors = np.load(pjoin(meta_root, "%s_data.npy" % prefix))
        with open(pjoin(meta_root, "%s_words.pkl" % prefix), "rb") as words_file:
            words = pickle.load(words_file)
        with open(pjoin(meta_root, "%s_idx.pkl" % prefix), "rb") as index_file:
            word2idx = pickle.load(index_file)
        self.word2vec = {w: vectors[word2idx[w]] for w in words}

    def pos_one_hot(self, pos):
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
            pos_vec = self.pos_one_hot(vip_pos if vip_pos is not None else pos)
        else:
            word_vec = self.word2vec["unk"]
            pos_vec = self.pos_one_hot("OTHER")
        return word_vec, pos_vec


def create_guo_text_input_builder(word_vectorizer: Any, max_tokens: int = 20) -> Callable:
    def build_text(tokens: list[str]) -> tuple[np.ndarray, np.ndarray]:
        items = ["sos/OTHER"] + tokens[:max_tokens] + ["eos/OTHER"]
        word_embeddings = np.stack([word_vectorizer[item][0] for item in items]).astype(np.float32)
        pos_onehots = np.stack([word_vectorizer[item][1] for item in items]).astype(np.float32)
        return word_embeddings, pos_onehots

    return build_text
