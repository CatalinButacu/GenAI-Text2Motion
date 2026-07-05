import pickle
from pathlib import Path

import numpy as np

GLOVE_DIM = 300

POS_ENUMERATOR = {
    "VERB": 0, "NOUN": 1, "DET": 2, "ADP": 3, "NUM": 4, "AUX": 5, "PRON": 6, "ADJ": 7,
    "ADV": 8, "Loc_VIP": 9, "Body_VIP": 10, "Obj_VIP": 11, "Act_VIP": 12, "Desc_VIP": 13,
    "OTHER": 14,
}  # fmt: skip
VIP_DICT = {
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
NUM_POS = len(POS_ENUMERATOR)


def tokens_to_words(token_list: list[str]) -> list[str]:
    words = []
    for token in token_list:
        token = token.strip()
        if not token:
            continue
        words.append(token.rsplit("/", 1)[0].lower())
    return words


def load_glove(glove_path: Path, vocab: set[str] | None = None) -> dict[str, np.ndarray]:
    vectors: dict[str, np.ndarray] = {}
    with open(glove_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if vocab is not None and word not in vocab:
                continue
            vectors[word] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def words_to_embeddings(words: list[str], glove: dict[str, np.ndarray]) -> np.ndarray:
    zero = np.zeros(GLOVE_DIM, dtype=np.float32)
    return np.stack([glove.get(word, zero) for word in words], axis=0)


def load_special_vectors(vab_data_path: Path, vab_idx_path: Path) -> dict[str, np.ndarray]:
    vectors = np.load(vab_data_path)
    word2idx = pickle.loads(Path(vab_idx_path).read_bytes())
    return {token: vectors[word2idx[token]].astype(np.float32) for token in ("sos", "eos", "unk")}


def pos_one_hot(word: str, pos_tag: str) -> np.ndarray:
    index = None
    for vip_category, words in VIP_DICT.items():
        if word in words:
            index = POS_ENUMERATOR[vip_category]
            break
    if index is None:
        index = POS_ENUMERATOR.get(pos_tag, POS_ENUMERATOR["OTHER"])
    vec = np.zeros(NUM_POS, dtype=np.float32)
    vec[index] = 1.0
    return vec


def vectorize_caption(
    pos_tagged_tokens: list[str],
    glove: dict[str, np.ndarray],
    special: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    word_embs = [special["sos"]]
    pos_onehots = [pos_one_hot("sos", "OTHER")]
    for token in pos_tagged_tokens:
        word, _, tag = token.partition("/")
        word = word.lower()
        word_embs.append(glove.get(word, special["unk"]))
        pos_onehots.append(pos_one_hot(word, tag))
    word_embs.append(special["eos"])
    pos_onehots.append(pos_one_hot("eos", "OTHER"))
    return np.stack(word_embs, axis=0), np.stack(pos_onehots, axis=0)
