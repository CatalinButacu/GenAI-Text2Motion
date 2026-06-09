"""GloVe word embeddings for the Guo matcher text branch (R-precision / matching score).

The matcher's ``TextMatcher`` consumes per-word GloVe-300 vectors (POS fed as zeros, see
``matcher.py``). Captions are tokenized via the official ``tokens`` field stored in
``texts/<id>.txt`` ('word/POS ...'), so the words already match the evaluator's preprocessing.
Reference: Guo et al., CVPR 2022 (github.com/EricGuo5513/text-to-motion).
"""

import pickle
from pathlib import Path

import numpy as np

GLOVE_DIM = 300

# POS one-hot scheme + VIP word lists, copied EXACTLY from Guo et al.'s WordVectorizer
# (github.com/EricGuo5513/text-to-motion, utils/word_vectorizer.py). The matcher's text encoder was
# trained with these, so reproducing them is required for comparable R-precision.
POS_ENUMERATOR = {
    "VERB": 0, "NOUN": 1, "DET": 2, "ADP": 3, "NUM": 4, "AUX": 5, "PRON": 6, "ADJ": 7,
    "ADV": 8, "Loc_VIP": 9, "Body_VIP": 10, "Obj_VIP": 11, "Act_VIP": 12, "Desc_VIP": 13,
    "OTHER": 14,
}  # fmt: skip
VIP_DICT = {
    "Loc_VIP": (
        "left", "right", "clockwise", "counterclockwise", "anticlockwise", "forward", "back",
        "backward", "up", "down", "straight", "curve",
    ),
    "Body_VIP": (
        "arm", "chin", "foot", "feet", "face", "hand", "mouth", "leg", "waist", "eye", "knee",
        "shoulder", "thigh",
    ),
    "Obj_VIP": (
        "stair", "dumbbell", "chair", "window", "floor", "car", "ball", "handrail", "baseball",
        "basketball",
    ),
    "Act_VIP": (
        "walk", "run", "swing", "pick", "bring", "kick", "put", "squat", "throw", "hop", "dance",
        "jump", "turn", "stumble", "stop", "sit", "lift", "lower", "raise", "wash", "stand", "kneel",
        "stroll", "rub", "bend", "balance", "flap", "jog", "shuffle", "lean", "rotate", "spin",
        "spread", "climb",
    ),
    "Desc_VIP": (
        "slowly", "carefully", "fast", "careful", "slow", "quickly", "happy", "angry", "sad",
        "happily", "angrily", "sadly",
    ),
}
NUM_POS = len(POS_ENUMERATOR)


def tokens_to_words(token_list: list[str]) -> list[str]:
    """['a/DET', 'man/NOUN', ...] -> ['a', 'man', ...] (word part, lowercased)."""
    words = []
    for token in token_list:
        token = token.strip()
        if not token:
            continue
        words.append(token.rsplit("/", 1)[0].lower())
    return words


def load_glove(glove_path: Path, vocab: set[str] | None = None) -> dict[str, np.ndarray]:
    """word -> (300,) float32. When ``vocab`` is given, load only those words (much faster)."""
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
    """['a', 'man', ...] -> (L, 300). Unknown words map to the zero vector."""
    zero = np.zeros(GLOVE_DIM, dtype=np.float32)
    return np.stack([glove.get(word, zero) for word in words], axis=0)


def load_special_vectors(vab_data_path: Path, vab_idx_path: Path) -> dict[str, np.ndarray]:
    """sos/eos/unk vectors from a Guo vocab file (``*_data.npy`` + ``*_idx.pkl``)."""
    vectors = np.load(vab_data_path)
    word2idx = pickle.loads(Path(vab_idx_path).read_bytes())
    return {token: vectors[word2idx[token]].astype(np.float32) for token in ("sos", "eos", "unk")}


def pos_one_hot(word: str, pos_tag: str) -> np.ndarray:
    """(15,) POS one-hot. A VIP-list word takes its VIP category; otherwise the spaCy POS tag;
    otherwise OTHER. Exactly Guo's WordVectorizer.__getitem__ logic."""
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
    """A caption's 'word/POS' tokens -> (word_embs (L+2, 300), pos_onehots (L+2, 15)) with the
    sos/eos boundary tokens Guo's dataset prepends/appends. OOV words use the 'unk' vector."""
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
