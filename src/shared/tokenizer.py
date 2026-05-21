import numpy as np

from src.shared.constants import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID


def tokenize(
    text: str,
    vocab: dict[str, int],
    max_len: int = 64,
) -> np.ndarray:
    bos = vocab.get("<BOS>", BOS_TOKEN_ID)
    eos = vocab.get("<EOS>", EOS_TOKEN_ID)
    unk = vocab.get("<UNK>", UNK_TOKEN_ID)
    tokens = [bos] + [vocab.get(w, unk) for w in text.lower().split()[: max_len - 2]] + [eos]
    tokens += [PAD_TOKEN_ID] * (max_len - len(tokens))

    return np.array(tokens[:max_len], dtype=np.int64)


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab: dict[str, int] = {
        "<PAD>": PAD_TOKEN_ID,
        "<UNK>": UNK_TOKEN_ID,
        "<BOS>": BOS_TOKEN_ID,
        "<EOS>": EOS_TOKEN_ID,
    }

    for t in texts:
        for w in t.lower().split():
            vocab.setdefault(w, len(vocab))

    return vocab
