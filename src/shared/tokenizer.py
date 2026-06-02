import numpy as np

from src.shared.constants import CONSTS


def tokenize(
    text: str,
    vocab: dict[str, int],
    max_len: int = 64,
) -> np.ndarray:
    tokens = CONSTS.tokens
    bos = vocab.get("<BOS>", tokens.bos_token_id)
    eos = vocab.get("<EOS>", tokens.eos_token_id)
    unk = vocab.get("<UNK>", tokens.unk_token_id)
    tokens = [bos] + [vocab.get(w, unk) for w in text.lower().split()[: max_len - 2]] + [eos]
    tokens += [CONSTS.tokens.pad_token_id] * (max_len - len(tokens))

    return np.array(tokens[:max_len], dtype=np.int64)


def build_vocab(texts: list[str]) -> dict[str, int]:
    tokens = CONSTS.tokens
    vocab: dict[str, int] = {
        "<PAD>": tokens.pad_token_id,
        "<UNK>": tokens.unk_token_id,
        "<BOS>": tokens.bos_token_id,
        "<EOS>": tokens.eos_token_id,
    }

    for t in texts:
        for w in t.lower().split():
            vocab.setdefault(w, len(vocab))

    return vocab
