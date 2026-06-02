"""Tokenizer contract tests.

Every inference call routes through tokenize() and build_vocab(). A regression
that silently corrupts the special-token IDs would poison the entire model
behavior — and there has been zero unit test guarding these until now.

The contract: PAD=0, UNK=1, BOS=2, EOS=3, and the output is always exactly
max_len tokens long, starts with BOS, contains EOS before padding.
"""

from __future__ import annotations

import unittest

from src.shared.constants import CONSTS
from src.shared.tokenizer import build_vocab, tokenize

TOKENS = CONSTS.tokens


class TestSpecialTokenIds(unittest.TestCase):
    """The IDs are wired into model architectures and checkpoints — they MUST stay 0..3."""

    def test_pad_is_zero(self):
        self.assertEqual(TOKENS.pad_token_id, 0)

    def test_unk_is_one(self):
        self.assertEqual(TOKENS.unk_token_id, 1)

    def test_bos_is_two(self):
        self.assertEqual(TOKENS.bos_token_id, 2)

    def test_eos_is_three(self):
        self.assertEqual(TOKENS.eos_token_id, 3)


class TestBuildVocab(unittest.TestCase):

    def test_reserves_special_tokens_first(self):
        vocab = build_vocab(["alpha beta"])
        self.assertEqual(vocab["<PAD>"], TOKENS.pad_token_id)
        self.assertEqual(vocab["<UNK>"], TOKENS.unk_token_id)
        self.assertEqual(vocab["<BOS>"], TOKENS.bos_token_id)
        self.assertEqual(vocab["<EOS>"], TOKENS.eos_token_id)

    def test_assigns_increasing_ids_to_corpus(self):
        vocab = build_vocab(["alpha beta"])
        self.assertEqual(vocab["alpha"], 4)
        self.assertEqual(vocab["beta"], 5)

    def test_deduplicates_across_texts(self):
        vocab = build_vocab(["alpha beta", "beta gamma", "alpha gamma"])
        self.assertEqual(len(vocab), 4 + 3)  # 4 specials + 3 unique words

    def test_lowercases_words(self):
        vocab = build_vocab(["Hello WORLD"])
        self.assertIn("hello", vocab)
        self.assertIn("world", vocab)
        self.assertNotIn("Hello", vocab)

    def test_empty_corpus_keeps_only_specials(self):
        vocab = build_vocab([])
        self.assertEqual(len(vocab), 4)


class TestTokenize(unittest.TestCase):

    def setUp(self) -> None:
        self.vocab = build_vocab(["the quick brown fox jumps over the lazy dog"])

    def test_output_has_exact_max_length(self):
        out = tokenize("the quick", self.vocab, max_len=16)
        self.assertEqual(out.shape, (16,))

    def test_starts_with_bos(self):
        out = tokenize("anything", self.vocab, max_len=8)
        self.assertEqual(int(out[0]), TOKENS.bos_token_id)

    def test_ends_meaningful_content_with_eos_before_padding(self):
        out = tokenize("the quick", self.vocab, max_len=8)
        # Expected: [BOS, the, quick, EOS, PAD, PAD, PAD, PAD]
        self.assertEqual(int(out[3]), TOKENS.eos_token_id)
        for i in range(4, 8):
            self.assertEqual(int(out[i]), TOKENS.pad_token_id)

    def test_oov_word_maps_to_unk(self):
        out = tokenize("the floofgrumble brown", self.vocab, max_len=16)
        # Position 2 is the OOV word, should be UNK
        self.assertEqual(int(out[2]), TOKENS.unk_token_id)

    def test_known_words_map_to_their_vocab_id(self):
        out = tokenize("the quick brown", self.vocab, max_len=16)
        self.assertEqual(int(out[1]), self.vocab["the"])
        self.assertEqual(int(out[2]), self.vocab["quick"])
        self.assertEqual(int(out[3]), self.vocab["brown"])

    def test_empty_text_yields_bos_eos_then_padding(self):
        out = tokenize("", self.vocab, max_len=8)
        self.assertEqual(int(out[0]), TOKENS.bos_token_id)
        self.assertEqual(int(out[1]), TOKENS.eos_token_id)
        for i in range(2, 8):
            self.assertEqual(int(out[i]), TOKENS.pad_token_id)

    def test_truncates_oversized_input(self):
        words = " ".join(["the"] * 50)
        out = tokenize(words, self.vocab, max_len=10)
        # Output must be exactly 10 long even though input has 50 words
        self.assertEqual(out.shape, (10,))
        # Last position must still be valid (PAD, BOS, EOS, or a real token id)
        self.assertGreaterEqual(int(out[-1]), 0)

    def test_lowercases_input_before_lookup(self):
        out = tokenize("THE QUICK", self.vocab, max_len=8)
        # Should match lowercase entries from vocab
        self.assertEqual(int(out[1]), self.vocab["the"])
        self.assertEqual(int(out[2]), self.vocab["quick"])

    def test_deterministic_across_calls(self):
        a = tokenize("the quick brown fox", self.vocab, max_len=12)
        b = tokenize("the quick brown fox", self.vocab, max_len=12)
        self.assertTrue((a == b).all())


if __name__ == "__main__":
    unittest.main()
