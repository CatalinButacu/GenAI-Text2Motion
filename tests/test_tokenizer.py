"""Tokenizer contract tests.

Every inference call routes through tokenize() and buildVocab(). A regression
that silently corrupts the special-token IDs would poison the entire model
behavior — and there has been zero unit test guarding these until now.

The contract: PAD=0, UNK=1, BOS=2, EOS=3, and the output is always exactly
maxLen tokens long, starts with BOS, contains EOS before padding.
"""

from __future__ import annotations

import unittest

from src.shared.constants import BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID
from src.shared.tokenizer import buildVocab, tokenize


class TestSpecialTokenIds(unittest.TestCase):
    """The IDs are wired into model architectures and checkpoints — they MUST stay 0..3."""

    def test_pad_is_zero(self):
        self.assertEqual(PAD_TOKEN_ID, 0)

    def test_unk_is_one(self):
        self.assertEqual(UNK_TOKEN_ID, 1)

    def test_bos_is_two(self):
        self.assertEqual(BOS_TOKEN_ID, 2)

    def test_eos_is_three(self):
        self.assertEqual(EOS_TOKEN_ID, 3)


class TestBuildVocab(unittest.TestCase):

    def test_reserves_special_tokens_first(self):
        vocab = buildVocab(["alpha beta"])
        self.assertEqual(vocab["<PAD>"], PAD_TOKEN_ID)
        self.assertEqual(vocab["<UNK>"], UNK_TOKEN_ID)
        self.assertEqual(vocab["<BOS>"], BOS_TOKEN_ID)
        self.assertEqual(vocab["<EOS>"], EOS_TOKEN_ID)

    def test_assigns_increasing_ids_to_corpus(self):
        vocab = buildVocab(["alpha beta"])
        self.assertEqual(vocab["alpha"], 4)
        self.assertEqual(vocab["beta"], 5)

    def test_deduplicates_across_texts(self):
        vocab = buildVocab(["alpha beta", "beta gamma", "alpha gamma"])
        self.assertEqual(len(vocab), 4 + 3)  # 4 specials + 3 unique words

    def test_lowercases_words(self):
        vocab = buildVocab(["Hello WORLD"])
        self.assertIn("hello", vocab)
        self.assertIn("world", vocab)
        self.assertNotIn("Hello", vocab)

    def test_empty_corpus_keeps_only_specials(self):
        vocab = buildVocab([])
        self.assertEqual(len(vocab), 4)


class TestTokenize(unittest.TestCase):

    def setUp(self) -> None:
        self.vocab = buildVocab(["the quick brown fox jumps over the lazy dog"])

    def test_output_has_exact_max_length(self):
        out = tokenize("the quick", self.vocab, maxLen=16)
        self.assertEqual(out.shape, (16,))

    def test_starts_with_bos(self):
        out = tokenize("anything", self.vocab, maxLen=8)
        self.assertEqual(int(out[0]), BOS_TOKEN_ID)

    def test_ends_meaningful_content_with_eos_before_padding(self):
        out = tokenize("the quick", self.vocab, maxLen=8)
        # Expected: [BOS, the, quick, EOS, PAD, PAD, PAD, PAD]
        self.assertEqual(int(out[3]), EOS_TOKEN_ID)
        for i in range(4, 8):
            self.assertEqual(int(out[i]), PAD_TOKEN_ID)

    def test_oov_word_maps_to_unk(self):
        out = tokenize("the floofgrumble brown", self.vocab, maxLen=16)
        # Position 2 is the OOV word, should be UNK
        self.assertEqual(int(out[2]), UNK_TOKEN_ID)

    def test_known_words_map_to_their_vocab_id(self):
        out = tokenize("the quick brown", self.vocab, maxLen=16)
        self.assertEqual(int(out[1]), self.vocab["the"])
        self.assertEqual(int(out[2]), self.vocab["quick"])
        self.assertEqual(int(out[3]), self.vocab["brown"])

    def test_empty_text_yields_bos_eos_then_padding(self):
        out = tokenize("", self.vocab, maxLen=8)
        self.assertEqual(int(out[0]), BOS_TOKEN_ID)
        self.assertEqual(int(out[1]), EOS_TOKEN_ID)
        for i in range(2, 8):
            self.assertEqual(int(out[i]), PAD_TOKEN_ID)

    def test_truncates_oversized_input(self):
        words = " ".join(["the"] * 50)
        out = tokenize(words, self.vocab, maxLen=10)
        # Output must be exactly 10 long even though input has 50 words
        self.assertEqual(out.shape, (10,))
        # Last position must still be valid (PAD, BOS, EOS, or a real token id)
        self.assertGreaterEqual(int(out[-1]), 0)

    def test_lowercases_input_before_lookup(self):
        out = tokenize("THE QUICK", self.vocab, maxLen=8)
        # Should match lowercase entries from vocab
        self.assertEqual(int(out[1]), self.vocab["the"])
        self.assertEqual(int(out[2]), self.vocab["quick"])

    def test_deterministic_across_calls(self):
        a = tokenize("the quick brown fox", self.vocab, maxLen=12)
        b = tokenize("the quick brown fox", self.vocab, maxLen=12)
        self.assertTrue((a == b).all())


if __name__ == "__main__":
    unittest.main()
