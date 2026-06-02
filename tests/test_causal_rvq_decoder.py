"""Causal RVQ decoder must not leak future latent information.

The streaming inference loop needs the RVQ decoder to emit frame ``t`` from
latents up to ``ceil(t / down_t)`` only. The symmetric default decoder uses
``ConvTranspose1d`` which mixes future frames at upsample boundaries, so
streaming with it would invalidate the constant-memory claim.

This test verifies the causality property structurally: feed two latent
sequences that AGREE on a prefix and DIFFER on the suffix, decode both, and
assert the output AGREES on the corresponding raw-frame prefix. Pass = no
future leakage. Fail = the decoder is contaminating past frames with future
latents and must not be used for streaming.

Run on CPU; the tokenizer here is intentionally tiny.
"""
from __future__ import annotations

import unittest

import torch

from src.architecture.rvq_tokenizer import MotionRVQTokenizer


class TestCausalDecoder(unittest.TestCase):

    DOWN_T = 4
    LATENT_LEN = 8
    MOTION_DIM = 16
    LATENT_DIM = 16
    BATCH = 2

    def _build(self, causal: bool) -> MotionRVQTokenizer:
        torch.manual_seed(0)

        return MotionRVQTokenizer(
            motion_dim=self.MOTION_DIM,
            latent_dim=self.LATENT_DIM,
            n_codebooks=2,
            codebook_size=8,
            down_t=self.DOWN_T,
            causal_decoder=causal,
        ).eval()

    def test_symmetric_decoder_leaks_future(self):
        """Sanity: the default decoder is NOT causal -- prefix outputs change
        when we change the suffix. If this test fails, the assumption that
        we need the causal variant in the first place is wrong.
        """
        tok = self._build(causal=False)
        z = torch.randn(self.BATCH, self.LATENT_LEN, self.LATENT_DIM)
        z_alt = z.clone()
        # Disturb the suffix only
        z_alt[:, self.LATENT_LEN // 2 :] += 100.0

        with torch.no_grad():
            out_a = tok.decoder(z.transpose(1, 2)).transpose(1, 2)
            out_b = tok.decoder(z_alt.transpose(1, 2)).transpose(1, 2)
        # The symmetric decoder mixes future frames, so the prefix MUST drift
        # by at least some non-trivial amount past the boundary. We just need
        # *some* difference in the prefix region to confirm the leak; the
        # exact magnitude depends on receptive field.
        prefix_t = (self.LATENT_LEN // 2) * self.DOWN_T
        prefix_diff = (out_a[:, :prefix_t] - out_b[:, :prefix_t]).abs().max()
        self.assertGreater(
            prefix_diff.item(), 1e-3,
            "symmetric decoder unexpectedly preserves prefix when suffix changes; "
            "either the receptive field is shorter than expected or the kernel "
            "weights happened to zero out the leak path",
        )

    def test_causal_decoder_is_causal(self):
        """The whole point: causal decoder's prefix output must be invariant
        to suffix changes. We split the latent sequence in half, perturb the
        second half by a large amount, and require the decoded first half
        to match BIT-EXACT (modulo float noise) the unperturbed decode.
        """
        tok = self._build(causal=True)
        z = torch.randn(self.BATCH, self.LATENT_LEN, self.LATENT_DIM)
        z_alt = z.clone()
        z_alt[:, self.LATENT_LEN // 2 :] += 100.0

        with torch.no_grad():
            out_a = tok.decoder(z.transpose(1, 2)).transpose(1, 2)
            out_b = tok.decoder(z_alt.transpose(1, 2)).transpose(1, 2)
        prefix_t = (self.LATENT_LEN // 2) * self.DOWN_T
        torch.testing.assert_close(
            out_a[:, :prefix_t],
            out_b[:, :prefix_t],
            atol=1e-5,
            rtol=1e-5,
            msg="causal decoder leaked future-latent info into the prefix; "
            "constant-memory streaming claim is invalid until this is fixed",
        )

    def test_causal_and_symmetric_shapes_match(self):
        """Both decoders must produce the same output shape so the rest of
        the pipeline (loss, decode, render) is decoder-agnostic.
        """
        for causal in (False, True):
            tok = self._build(causal=causal)
            z = torch.randn(self.BATCH, self.LATENT_LEN, self.LATENT_DIM)

            with torch.no_grad():
                out = tok.decoder(z.transpose(1, 2)).transpose(1, 2)
            self.assertEqual(
                tuple(out.shape),
                (self.BATCH, self.LATENT_LEN * self.DOWN_T, self.MOTION_DIM),
                f"causal={causal} decoder output shape wrong",
            )


if __name__ == "__main__":
    unittest.main()
