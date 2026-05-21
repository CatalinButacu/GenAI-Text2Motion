"""RVQ tokenizer: training-step smoke + codec round-trip on synthetic motion.

The training-step smoke catches the class of bugs that has cost real cloud
money historically (project_cloud_lessons_2026_05_13.md): silent NaN in
loss, gradients that don't flow, optimizer state corruption, codebook
collapse on tiny configs. None of these require a GPU, a dataset, or
checkpoints — just a 30 ms forward+backward on random data.

Codec round-trip uses a tiny config so it runs in well under a second.
"""

from __future__ import annotations

import math
import unittest

import torch

from src.modules.motion.rvq_tokenizer import (
    MotionRVQTokenizer,
    ResidualVectorQuantizer,
    RVQCodebook,
)
from src.modules.motion.training.trainer_utils import tokenCeLoss


def tinyTokenizer(downT: int = 4) -> MotionRVQTokenizer:
    return MotionRVQTokenizer(
        motionDim=24,
        latentDim=16,
        nCodebooks=2,
        codebookSize=32,
        downT=downT,
    )


def makeMotionBatch(batch: int = 2, frames: int = 32, motionDim: int = 24,
                    seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, frames, motionDim, generator=g)


class TestRVQTrainingStep(unittest.TestCase):
    """Single-batch forward -> backward -> optimizer.step.

    The cheapest, highest-leverage 'is training broken?' test we have.
    """

    def test_loss_finite_after_one_step(self):
        tok = tinyTokenizer()
        opt = torch.optim.AdamW(tok.parameters(), lr=1e-3)
        motion = makeMotionBatch()

        tok.train()
        recon, indices, commitLoss = tok(motion)
        reconLoss = torch.nn.functional.mse_loss(recon, motion)
        loss = reconLoss + 0.25 * commitLoss

        self.assertTrue(torch.isfinite(loss).item(), f"non-finite loss: {loss}")
        loss.backward()

        for name, p in tok.named_parameters():
            if p.grad is None:
                continue
            self.assertTrue(torch.isfinite(p.grad).all().item(),
                            f"non-finite grad in {name}")

        opt.step()
        # Indices live in a flat range across all codebooks
        self.assertTrue((indices >= 0).all().item())
        self.assertTrue((indices < tok.rvq.codebookSize).all().item())

    def test_loss_strictly_decreases_over_three_steps(self):
        torch.manual_seed(42)
        tok = tinyTokenizer()
        opt = torch.optim.AdamW(tok.parameters(), lr=5e-3)
        motion = makeMotionBatch(seed=1)

        losses: list[float] = []
        for _ in range(3):
            opt.zero_grad()
            recon, _, commitLoss = tok(motion)
            loss = torch.nn.functional.mse_loss(recon, motion) + 0.25 * commitLoss
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))

        # Loose decrease: must strictly drop by step 3. Slight noise step-to-step is OK.
        self.assertLess(losses[2], losses[0],
                        f"loss did not decrease: start={losses[0]:.4f} -> end={losses[2]:.4f}")

    def test_gradient_flows_into_encoder_and_decoder(self):
        tok = tinyTokenizer()
        motion = makeMotionBatch()
        recon, _, commitLoss = tok(motion)
        loss = torch.nn.functional.mse_loss(recon, motion) + 0.25 * commitLoss
        loss.backward()

        encoderHasGrad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tok.encoder.parameters()
        )
        decoderHasGrad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in tok.decoder.parameters()
        )
        self.assertTrue(encoderHasGrad, "encoder received no gradient")
        self.assertTrue(decoderHasGrad, "decoder received no gradient")


class TestRVQCodec(unittest.TestCase):
    """Encode -> decode round-trip + determinism contracts."""

    def test_encode_decode_shapes_consistent(self):
        tok = tinyTokenizer(downT=4)
        tok.eval()
        motion = makeMotionBatch(batch=1, frames=32, motionDim=24)

        with torch.no_grad():
            indices = tok.encode(motion)
            recon = tok.decode(indices)

        self.assertEqual(indices.shape, (1, 32 // 4, 2))
        self.assertEqual(recon.shape, motion.shape)

    def test_encode_is_deterministic(self):
        tok = tinyTokenizer()
        tok.eval()
        motion = makeMotionBatch(seed=7)

        with torch.no_grad():
            idx1 = tok.encode(motion)
            idx2 = tok.encode(motion)

        self.assertTrue(torch.equal(idx1, idx2))

    def test_decode_is_deterministic(self):
        tok = tinyTokenizer()
        tok.eval()
        motion = makeMotionBatch(seed=8)

        with torch.no_grad():
            indices = tok.encode(motion)
            recon1 = tok.decode(indices)
            recon2 = tok.decode(indices)

        self.assertTrue(torch.allclose(recon1, recon2))

    def test_overfit_single_sample_reduces_recon_mse(self):
        """Training on one tiny clip for ~50 steps must drive recon MSE down."""
        torch.manual_seed(0)
        tok = tinyTokenizer(downT=2)
        opt = torch.optim.AdamW(tok.parameters(), lr=5e-3)
        motion = makeMotionBatch(batch=1, frames=16, motionDim=24, seed=99)

        tok.train()
        with torch.no_grad():
            recon0, _, _ = tok(motion)
            initialMse = float(torch.nn.functional.mse_loss(recon0, motion))

        for _ in range(50):
            opt.zero_grad()
            recon, _, commitLoss = tok(motion)
            mse = torch.nn.functional.mse_loss(recon, motion)
            loss = mse + 0.25 * commitLoss
            loss.backward()
            opt.step()

        finalMse = float(mse.detach())
        # 30 % drop is a comfortable signal "training reduces loss"; sets a low bar
        # so flakes from random seeds don't dominate but a regression that breaks
        # gradient flow or freezes the optimizer would still trip it.
        self.assertLess(finalMse, initialMse * 0.7,
                        f"recon MSE failed to drop ≥30%: "
                        f"{initialMse:.4f} -> {finalMse:.4f}")


class TestRVQCodebookUtils(unittest.TestCase):
    """Dead-code revival and EMA update contract."""

    def test_dead_code_revival_resets_low_usage_entries(self):
        torch.manual_seed(0)
        cb = RVQCodebook(numEntries=16, latentDim=8)
        # Set most entries' clusterSize to ~0 (dead)
        cb.clusterSize.zero_()
        cb.clusterSize[0] = 100.0  # one active

        flat = torch.randn(64, 8)
        nReset = cb.resetDeadCodes(flat, threshold=1.0)
        self.assertEqual(nReset, 15)
        self.assertTrue((cb.clusterSize > 0.0).all().item())

    def test_residual_quantizer_commit_loss_nonneg(self):
        rvq = ResidualVectorQuantizer(nCodebooks=3, codebookSize=8, latentDim=4)
        x = torch.randn(2, 5, 4)
        rvq.eval()
        _, indices, commitLoss = rvq(x)

        self.assertEqual(indices.shape, (2, 5, 3))
        self.assertGreaterEqual(float(commitLoss), 0.0)
        self.assertTrue(math.isfinite(float(commitLoss)))


class TestTokenCeLoss(unittest.TestCase):
    """Value-level contracts on the cross-entropy loss used during SSM training."""

    def test_perfect_logits_yield_near_zero_loss(self):
        B, T, K, V = 2, 4, 3, 8
        targets = torch.randint(0, V, (B, T, K))
        # Force the logit at the target index sky-high; others stay at zero.
        logits = torch.zeros(B, T, K, V)
        scatterIdx = targets.unsqueeze(-1)
        logits.scatter_(-1, scatterIdx, 100.0)
        mask = torch.ones(B, T)

        loss = tokenCeLoss(logits, targets, mask)
        self.assertLess(float(loss), 1e-3)

    def test_uniform_logits_match_log_v(self):
        B, T, K, V = 2, 4, 3, 8
        logits = torch.zeros(B, T, K, V)  # uniform softmax -> 1/V
        targets = torch.zeros((B, T, K), dtype=torch.long)
        mask = torch.ones(B, T)

        loss = tokenCeLoss(logits, targets, mask)
        self.assertAlmostEqual(float(loss), math.log(V), places=4)

    def test_mask_excludes_padded_positions(self):
        B, T, K, V = 1, 4, 2, 8
        logits = torch.zeros(B, T, K, V)
        targets = torch.zeros((B, T, K), dtype=torch.long)
        # Make the last 2 positions catastrophically wrong; mask them out.
        targets[0, 2:, :] = V - 1
        logits[0, 2:, :, 0] = 100.0  # high prob for class 0, but target is V-1
        maskAll = torch.ones(B, T)
        maskValid = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        lossAll = float(tokenCeLoss(logits, targets, maskAll))
        lossValid = float(tokenCeLoss(logits, targets, maskValid))
        self.assertLess(lossValid, lossAll * 0.5,
                        "mask did not exclude the wrong-target padded positions")


if __name__ == "__main__":
    unittest.main()
