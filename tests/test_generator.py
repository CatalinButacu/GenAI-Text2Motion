"""Generator core (Contribution B): stream==batch parity for BOTH backbones (the bounded-memory
claim made testable), single-batch overfit, and the fixed-state vs growing-KV-cache contrast.
Synthetic tokens -- no real data needed."""

import torch

from text2motion.model.generator import MotionGenerator, token_ce_loss
from text2motion.shared.config import GeneratorCfg


def small_cfg(backbone: str) -> GeneratorCfg:
    return GeneratorCfg(
        backbone=backbone,
        d_model=64,
        n_layers=2,
        d_text=32,
        num_codebooks=2,
        codebook_size=16,
        max_seq_len=16,
        d_state=8,
        d_conv=4,
        expand=2,
        dt_rank=8,
        n_heads=4,
    )


def streamed_logits(gen: MotionGenerator, tokens: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
    """Reproduce forward()'s teacher-forced logits by stepping one input at a time."""
    state = gen.backbone.init_state(tokens.size(0), tokens.device)
    prefix = gen.text_prefix(text)  # (B, P, d_model); P=1 here, stepped like stream() does
    for position in range(prefix.size(1)):
        h, state = gen.backbone.step(prefix[:, position], state)
    outs = [gen.logits(h)]

    for t in range(tokens.size(1) - 1):
        h, state = gen.backbone.step(gen.embed_tokens(tokens[:, t]), state)
        outs.append(gen.logits(h))

    return torch.stack(outs, dim=1)  # (B, L, R, V)


def _parity(backbone: str) -> None:
    torch.manual_seed(0)
    cfg = small_cfg(backbone)
    gen = MotionGenerator(cfg).eval()
    tokens = torch.randint(0, cfg.codebook_size, (2, 8, cfg.num_codebooks))
    text = torch.randn(2, cfg.d_text)

    with torch.no_grad():
        parallel = gen(tokens, text)
        streamed = streamed_logits(gen, tokens, text)

    assert parallel.shape == (2, 8, cfg.num_codebooks, cfg.codebook_size)
    assert torch.allclose(parallel, streamed, atol=1e-4), (parallel - streamed).abs().max().item()


def test_parity_mamba():
    _parity("mamba")


def test_parity_transformer():
    _parity("transformer")


def test_state_is_bounded_mamba_but_kv_grows_transformer():
    """Mamba's recurrent state is fixed-size in T (the efficiency claim); the transformer KV grows."""
    mamba = MotionGenerator(small_cfg("mamba")).eval()
    st = mamba.backbone.init_state(1, torch.device("cpu"))
    shapes0 = [(s[0].shape, s[1].shape) for s in st]
    x = torch.randn(1, mamba.cfg.d_model)

    for _ in range(20):
        _, st = mamba.backbone.step(x, st)

    shapes1 = [(s[0].shape, s[1].shape) for s in st]
    assert shapes0 == shapes1  # fixed-size state regardless of steps taken

    tf = MotionGenerator(small_cfg("transformer")).eval()
    tst = tf.backbone.init_state(1, torch.device("cpu"))
    _, tst = tf.backbone.step(x, tst)
    after1 = tst[0][0].size(2)
    _, tst = tf.backbone.step(x, tst)
    after2 = tst[0][0].size(2)
    assert after2 == after1 + 1  # KV-cache grows by one key per step


def test_single_batch_overfit_mamba():
    torch.manual_seed(0)
    cfg = small_cfg("mamba")
    gen = MotionGenerator(cfg).train()
    opt = torch.optim.Adam(gen.parameters(), lr=3e-3)
    tokens = torch.randint(0, cfg.codebook_size, (2, 8, cfg.num_codebooks))
    text = torch.randn(2, cfg.d_text)

    loss0 = token_ce_loss(gen(tokens, text), tokens).item()

    for _ in range(300):
        loss = token_ce_loss(gen(tokens, text), tokens)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # must memorise the fixed batch: CE collapses and argmax matches the targets
    assert loss.item() < 0.2 * loss0
    acc = (gen(tokens, text).argmax(-1) == tokens).float().mean().item()
    assert acc > 0.95
