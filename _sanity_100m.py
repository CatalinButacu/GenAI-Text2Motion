from dataclasses import replace

import torch

from text2motion.app.config import load_config
from text2motion.generation.model import MotionGeneratorModule
from text2motion.generation.trainer import GeneratorTrainer
from text2motion.tokenization.model import ResidualFsqTokenizer

cfg = load_config("configs/final100m.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
steps = 150
batch = 2

tokenizer = ResidualFsqTokenizer(cfg.tokenizer)
tokenizer.load_state_dict(torch.load("checkpoints/tokenizer_fsq.pt", map_location="cpu"))
tokenizer.to(device).eval()

for backbone in ["transformer", "mamba"]:
    n_layers = cfg.generator.mamba_n_layers if backbone == "mamba" else cfg.generator.n_layers
    gen_cfg = replace(
        cfg.generator,
        backbone=backbone,
        n_layers=n_layers,
        use_kernel=False,  # local gate: eager scan + checkpointing; kernel is canary-validated
        num_codebooks=cfg.tokenizer.num_quantizers,
        codebook_size=tokenizer.codebook_size,
    )
    torch.manual_seed(cfg.seed)
    generator = MotionGeneratorModule(gen_cfg).to(device)
    trainer = GeneratorTrainer(
        generator,
        tokenizer,
        replace(cfg.train, lr=3e-4, cfg_dropout=0.0, pkeep=1.0),
        downsample=cfg.tokenizer.downsample,
    )
    n_params = sum(p.numel() for p in generator.parameters())

    motion = torch.randn(batch, 64, 263, device=device)
    text = torch.randn(batch, cfg.generator.text_prefix_len, cfg.generator.d_text, device=device)
    lengths = torch.tensor([64, 48], device=device)

    first = trainer.train_step(motion, text, lengths)
    for step in range(steps):
        last = trainer.train_step(motion, text, lengths)
        if step % 30 == 0:
            print(f"{backbone} step {step:3d}  ce {last['ce']:.4f}  total {last['total']:.4f}")
    verdict = "PASS" if last["ce"] < 0.25 * first["ce"] else "FAIL"
    print(f"{backbone}: {n_params:,} params  ce {first['ce']:.3f} -> {last['ce']:.3f}  [{verdict}]")
    del generator, trainer
    if device == "cuda":
        torch.cuda.empty_cache()
