# python scripts/research/diffusion_decoder_pilot.py --minutes 60 --out_dir artifacts/outputs/qa/diffusion_decoder_pilot
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from text2motion.app.bootstrap import ApplicationBootstrap
from text2motion.motion.contracts import Split
from text2motion.motion.normalization import MotionScaler
from text2motion.motion.storage import FEATURES_DIR, base_clip_ids
from text2motion.tokenization.contracts import TokenizerReconstruction
from text2motion.tokenization.metrics import evaluate_tokenizer_reconstruction
from text2motion.tokenization.nn_model import TokenizerNetwork

TOKENIZER_CONFIG = "configs/tokenizer/fsq_g8_v1024.yaml"
SEED = 2026
FEATURES = 263
WINDOW = 64
DIFFUSION_STEPS = 1000
DILATIONS = (1, 1, 2, 2, 4, 4, 1, 1)
CHUNK_FRAMES = 16
GATE_STEPS = 300
GATE_RATIO = 0.5


class DenoiserBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, width)
        self.film = nn.Linear(width, 2 * width)
        self.mix = nn.Conv1d(width, width, 1)

    def forward(self, hidden: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(time_embedding).unsqueeze(-1).chunk(2, dim=1)
        update = self.norm(self.conv(hidden)) * (1 + scale) + shift
        return hidden + self.mix(F.silu(update))


class RefinerDenoiser(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.inlet = nn.Conv1d(2 * FEATURES, width, 1)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([DenoiserBlock(width, dilation) for dilation in DILATIONS])
        self.outlet = nn.Conv1d(width, FEATURES, 1)
        if self.outlet.bias is None:
            raise ValueError("the denoiser outlet needs a bias")
        nn.init.zeros_(self.outlet.weight)
        nn.init.zeros_(self.outlet.bias)

    def time_features(self, steps: torch.Tensor) -> torch.Tensor:
        half = self.width // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=steps.device) / half)
        angles = steps.float().unsqueeze(-1) * frequencies
        return self.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=-1))

    def forward(
        self, noisy: torch.Tensor, condition: torch.Tensor, steps: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.inlet(torch.cat([noisy, condition], dim=-1).transpose(1, 2))
        time_embedding = self.time_features(steps)
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        return condition + self.outlet(hidden).transpose(1, 2)


def cosine_alpha_bars(device: str) -> torch.Tensor:
    positions = (
        torch.arange(DIFFUSION_STEPS + 1, device=device, dtype=torch.float64) / DIFFUSION_STEPS
    )
    curve = torch.cos((positions + 0.008) / 1.008 * math.pi / 2) ** 2
    return (curve[1:] / curve[0]).clamp(1e-5, 1.0).float()


@dataclass(frozen=True)
class SamplerSetting:
    steps: int
    start: int

    @property
    def name(self) -> str:
        return f"diffusion_{self.steps}_steps_from_{self.start}"


def ddim_sample(
    denoiser: RefinerDenoiser,
    condition: torch.Tensor,
    alpha_bars: torch.Tensor,
    setting: SamplerSetting,
) -> torch.Tensor:
    generator = torch.Generator(device=condition.device).manual_seed(SEED)
    noise = torch.randn(condition.shape, generator=generator, device=condition.device)
    alpha_start = alpha_bars[setting.start]
    sample = alpha_start.sqrt() * condition + (1 - alpha_start).sqrt() * noise
    schedule = torch.linspace(setting.start, 0, setting.steps, device=condition.device).long()
    clean = condition
    for index, step in enumerate(schedule):
        batch_steps = step.expand(condition.shape[0])
        clean = denoiser(sample, condition, batch_steps)
        if index == setting.steps - 1:
            break
        alpha_now = alpha_bars[step]
        alpha_next = alpha_bars[schedule[index + 1]]
        noise = (sample - alpha_now.sqrt() * clean) / (1 - alpha_now).sqrt()
        sample = alpha_next.sqrt() * clean + (1 - alpha_next).sqrt() * noise
    return clean


class DiffusionDecodedTokenizer(TokenizerNetwork):
    def __init__(
        self,
        tokenizer: TokenizerNetwork,
        denoiser: RefinerDenoiser,
        alpha_bars: torch.Tensor,
        setting: SamplerSetting,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.denoiser = denoiser
        self.alpha_bars = alpha_bars
        self.setting = setting
        self.downsample = tokenizer.downsample

    def forward(self, motion: torch.Tensor) -> TokenizerReconstruction:
        indices = self.tokenizer.encode(motion)
        condition = self.tokenizer.decode(indices)
        return TokenizerReconstruction(
            reconstruction=ddim_sample(self.denoiser, condition, self.alpha_bars, self.setting),
            token_indices=indices,
            commitment_loss=torch.zeros((), device=motion.device),
        )


def training_clip_paths(root: Path) -> list[tuple[Path, int]]:
    clips = []
    for clip_id in base_clip_ids(root, Split.TRAIN):
        path = root / FEATURES_DIR / f"{clip_id}.npy"
        frames = int(np.load(path, mmap_mode="r").shape[0])
        if frames < WINDOW:
            continue
        clips.append((path, frames))
    return clips


@dataclass(frozen=True)
class ClipBank:
    features: torch.Tensor
    starts: torch.Tensor
    lengths: torch.Tensor


def clip_bank(clips: list[tuple[Path, int]], scaler: MotionScaler, device: str) -> ClipBank:
    total = sum(frames for _, frames in clips)
    features = torch.empty(total, FEATURES, device=device)
    starts = []
    offset = 0
    for path, frames in clips:
        clip = np.array(np.load(path, mmap_mode="r"), np.float32)
        normalized = np.asarray(scaler.normalize(clip), np.float32)
        features[offset : offset + frames] = torch.from_numpy(normalized).to(device)
        starts.append(offset)
        offset += frames
    lengths = [frames for _, frames in clips]
    return ClipBank(
        features=features,
        starts=torch.tensor(starts, device=device),
        lengths=torch.tensor(lengths, device=device),
    )


def sample_batch(bank: ClipBank, batch: int, generator: torch.Generator) -> torch.Tensor:
    device = bank.features.device
    picks = torch.randint(0, len(bank.starts), (batch,), generator=generator, device=device)
    room = (bank.lengths[picks] - WINDOW + 1).float()
    offsets = (torch.rand(batch, generator=generator, device=device) * room).long()
    first = bank.starts[picks] + offsets
    frames = first.unsqueeze(1) + torch.arange(WINDOW, device=device)
    return bank.features[frames]


def diffusion_loss(
    denoiser: RefinerDenoiser,
    clean: torch.Tensor,
    condition: torch.Tensor,
    alpha_bars: torch.Tensor,
    steps: torch.Tensor,
) -> torch.Tensor:
    alpha = alpha_bars[steps].view(-1, 1, 1)
    noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * torch.randn_like(clean)
    return F.mse_loss(denoiser(noisy, condition, steps), clean)


def overfit_gate(
    tokenizer: TokenizerNetwork,
    clean: torch.Tensor,
    alpha_bars: torch.Tensor,
    width: int,
    device: str,
) -> dict[str, float]:
    probe = RefinerDenoiser(width).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    condition = tokenizer.decode(tokenizer.encode(clean))
    steps = torch.full((clean.shape[0],), 100, device=device)
    with torch.no_grad():
        first = float(diffusion_loss(probe, clean, condition, alpha_bars, steps))
    for _ in range(GATE_STEPS):
        optimizer.zero_grad()
        loss = diffusion_loss(probe, clean, condition, alpha_bars, steps)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        last = float(diffusion_loss(probe, clean, condition, alpha_bars, steps))
    if last > GATE_RATIO * first:
        raise RuntimeError(f"overfit gate failed: loss {first:.4f} -> {last:.4f}")
    return {"gate_first_loss": first, "gate_last_loss": last}


def train_denoiser(
    tokenizer: TokenizerNetwork,
    bank: ClipBank,
    alpha_bars: torch.Tensor,
    arguments: argparse.Namespace,
    device: str,
) -> tuple[RefinerDenoiser, list[dict[str, float]]]:
    denoiser = RefinerDenoiser(arguments.width).to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=arguments.lr)
    generator = torch.Generator(device=device).manual_seed(SEED)
    deadline = time.monotonic() + arguments.minutes * 60
    history = []
    step = 0
    while time.monotonic() < deadline:
        clean = sample_batch(bank, arguments.batch_size, generator)
        with torch.no_grad():
            condition = tokenizer.decode(tokenizer.encode(clean))
        steps = torch.randint(0, DIFFUSION_STEPS, (clean.shape[0],), device=device)
        optimizer.zero_grad()
        loss = diffusion_loss(denoiser, clean, condition, alpha_bars, steps)
        loss.backward()
        optimizer.step()
        step += 1
        if step % 500 == 0:
            history.append({"step": step, "loss": float(loss)})
            print(f"step {step} loss {float(loss):.5f}", flush=True)
    return denoiser, history


def median_milliseconds(function, repeats: int) -> float:
    function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def chunk_latency(
    tokenizer: TokenizerNetwork,
    denoiser: RefinerDenoiser,
    alpha_bars: torch.Tensor,
    setting: SamplerSetting,
    device: str,
) -> dict[str, float]:
    frames = CHUNK_FRAMES * 3
    motion = torch.randn(1, frames, FEATURES, device=device)
    with torch.no_grad():
        tokens = tokenizer.encode(motion)
        convolution = median_milliseconds(lambda: tokenizer.decode(tokens), 50)
        condition = tokenizer.decode(tokens)
        diffusion = median_milliseconds(
            lambda: ddim_sample(denoiser, condition, alpha_bars, setting), 50
        )
    return {
        "chunk_ms_convolution": convolution,
        "chunk_ms_diffusion_total": convolution + diffusion,
    }


def reconstruction_scores(
    module: TokenizerNetwork, app, scaler: MotionScaler, root: Path, evaluator, device: str
) -> dict[str, float]:
    return evaluate_tokenizer_reconstruction(
        module,
        root,
        scaler,
        max_frames=app.config.data.max_motion_len,
        device=device,
        split=Split.VALIDATION,
        shuffle_seed=SEED,
        reconstruction_fid_scorer=evaluator.calculate_reconstruction_fid,
    )


def write_result(out_dir: Path, result: dict[str, object]) -> None:
    (out_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--sample_steps", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--sample_starts", type=int, nargs="+", default=[999, 250])
    parser.add_argument(
        "--out_dir", type=Path, default=Path("artifacts/outputs/qa/diffusion_decoder_pilot")
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    app = ApplicationBootstrap.from_config(TOKENIZER_CONFIG, seed=SEED)
    device = app.device
    tokenizer = app.create_tokenizer_network()
    checkpoint = Path(app.config.paths.checkpoints_dir) / "tokenizer" / "fsq_g8_v1024.pt"
    tokenizer.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    tokenizer.to(device).eval().requires_grad_(False)
    repository = app.create_motion_repository()
    scaler = repository.load_scaler()
    evaluator = app.create_humanml3d_generation_evaluator()
    alpha_bars = cosine_alpha_bars(device)
    clips = training_clip_paths(repository.root)
    print(f"{len(clips)} training clips with at least {WINDOW} frames", flush=True)
    result: dict[str, object] = {"train_clips": len(clips)}
    result["baseline"] = reconstruction_scores(
        tokenizer, app, scaler, repository.root, evaluator, device
    )
    print(f"baseline {result['baseline']}", flush=True)
    bank = clip_bank(clips, scaler, device)
    gate_generator = torch.Generator(device=device).manual_seed(SEED)
    gate_batch = sample_batch(bank, 8, gate_generator)
    result.update(overfit_gate(tokenizer, gate_batch, alpha_bars, arguments.width, device))
    denoiser, history = train_denoiser(tokenizer, bank, alpha_bars, arguments, device)
    del bank, gate_batch
    torch.cuda.empty_cache()
    denoiser.eval()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(denoiser.state_dict(), arguments.out_dir / "denoiser.pt")
    result["history"] = history
    write_result(arguments.out_dir, result)
    settings = [
        SamplerSetting(steps, start)
        for steps in arguments.sample_steps
        for start in arguments.sample_starts
    ]
    for setting in settings:
        candidate = DiffusionDecodedTokenizer(tokenizer, denoiser, alpha_bars, setting)
        result[setting.name] = {
            **reconstruction_scores(candidate, app, scaler, repository.root, evaluator, device),
            **chunk_latency(tokenizer, denoiser, alpha_bars, setting, device),
        }
        print(f"{setting.name} {result[setting.name]}", flush=True)
        write_result(arguments.out_dir, result)


if __name__ == "__main__":
    main()
