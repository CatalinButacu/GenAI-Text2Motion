import json
from pathlib import Path

import torch

from text2motion.app.runtime.checkpoint import GeneratorBundle
from text2motion.generation.contracts import (
    Backbone,
    GeneratorConfig,
    GeneratorTrainingConfig,
    SamplingConfig,
    StreamAnchors,
)
from text2motion.generation.model.generator import MotionTokenGenerator
from text2motion.generation.model.sampling import seeded_sampling_uniforms
from text2motion.generation.model.spec import GeneratorModelSpec
from text2motion.generation.training.resume import GeneratorCheckpoint
from text2motion.generation.training.trainer import GeneratorTrainer
from text2motion.tokenization.contracts import TokenizerConfig
from text2motion.tokenization.nn_tokenizer import FiniteScalarQuantizationTokenizer

GUARD_FILE = Path(__file__).with_name("twin_guard.json")
SEED = 2026
STEPS = 40
SAMPLING = SamplingConfig(temperature=0.5, top_p=0.9, cfg_scale=2.0)
ANCHORS = StreamAnchors(every=10, context=8)
TOK = TokenizerConfig(in_dim=263, width=64, downsample=4, num_quantizers=2, fsq_levels=(4, 4))
GEN = GeneratorConfig(
    d_model=64,
    n_layers=2,
    mamba_n_layers=3,
    d_text=32,
    max_seq_len=32,
    text_prefix_len=2,
    use_end_token=True,
    n_heads=4,
    d_state=8,
    d_conv=4,
    expand=2,
    dt_rank=8,
)


def twin_spec(backbone: Backbone) -> GeneratorModelSpec:
    return GeneratorModelSpec.resolve(
        GEN, backbone, codebook_size=16, num_codebooks=TOK.num_quantizers, use_kernel=False
    )


def seeded_twin(spec: GeneratorModelSpec) -> MotionTokenGenerator:
    torch.manual_seed(SEED)
    return spec.create_model("cpu")


def shapes(state: dict) -> dict:
    return {name: list(tensor.shape) for name, tensor in state.items()}


def resume_record(generator: MotionTokenGenerator) -> dict:
    trainer = GeneratorTrainer(
        generator,
        FiniteScalarQuantizationTokenizer(TOK),
        GeneratorTrainingConfig(),
        downsample=TOK.downsample,
    )
    state = GeneratorCheckpoint(trainer).resume_state(0, float("inf"), 0, None)
    return {
        "keys": sorted(state),
        "generator": shapes(state["generator"]),
        "ema": shapes(state["ema"]),
    }


def bundle_record(spec: GeneratorModelSpec, folder: Path) -> dict:
    tokenizer_file = folder / "tokenizer.pt"
    torch.save({"weight": torch.ones(1)}, tokenizer_file)
    payload = GeneratorBundle.capture(
        seeded_twin(spec),
        torch.nn.Linear(5, 7),
        backbone=spec.backbone,
        tokenizer_ckpt=tokenizer_file,
        resolved_generator_config=spec.as_dict(),
        epoch=0,
        validation_fid=0.0,
    ).payload()
    return {
        "keys": sorted(payload),
        "tokenizer_keys": sorted(payload["tokenizer"]),
        "config_keys": sorted(payload["resolved_generator_config"]),
        "generator": shapes(payload["generator"]),
    }


def stream_tokens(generator: MotionTokenGenerator) -> list:
    text = torch.randn(
        1, GEN.text_prefix_len, GEN.d_text, generator=torch.Generator().manual_seed(SEED)
    )
    uniforms = seeded_sampling_uniforms([SEED], STEPS, TOK.num_quantizers, "cpu")
    stream = generator.eval().stream_token_indices(
        text, STEPS, sampling=SAMPLING, uniforms=uniforms, anchors=ANCHORS
    )
    return [tokens[0].tolist() for tokens in stream]


def twin_record(backbone: Backbone, folder: Path) -> dict:
    spec = twin_spec(backbone)
    generator = seeded_twin(spec)
    return {
        "parameter_order": [name for name, _ in generator.named_parameters()],
        "resume_checkpoint": resume_record(generator),
        "bundle_checkpoint": bundle_record(spec, folder),
        "stream_tokens": stream_tokens(seeded_twin(spec)),
    }


def measure(folder: Path) -> dict:
    return {backbone.value: twin_record(backbone, folder) for backbone in Backbone}


def test_both_twins_match_the_recorded_checkpoint_layout_and_token_stream(tmp_path):
    recorded = json.loads(GUARD_FILE.read_text(encoding="utf-8"))
    measured = measure(tmp_path)

    assert set(recorded) == {backbone.value for backbone in Backbone}
    for name, record in recorded.items():
        assert len(record["stream_tokens"]) == STEPS
        for part, expected in record.items():
            assert measured[name][part] == expected, f"{name}: {part} changed"
        assert set(measured[name]) == set(record)
