"""Typed, config-driven settings — the convention for this and ALL future development.

Rules of the road:
  * No module-level global constants, no ``from ... import *``, no magic numbers in code.
  * Instantiate one ``Config`` (optionally from YAML), then pass a function ONLY the sub-config
    it needs (variable isolation), e.g. ``build_smplx(cfg.avatar)`` -- not the whole ``cfg``.
  * Derived values (representation dims, slices) are computed from the spec, never hand-written
    in two places.

    cfg = load_config("configs/default.yaml")
    slices = cfg.avatar.slices()       # {"body_pose": slice(6, 69), ...}
    dim = cfg.avatar.pose_dim          # 168, derived from pose_segments
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Hml3dReprCfg:
    """Primary representation: standard HumanML3D-263 (body-only, 22-joint skeleton).

    This is what the reused Guo et al. evaluator consumes and what the citable FID is reported on.
    The 263 feature is HumanML3D's own packing (root velocities, ric, rot6d, local velocities, foot
    contacts) in its native frame -- do not re-pack it; use the recovery helper for joint positions.
    """

    dim: int = 263
    num_joints: int = 22
    fps: int = 20  # standard HumanML3D is 20 fps (matches the Guo evaluator / T2M-GPT / MoMask)


@dataclass(frozen=True)
class SmplxAvatarCfg:
    """Secondary track: SMPL-X whole-body, 168-dim per frame (deferred 168 demo track).

    Layout matches the donor exactly so its pretrained 168 RVQ + render code can be reused:
    ``root_orient(3) . trans(3) . body(63) . left_hand(45) . right_hand(45) . jaw(3) . eyes(6)``.
    Stored/trained in **Z-up**; converted to Y-up only at render time. AMASS here is SMPL-H, so
    jaw/eyes are zero-filled (face deferred). ``pose_dim`` and slices are derived from segments.
    """

    model_type: str = "smplx"
    gender: str = "neutral"
    num_joints: int = 55
    num_vertices: int = 10475
    num_betas: int = 10
    use_pca: bool = False  # full axis-angle hands (45 each)
    fps: int = 30
    up_axis: str = "z"  # representation is Z-up; render converts to Y-up
    models_path: Path | None = None
    pose_segments: tuple[tuple[str, int], ...] = (
        ("root_orient", 3),
        ("trans", 3),
        ("body_pose", 63),
        ("left_hand", 45),
        ("right_hand", 45),
        ("jaw_pose", 3),
        ("eyes_pose", 6),
    )

    @property
    def pose_dim(self) -> int:
        return sum(dim for _, dim in self.pose_segments)

    def slices(self) -> dict[str, slice]:
        out: dict[str, slice] = {}
        start = 0

        for name, dim in self.pose_segments:
            out[name] = slice(start, start + dim)
            start += dim

        return out


@dataclass(frozen=True)
class PathsCfg:
    """Resource locations. Data + SMPL-X bodies live in the read-only donor and are license-gated
    (never committed). Every field is overridable from YAML / per machine."""

    donor_root: Path = Path(r"D:\Facultate\dissertation")
    humanml3d_dir: Path | None = None  # donor data/humanml3d (motion_data, mean_std, split, texts)
    eval_matcher: Path | None = None  # Guo et al. finest.tar -- reuse unmodified (ADR 0001)
    eval_stats_dir: Path | None = None  # canonical 263 eval mean.npy/std.npy (T2M Comp_v6 meta)
    t2m_vqvae: Path | None = None  # T2M-GPT's released 263 VQ-VAE -- for harness validation
    glove_dir: Path | None = None  # GloVe 300d for the R-precision text encoder
    amass_dir: Path | None = (
        None  # Z-up AMASS root (SMPL-X release; the source for 263 regeneration)
    )
    smplx_models: Path | None = (
        None  # unzipped SMPL-X bodies; <dir>/SMPLX_{GENDER}.npz (263 regen +
    )
    # the 168 demo). The donor AMASS is the SMPL-X release, so this is the body model for regeneration.
    stats_dir: Path | None = None
    # --- standard HumanML3D-263 regeneration (text2motion/data/hml3d). Uses SMPL-X (above), since the
    # donor AMASS is the SMPL-X release. The SMPL-H/DMPL fields below are LEGACY (old SMPL-H pipeline),
    # unused on the SMPL-X 263 path; kept only for a future byte-exact official-repro option. ---
    smplh_dir: Path | None = (
        None  # LEGACY (old SMPL-H pipeline) -- not used by the SMPL-X 263 regen
    )
    dmpl_dir: Path | None = None  # LEGACY (old SMPL-H pipeline) -- not used by the SMPL-X 263 regen
    body_model_dir: Path | None = None  # LEGACY convenience root for smplh/ + dmpls/
    hml3d_index_csv: Path | None = None  # official HumanML3D index.csv (AMASS->clip mapping)
    hml3d_out_dir: Path | None = None  # regenerated 263 output (new_joint_vecs, Mean/Std, ...)
    texts_dir: Path | None = None  # official HumanML3D texts/<id>.txt (+ M<id>.txt mirror captions)
    cache_dir: Path = Path("data/.cache")
    checkpoints_dir: Path = Path("checkpoints")
    outputs_dir: Path = Path("outputs")


@dataclass(frozen=True)
class DataCfg:
    """Which corpus + representation a run uses. Primary track = HumanML3D-263 (see ADR 0001)."""

    track: str = "hml3d263"  # "hml3d263" (primary) or "smplx168" (deferred demo)
    sources: tuple[str, ...] = ("humanml3d",)
    mirror_augment: bool = True
    max_motion_len: int = 196  # HumanML3D standard cap
    min_motion_len: int = 40


@dataclass(frozen=True)
class TokenizerCfg:
    """Grouped-FSQ motion tokenizer (Contribution A). FSQ (arXiv:2309.15505) over G independent groups
    (the partitioned-latent winner: grouped recon-FID 0.0266 beats strong-RVQ 0.0382). The residual
    variant collapsed (0.22) on the fixed FSQ grid at low latent dim -- kept selectable via `quantizer`
    as the documented motivating finding. Per-group implicit codebook = prod(fsq_levels); grouped
    concatenates the G codes into a G*fsq_dim latent. See .claude/skills/motion-tokenizer and
    .claude/docs/references.md."""

    in_dim: int = 263
    width: int = 512  # fast iterations on a 4GB GPU (FSQ vs RVQ compared at matched width)
    downsample: int = 4  # temporal downsample (two stride-2 convs)
    num_quantizers: int = 6
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)  # per-level codebook = 8*5*5*5 = 1000 (~2^10)
    quantizer: str = (
        "grouped"  # "grouped" (G groups -> G*dim latent) | "residual" (sum -> dim latent)
    )
    quant_dropout: float = (
        0.2  # prob of dropping trailing residual levels during training (residual only)
    )
    n_resblocks: int = 3  # T2M-GPT VQVAEV3 NRES3


@dataclass(frozen=True)
class RvqBaselineCfg:
    """Strong RVQ baseline-to-beat for Contribution A. Recipe is the T2M-GPT (arXiv:2301.06052) +
    MoMask (arXiv:2312.00063) standard: EMA codebook (decay 0.99) + dead-code reset + commitment
    loss + quantization dropout, matching EnCodec/SoundStream RVQ mechanics. It SHARES the conv
    enc/dec with the FSQ tokenizer (same width/downsample/resblocks); only the quantizer differs, so
    the comparison isolates FSQ-vs-VQ at matched capacity. Target to beat: MoMask recon FID 0.019 /
    MPJPE 29.5 mm. See .claude/docs/references.md (Q3 recipe)."""

    in_dim: int = 263
    width: int = 512
    downsample: int = 4
    n_resblocks: int = 3
    num_quantizers: int = 6  # MoMask: 6 residual levels
    codebook_size: int = 512  # MoMask: 512 codes/level
    code_dim: int = 512  # VQ latent dim (encoder width is projected to this)
    ema_decay: float = 0.99  # T2M-GPT mu / EnCodec
    commitment_beta: float = 0.02  # T2M-GPT commit weight
    quant_dropout: float = 0.2  # MoMask quantize_dropout_prob
    reset_threshold: float = 1.0  # codes with EMA cluster size < this get reinitialised


@dataclass(frozen=True)
class TextEncoderCfg:
    """Caption encoder for conditioning. The field standard (T2M-GPT arXiv:2301.06052, MoMask
    arXiv:2312.00063, Mogo) is CLIP ViT-B/32's pooled 512-d sentence vector, projected and fed as a
    single prefix token -- which is exactly ``MotionGenerator``'s interface. We do NOT fully freeze
    it (prior-plateau lesson): the last ``unfreeze_last_n`` transformer layers, the final layer-norm
    and (optionally) the text projection are trainable. ``out_dim`` MUST equal ``GeneratorCfg.d_text``.
    See .claude/docs/references.md."""

    model_id: str = "openai/clip-vit-base-patch32"
    out_dim: int = 512  # CLIP ViT-B/32 text projection dim; must equal GeneratorCfg.d_text
    max_length: int = 77  # CLIP context length
    unfreeze_last_n: int = 1  # unfreeze the last N transformer layers (+ final LN + projection)
    unfreeze_projection: bool = True
    prefix_len: int = 1  # 1 = pooled vector only (T2M-GPT mold); P>1 = pooled + first P-1 token
    # hidden states (AttT2M-style fine-grained conditioning). Must equal GeneratorCfg.text_prefix_len.


@dataclass(frozen=True)
class GeneratorCfg:
    """Causal token-AR generator (Contribution B). `backbone='mamba'` is the contribution (first
    token-AR S6 motion generator); `backbone='transformer'` is the controlled twin (T2M-GPT mold).
    Both are causal/streamable; they differ in runtime memory (fixed SSM state vs growing KV-cache).
    num_codebooks/codebook_size MUST match the tokenizer. See ADR 0002 + .claude/docs/references.md."""

    backbone: str = "mamba"  # "mamba" (Contribution B) | "transformer" (twin baseline)
    d_model: int = 512
    n_layers: int = 8  # transformer-twin depth; a Mamba block is ~half an attn+MLP block, so:
    mamba_n_layers: int = (
        15  # param-matched to the twin (31.84M vs 31.64M, +0.6%); resolved per run
    )
    d_text: int = 512  # text-embedding dim (CLIP/SBERT), prepended as a prefix
    num_codebooks: int = 6  # must equal tokenizer.num_quantizers
    codebook_size: int = 1000  # must equal tokenizer per-level codebook (prod fsq_levels)
    max_seq_len: int = 96  # positions: text_prefix_len + downsampled steps (+ END), with headroom
    dropout: float = 0.1
    text_prefix_len: int = 1  # tokens of text prefix; must equal TextEncoderCfg.prefix_len
    use_end_token: bool = (
        False  # vocab+1 END per head -> self-terminating length (False = legacy ckpts)
    )
    use_kernel: bool = False  # mamba-ssm fused selective scan in training forward. EXPLICIT opt-in
    # (cloud config): when True the import must succeed — no silent fallback (fail-loud rule).
    # mamba (S6) backbone
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int = 32  # selective-Δ rank (≈ d_model/16)
    # transformer twin backbone
    n_heads: int = 8


@dataclass(frozen=True)
class TrainCfg:
    """Generator training recipe (escapes the prior plateau). See .claude/skills/t2m-losses.
    Loss = token-CE + w_recon*soft-decode-recon + w_velocity*vel + w_foot*foot + w_root*root."""

    lr: float = 2e-4
    text_encoder_lr: float = 1e-5  # small LR for the partially-unfrozen CLIP text encoder
    weight_decay: float = (
        0.01  # AdamW decoupled decay (0.0 = plain Adam; the 2026-06 run overfit by ep 20)
    )
    warmup_steps: int = (
        1000  # linear LR warmup; then cosine decay over the FULL run, so size the run
    )
    lr_min_ratio: float = 0.01  # cosine floor as a fraction of peak LR
    ema_decay: float = 0.999  # eval the EMA copy
    cfg_dropout: float = 0.1  # drop text condition this often so CFG works at inference
    pkeep: float = 0.8  # teacher-forcing input corruption: keep a token with this prob, else random
    # (T2M-GPT uses 0.5; fights memorization + exposure bias. 1.0 disables — tests pin that.)
    amp: str = (
        "off"  # "bf16" wraps forward+loss in autocast (Ampere+: A10G/3050); opt/EMA stay fp32
    )
    w_recon: float = 0.5
    w_velocity: float = 0.3
    w_foot: float = 0.1
    w_root: float = 0.3


@dataclass(frozen=True)
class Config:
    seed: int = 42
    device: str = "cuda"
    hml3d: Hml3dReprCfg = field(default_factory=Hml3dReprCfg)
    avatar: SmplxAvatarCfg = field(default_factory=SmplxAvatarCfg)
    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    tokenizer: TokenizerCfg = field(default_factory=TokenizerCfg)
    rvq_baseline: RvqBaselineCfg = field(default_factory=RvqBaselineCfg)
    text_encoder: TextEncoderCfg = field(default_factory=TextEncoderCfg)
    generator: GeneratorCfg = field(default_factory=GeneratorCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


def _to_paths(raw: dict, keys: set[str]) -> dict:
    """Coerce the named keys from str to Path (YAML stores them as strings)."""
    out = dict(raw)

    for key in keys:
        if out.get(key) is not None:
            out[key] = Path(out[key])

    return out


def load_config(path: str | Path) -> Config:
    """Build a ``Config`` from YAML. Absent sections fall back to dataclass defaults."""
    raw = yaml.safe_load(Path(path).read_text()) or {}

    avatar_raw = raw.get("avatar", {})

    if "pose_segments" in avatar_raw:
        avatar_raw["pose_segments"] = tuple(
            (str(name), int(dim)) for name, dim in avatar_raw["pose_segments"]
        )

    avatar = SmplxAvatarCfg(**_to_paths(avatar_raw, {"models_path"}))

    path_keys = set(PathsCfg.__dataclass_fields__)
    paths = PathsCfg(**_to_paths(raw.get("paths", {}), path_keys))

    tok_raw = dict(raw.get("tokenizer", {}))

    if "fsq_levels" in tok_raw:
        tok_raw["fsq_levels"] = tuple(int(v) for v in tok_raw["fsq_levels"])

    return Config(
        seed=raw.get("seed", 42),
        device=raw.get("device", "cuda"),
        hml3d=Hml3dReprCfg(**raw.get("hml3d", {})),
        avatar=avatar,
        paths=paths,
        data=DataCfg(**raw.get("data", {})),
        tokenizer=TokenizerCfg(**tok_raw),
        rvq_baseline=RvqBaselineCfg(**raw.get("rvq_baseline", {})),
        text_encoder=TextEncoderCfg(**raw.get("text_encoder", {})),
        generator=GeneratorCfg(**raw.get("generator", {})),
        train=TrainCfg(**raw.get("train", {})),
    )
