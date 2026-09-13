from pathlib import Path

import numpy as np
import torch

from text2motion.app.config import load_config
from text2motion.evaluation.contracts import DIVERSITY_PAIRS, GenerationEvaluationProtocol
from text2motion.evaluation.matcher import (
    GuoWordPosVectorizer,
    create_guo_text_input_builder,
    load_frozen_guo_embedding_encoders,
    load_guo_motion_normalization_stats,
)
from text2motion.evaluation.metrics import (
    compute_guo_motion_embeddings,
    compute_guo_text_embeddings,
    diversity,
    fid,
    mm_dist,
    r_precision,
)
from text2motion.motion.annotations import parse_text_file

_MIN_FRAMES = 20


def main() -> None:
    config = load_config("configs/default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(config.paths.hml3d_out_dir)
    text_dir = Path(config.paths.texts_dir or root / "texts")
    if config.paths.eval_matcher is None or config.paths.eval_stats_dir is None:
        raise ValueError("paths.eval_matcher and paths.eval_stats_dir are required")

    build_text = create_guo_text_input_builder(GuoWordPosVectorizer(str(config.paths.our_vab_dir), "our_vab"))
    eval_mean, eval_std = load_guo_motion_normalization_stats(config.paths.eval_stats_dir)
    motion_matcher, text_matcher = load_frozen_guo_embedding_encoders(config.paths.eval_matcher, device=device)

    raw_ids = (root / "test.txt").read_text(encoding="utf-8").splitlines()
    clip_ids = [name.strip() for name in raw_ids if name.strip()]
    clip_ids = list(dict.fromkeys(name[1:] if name.startswith("M") else name for name in clip_ids))

    features: list[np.ndarray] = []
    caption_tokens: list[list[list[str]]] = []
    for clip_id in clip_ids:
        feature_path = root / "new_joint_vecs" / f"{clip_id}.npy"
        text_path = text_dir / f"{clip_id}.txt"
        if not feature_path.is_file() or not text_path.is_file():
            continue
        feature = np.load(feature_path).astype(np.float32)
        if feature.shape[0] < _MIN_FRAMES or np.isnan(feature).any():
            continue
        token_lists = [item.tokens for item in parse_text_file(text_path) if item.tokens]
        if not token_lists:
            continue
        features.append(feature[: config.data.max_motion_len])
        caption_tokens.append(token_lists)

    print(f"usable test clips: {len(features)}")
    motion_embeddings = compute_guo_motion_embeddings(
        motion_matcher, features, eval_mean, eval_std, device=device
    )

    text_pairs, owners = [], []
    for clip_index, token_lists in enumerate(caption_tokens):
        for tokens in token_lists:
            text_pairs.append(build_text(tokens))
            owners.append(clip_index)
    text_embeddings = compute_guo_text_embeddings(text_matcher, text_pairs, device=device)
    owners_array = np.asarray(owners)
    captions_per_clip = [
        np.where(owners_array == clip_index)[0] for clip_index in range(len(features))
    ]

    print(f"mean motion emb norm: {np.linalg.norm(motion_embeddings, axis=1).mean():.3f}")
    print(f"mean text   emb norm: {np.linalg.norm(text_embeddings, axis=1).mean():.3f}")

    rng = np.random.default_rng(0)
    precisions, distances = [], []
    for _ in range(GenerationEvaluationProtocol().reps):
        chosen = np.array([rng.choice(indices) for indices in captions_per_clip])
        selected = text_embeddings[chosen]
        permutation = rng.permutation(len(motion_embeddings))
        precisions.append(r_precision(selected[permutation], motion_embeddings[permutation]))
        distances.append(mm_dist(selected[permutation], motion_embeddings[permutation]))

    precision = np.stack(precisions)
    shuffled = np.random.default_rng(0).permutation(len(motion_embeddings))
    half = len(motion_embeddings) // 2
    real_fid = fid(motion_embeddings[shuffled[:half]], motion_embeddings[shuffled[half:]])
    motion_diversity = diversity(motion_embeddings, num_pairs=DIVERSITY_PAIRS)

    print(f"\n=== GT 'Real' baseline reproduction ({GenerationEvaluationProtocol().reps} reps) ===")
    print(f"{'metric':22}{'ours':>16}{'published':>14}")
    print(
        f"{'R-precision top-1':22}{precision[:, 0].mean():>10.3f} "
        f"+/-{precision[:, 0].std():.3f}{'0.511':>14}"
    )
    print(
        f"{'R-precision top-2':22}{precision[:, 1].mean():>10.3f} "
        f"+/-{precision[:, 1].std():.3f}{'0.703':>14}"
    )
    print(
        f"{'R-precision top-3':22}{precision[:, 2].mean():>10.3f} "
        f"+/-{precision[:, 2].std():.3f}{'0.797':>14}"
    )
    print(f"{'Matching Score':22}{np.mean(distances):>16.3f}{'2.974':>14}")
    print(f"{'Diversity':22}{motion_diversity:>16.3f}{'9.503':>14}")
    print(f"{'FID (real)':22}{real_fid:>16.4f}{'0.002':>14}")


if __name__ == "__main__":
    main()
