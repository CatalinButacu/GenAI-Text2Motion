from pathlib import Path

import numpy as np
import torch

from text2motion.data.hml3d.dataset import parse_text_file
from text2motion.eval.matcher import load_eval_stats, load_matchers
from text2motion.eval.metrics import diversity, fid, mm_dist, r_precision
from text2motion.eval.word_vectorizer import WordVectorizer  # vendored Guo vocab + POS (portable)
from text2motion.shared.config import load_config

w_vec = WordVectorizer(r"data/t2m_glove/glove", "our_vab")


def build_text(tokens):
    toks = ["sos/OTHER"] + tokens[:20] + ["eos/OTHER"]
    we = np.stack([w_vec[t][0] for t in toks]).astype(np.float32)
    pe = np.stack([w_vec[t][1] for t in toks]).astype(np.float32)
    return we, pe


cfg = load_config("configs/default.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path("data/HumanML3D_official")
text_dir = Path(cfg.paths.texts_dir)
vab_root = Path(r"D:\Facultate\dissertation\data\inter-x\processed\glove")
eval_mean, eval_std = load_eval_stats(cfg.paths.eval_stats_dir)

ids = [n.strip() for n in (out / "test.txt").read_text().splitlines() if n.strip()]
ids = list(dict.fromkeys(i[1:] if i.startswith("M") else i for i in ids))

feats, caption_tokens = [], []
for clip_id in ids:
    vec_path = out / "new_joint_vecs" / f"{clip_id}.npy"
    text_path = text_dir / f"{clip_id}.txt"
    if not vec_path.is_file() or not text_path.is_file():
        continue
    feature = np.load(vec_path).astype(np.float32)
    if feature.shape[0] < 20 or np.isnan(feature).any():
        continue
    anns = parse_text_file(text_path)
    token_lists = [a.tokens for a in anns if a.tokens]
    if not token_lists:
        continue
    feats.append(feature[:196])
    caption_tokens.append(token_lists)

print(f"usable test clips: {len(feats)}")

official_matcher = Path("data/official_evaluator/extracted/text_mot_match/model/finest.tar")
motion_matcher, text_matcher = load_matchers(official_matcher, device=device)


def embed_motions(feature_list, batch=32):
    feature_list = [
        f[: (f.shape[0] // 4) * 4] for f in feature_list
    ]  # crop to multiple of unit_length
    embs = []
    for i in range(0, len(feature_list), batch):
        group = feature_list[i : i + batch]
        max_t = max(f.shape[0] for f in group)
        x = np.zeros((len(group), max_t, 263), np.float32)
        lengths = [f.shape[0] for f in group]
        for j, f in enumerate(group):
            x[j, : f.shape[0]] = (f - eval_mean) / eval_std
        with torch.no_grad():
            e = motion_matcher(torch.from_numpy(x).to(device), torch.tensor(lengths, device=device))
        embs.append(e.cpu().numpy())
    return np.concatenate(embs)


def embed_caption_pairs(pairs, batch=32):
    embs = []
    for i in range(0, len(pairs), batch):
        group = pairs[i : i + batch]
        max_l = max(we.shape[0] for we, _ in group)
        we_pad = np.zeros((len(group), max_l, 300), np.float32)
        pe_pad = np.zeros((len(group), max_l, 15), np.float32)
        lengths = [we.shape[0] for we, _ in group]
        for j, (we, pe) in enumerate(group):
            we_pad[j, : we.shape[0]] = we
            pe_pad[j, : pe.shape[0]] = pe
        with torch.no_grad():
            e = text_matcher(
                torch.from_numpy(we_pad).to(device),
                torch.from_numpy(pe_pad).to(device),
                lengths=torch.tensor(lengths, device=device),
            )
        embs.append(e.cpu().numpy())
    return np.concatenate(embs)


motion_embs = embed_motions(feats)

flat_pairs, owner = [], []
for clip_index, token_lists in enumerate(caption_tokens):
    for toks in token_lists:
        flat_pairs.append(build_text(toks))
        owner.append(clip_index)
flat_text_embs = embed_caption_pairs(flat_pairs)
owner = np.array(owner)
captions_per_clip = [np.where(owner == c)[0] for c in range(len(feats))]

print(f"mean motion emb norm: {np.linalg.norm(motion_embs, axis=1).mean():.3f}")
print(f"mean text   emb norm: {np.linalg.norm(flat_text_embs, axis=1).mean():.3f}")

reps = 20
rng = np.random.default_rng(0)
rprec, mmdist = [], []
for _ in range(reps):
    chosen = np.array([rng.choice(idxs) for idxs in captions_per_clip])
    text_sel = flat_text_embs[chosen]
    perm = rng.permutation(len(motion_embs))
    rprec.append(r_precision(text_sel[perm], motion_embs[perm], pool_size=32, top_k=3))
    mmdist.append(mm_dist(text_sel[perm], motion_embs[perm]))

rprec = np.stack(rprec)
half = len(motion_embs) // 2
shuffled = np.random.default_rng(0).permutation(
    len(motion_embs)
)  # test ids are ordered: shuffle before half-split
fid_real = fid(motion_embs[shuffled[:half]], motion_embs[shuffled[half:]])
div = diversity(motion_embs, num_pairs=300)

print("\n=== GT 'Real' baseline reproduction (20 reps, random caption) ===")
print(f"{'metric':22}{'ours':>16}{'published':>14}")
print(
    f"{'R-precision top-1':22}{rprec[:, 0].mean():>10.3f} +/-{rprec[:, 0].std():.3f}{'0.511':>14}"
)
print(
    f"{'R-precision top-2':22}{rprec[:, 1].mean():>10.3f} +/-{rprec[:, 1].std():.3f}{'0.703':>14}"
)
print(
    f"{'R-precision top-3':22}{rprec[:, 2].mean():>10.3f} +/-{rprec[:, 2].std():.3f}{'0.797':>14}"
)
print(f"{'Matching Score':22}{np.mean(mmdist):>16.3f}{'2.974':>14}")
print(f"{'Diversity':22}{div:>16.3f}{'9.503':>14}")
print(f"{'FID (real)':22}{fid_real:>16.4f}{'0.002':>14}")
