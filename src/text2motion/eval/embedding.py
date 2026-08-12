from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def embed_motions(matcher, feats, eval_mean, eval_std, device, batch=32):
    out = []
    for start in range(0, len(feats), batch):
        group = feats[start : start + batch]
        max_t = max(f.shape[0] for f in group)
        padded = np.zeros((len(group), max_t, group[0].shape[-1]), np.float32)
        lengths = [f.shape[0] for f in group]
        for row, feat in enumerate(group):
            padded[row, : feat.shape[0]] = (feat - eval_mean) / eval_std
        emb = matcher(torch.from_numpy(padded).to(device), torch.tensor(lengths, device=device))
        out.append(emb.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_texts(matcher, pairs, device, batch=32):
    out = []
    for start in range(0, len(pairs), batch):
        group = pairs[start : start + batch]
        max_l = max(we.shape[0] for we, _ in group)
        we_pad = np.zeros((len(group), max_l, group[0][0].shape[-1]), np.float32)
        pe_pad = np.zeros((len(group), max_l, group[0][1].shape[-1]), np.float32)
        lengths = [we.shape[0] for we, _ in group]
        for row, (we, pe) in enumerate(group):
            we_pad[row, : we.shape[0]] = we
            pe_pad[row, : pe.shape[0]] = pe
        emb = matcher(
            torch.from_numpy(we_pad).to(device),
            torch.from_numpy(pe_pad).to(device),
            lengths=torch.tensor(lengths, device=device),
        )
        out.append(emb.cpu().numpy())
    return np.concatenate(out)
