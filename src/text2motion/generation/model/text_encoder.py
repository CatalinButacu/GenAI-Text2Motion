from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn
from transformers.masking_utils import create_causal_mask

from text2motion.generation.contracts import (
    TextEncoderConfig,
)


############################################################
#       CLIPTextEncoder
############################################################
class CLIPTextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        from transformers import (
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )

        self.cfg = cfg
        self.frozen_cache: FrozenTextStates | None = None
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model_id)
        self.model = CLIPTextModelWithProjection.from_pretrained(cfg.model_id)

        proj_dim = self.model.config.projection_dim
        if proj_dim != cfg.out_dim:
            raise ValueError(
                f"text encoder projection dim {proj_dim} != cfg.out_dim {cfg.out_dim} "
                f"(set GeneratorConfig.d_text and TextEncoderConfig.out_dim to {proj_dim})"
            )

        self.freeze_except_last(cfg.unfreeze_last_n, cfg.unfreeze_projection)

    def freeze_except_last(self, unfreeze_last_n: int, unfreeze_projection: bool) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)

        layers = self.model.text_model.encoder.layers
        if not 0 <= unfreeze_last_n <= len(layers):
            raise ValueError(f"unfreeze_last_n {unfreeze_last_n} out of [0, {len(layers)}]")

        for layer in layers[len(layers) - unfreeze_last_n :]:
            for p in layer.parameters():
                p.requires_grad_(True)

        if unfreeze_last_n > 0:
            for p in self.model.text_model.final_layer_norm.parameters():
                p.requires_grad_(True)

        if unfreeze_projection:
            for p in self.model.text_projection.parameters():
                p.requires_grad_(True)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def frozen_layer_count(self) -> int:
        return len(self.model.text_model.encoder.layers) - self.cfg.unfreeze_last_n

    def tokens(self, texts: list[str]) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.cfg.max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in tokens.items()}

    @torch.no_grad()
    def frozen_states(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        tm = self.model.text_model
        hidden = tm.embeddings(input_ids=tokens["input_ids"])
        mask = create_causal_mask(
            config=tm.config,
            inputs_embeds=hidden,
            attention_mask=tokens["attention_mask"],
            past_key_values=None,
        )
        for layer in tm.encoder.layers[: self.frozen_layer_count]:
            hidden = layer(hidden, mask, is_causal=True)
        return hidden

    def head(self, states: torch.Tensor, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        tm = self.model.text_model
        mask = create_causal_mask(
            config=tm.config,
            inputs_embeds=states,
            attention_mask=tokens["attention_mask"],
            past_key_values=None,
        )
        hidden = states
        for layer in tm.encoder.layers[self.frozen_layer_count :]:
            hidden = layer(hidden, mask, is_causal=True)
        last_hidden_state = tm.final_layer_norm(hidden)

        ids = tokens["input_ids"].to(dtype=torch.int)
        if tm.eos_token_id == 2:
            eos_pos = ids.argmax(dim=-1)
        else:
            eos_pos = (ids == cast(int, tm.eos_token_id)).int().argmax(dim=-1)
        pooled = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_pos
        ]
        text_embeds = self.model.text_projection(pooled)
        if self.cfg.prefix_len == 1:
            return text_embeds

        seq = last_hidden_state[:, : self.cfg.prefix_len - 1]
        seq_mask = tokens["attention_mask"][:, : self.cfg.prefix_len - 1]
        seq = seq * seq_mask.unsqueeze(-1).to(seq.dtype)
        return torch.cat([text_embeds.unsqueeze(1), seq], dim=1)

    def forward(self, texts: list[str]) -> torch.Tensor:
        t = self.tokens(texts)
        states = (
            self.frozen_states(t)
            if self.frozen_cache is None
            else self.frozen_cache.states(t, texts)
        )
        return self.head(states, t)


############################################################
#       FrozenTextStates
############################################################
def padded_chunk(chunk: list[str], batch_size: int) -> list[str]:
    return chunk + [chunk[0]] * (batch_size - len(chunk))


class FrozenTextStates:
    def __init__(
        self,
        encoder: CLIPTextEncoder,
        rows: np.ndarray,
        captions: dict[str, tuple[int, int]],
        batch_size: int,
    ) -> None:
        self.encoder = encoder
        self.rows = rows
        self.captions = captions
        self.batch_size = batch_size
        self.hits = 0
        self.misses = 0

    @classmethod
    def build(
        cls, encoder: CLIPTextEncoder, captions: list[str], batch_size: int, path: Path
    ) -> FrozenTextStates:
        unique = sorted(set(captions))
        if not unique:
            raise ValueError("no captions to cache")
        chunks = [unique[i : i + batch_size] for i in range(0, len(unique), batch_size)]
        index: dict[str, tuple[int, int]] = {}
        offset = 0
        for chunk in chunks:
            t = encoder.tokens(padded_chunk(chunk, batch_size))
            for i, text in enumerate(chunk):
                n = int(t["attention_mask"][i].sum())
                index[text] = (offset, n)
                offset += n
        width = encoder.model.text_model.config.hidden_size
        rows = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(offset, width))
        for chunk in chunks:
            s = encoder.frozen_states(encoder.tokens(padded_chunk(chunk, batch_size)))
            s = s.float().cpu().numpy()
            for i, text in enumerate(chunk):
                start, n = index[text]
                rows[start : start + n] = s[i, :n]
        rows.flush()
        path.with_suffix(".json").write_text(
            json.dumps({"batch_size": batch_size, "captions": index})
        )
        return cls.load(encoder, path)

    @classmethod
    def load(cls, encoder: CLIPTextEncoder, path: Path) -> FrozenTextStates:
        rows = np.load(path, mmap_mode="r")
        meta = json.loads(path.with_suffix(".json").read_text())
        batch_size = int(meta["batch_size"])
        captions = {text: (int(start), int(n)) for text, (start, n) in meta["captions"].items()}
        probe = sorted(captions)[:batch_size]
        s = encoder.frozen_states(encoder.tokens(padded_chunk(probe, batch_size)))
        s = s.float().cpu().numpy()
        for i, text in enumerate(probe):
            start, n = captions[text]
            if not np.array_equal(s[i, :n], rows[start : start + n]):
                raise RuntimeError(
                    f"frozen text states cache {path} does not match the encoder weights; "
                    f"delete {path} and {path.with_suffix('.json')} to rebuild them"
                )
        return cls(encoder, rows, captions, batch_size)

    def states(self, tokens: dict[str, torch.Tensor], texts: list[str]) -> torch.Tensor:
        if not all(text in self.captions for text in texts):
            self.misses += 1
            return self.encoder.frozen_states(tokens)
        self.hits += 1
        device = self.encoder.device
        out = torch.zeros(
            len(texts), tokens["input_ids"].shape[1], self.rows.shape[1], device=device
        )
        for i, text in enumerate(texts):
            start, n = self.captions[text]
            out[i, :n] = torch.from_numpy(np.array(self.rows[start : start + n])).to(device)
        return out
