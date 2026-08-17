from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TextEncoderConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    out_dim: int = 512
    max_length: int = 77
    unfreeze_last_n: int = 1
    unfreeze_projection: bool = True
    prefix_len: int = 1


class CLIPTextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderConfig) -> None:
        super().__init__()
        from transformers import (
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )

        self.cfg = cfg
        self.tokenizer = CLIPTokenizer.from_pretrained(cfg.model_id)
        self.model = CLIPTextModelWithProjection.from_pretrained(cfg.model_id)

        proj_dim = self.model.config.projection_dim
        if proj_dim != cfg.out_dim:
            raise ValueError(
                f"text encoder projection dim {proj_dim} != cfg.out_dim {cfg.out_dim} "
                f"(set GeneratorConfig.d_text and TextEncoderConfig.out_dim to {proj_dim})"
            )

        self._freeze_except_last(cfg.unfreeze_last_n, cfg.unfreeze_projection)

    def _freeze_except_last(self, unfreeze_last_n: int, unfreeze_projection: bool) -> None:
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

    def forward(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.cfg.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        out = self.model(**tokens)
        if self.cfg.prefix_len == 1:
            return out.text_embeds

        hidden = out.last_hidden_state[:, : self.cfg.prefix_len - 1]
        mask = tokens["attention_mask"][:, : self.cfg.prefix_len - 1]
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        return torch.cat([out.text_embeds.unsqueeze(1), hidden], dim=1)
