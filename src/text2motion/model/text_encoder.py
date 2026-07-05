import torch
from torch import nn

from text2motion.shared.config import TextEncoderCfg


class CLIPTextEncoder(nn.Module):
    def __init__(self, cfg: TextEncoderCfg) -> None:
        super().__init__()
        from transformers import (  # lazy; raises if absent
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
                f"(set GeneratorCfg.d_text and TextEncoderCfg.out_dim to {proj_dim})"
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

        if unfreeze_last_n > 0:  # the final norm is meaningful only if the top layers move
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
            return out.text_embeds  # (B, out_dim)

        hidden = out.last_hidden_state[:, : self.cfg.prefix_len - 1]  # (B, P-1, out_dim)
        mask = tokens["attention_mask"][:, : self.cfg.prefix_len - 1]  # zero the padded positions
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        return torch.cat([out.text_embeds.unsqueeze(1), hidden], dim=1)  # (B, P, out_dim)
