"""CLIP text encoder for caption conditioning.

T2M-GPT (arXiv:2301.06052), MoMask (arXiv:2312.00063) and Mogo all condition on CLIP ViT-B/32's
pooled, projected 512-d sentence vector, fed as a single prefix token -- exactly what
``MotionGenerator.text_prefix`` consumes. Keeping this encoder makes our FID directly comparable to
those baselines. Unlike the field (which fully freezes CLIP), we follow the prior-plateau lesson and
leave the last ``unfreeze_last_n`` transformer layers + final layer-norm + text projection trainable
(a small low-LR fine-tune; see ``TrainCfg.text_encoder_lr``).

``transformers`` is imported lazily so ``import text2motion`` works without it; install it with the
core deps (it is in ``pyproject``). See .claude/docs/references.md.
"""

import torch
from torch import nn

from text2motion.shared.config import TextEncoderCfg


def _require_transformers():
    try:
        import transformers  # noqa: F401

    except ImportError as exc:
        raise RuntimeError(
            "transformers not installed (needed for the CLIP text encoder). Run `uv sync`."
        ) from exc


class CLIPTextEncoder(nn.Module):
    """Caption (list[str]) -> (B, out_dim) pooled+projected CLIP text features.

    Tokenization happens inside ``forward`` (CLIP's own tokenizer, fixed 77-token context), so the
    trainer/data path only ever passes raw strings. Trainable params are restricted to the last
    ``cfg.unfreeze_last_n`` encoder layers, the final layer-norm and (optionally) the text projection;
    everything else is frozen. The output is the raw (un-normalized) projection, matching T2M-GPT's
    ``clip.encode_text``.
    """

    def __init__(self, cfg: TextEncoderCfg) -> None:
        super().__init__()
        _require_transformers()
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

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
        """list[str] of length B -> (B, out_dim). Gradients flow into the unfrozen params."""
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.cfg.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        return self.model(**tokens).text_embeds  # (B, out_dim)
