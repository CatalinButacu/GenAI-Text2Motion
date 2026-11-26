from __future__ import annotations

import torch
from torch import nn

from text2motion.motion.representation import DIM, SLICES

_MOVEMENT_INPUT = SLICES["vel"].stop


_MOVEMENT_WIDTH = 512


_MOVEMENT_KERNEL = 4


_GRU_INPUT = 512


_GRU_WIDTH = 1024


_EMBEDDING_DIM = 512


_GLOVE_DIM = 300


_POS_DIM = 15


def bigru_final_state(
    gru: nn.GRU, x: torch.Tensor, hidden: torch.Tensor, lengths: torch.Tensor | None
) -> torch.Tensor:
    h0 = hidden.repeat(1, x.size(0), 1)

    if lengths is None:
        _, h_n = gru(x, h0)
    else:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        _, h_n = gru(packed, h0)

    return torch.cat([h_n[0], h_n[1]], dim=-1)


############################################################
#       GuoMovementEncoder
############################################################
class GuoMovementEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(
                _MOVEMENT_INPUT,
                _MOVEMENT_WIDTH,
                kernel_size=_MOVEMENT_KERNEL,
                stride=_MOVEMENT_KERNEL // 2,
                padding=1,
            ),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(
                _MOVEMENT_WIDTH,
                _MOVEMENT_WIDTH,
                kernel_size=_MOVEMENT_KERNEL,
                stride=_MOVEMENT_KERNEL // 2,
                padding=1,
            ),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(_MOVEMENT_WIDTH, _GRU_INPUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.out_net(out)


############################################################
#       GuoMotionEncoder
############################################################
class GuoMotionEncoder(nn.Module):
    hidden: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.input_emb = nn.Linear(_GRU_INPUT, _GRU_WIDTH)
        self.gru = nn.GRU(
            _GRU_WIDTH, _GRU_WIDTH, num_layers=1, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(_GRU_WIDTH * 2, _GRU_WIDTH),
            nn.LayerNorm(_GRU_WIDTH),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(_GRU_WIDTH, _EMBEDDING_DIM),
        )
        self.register_buffer("hidden", torch.zeros(2, 1, _GRU_WIDTH))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        final = bigru_final_state(self.gru, self.input_emb(x), self.hidden, lengths)
        return self.output_net(final)


############################################################
#       GuoMotionEmbeddingEncoder
############################################################
class GuoMotionEmbeddingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.movement = GuoMovementEncoder()
        self.encoder = GuoMotionEncoder()

    def forward(self, feat263: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if feat263.size(-1) != DIM:
            raise ValueError(
                f"expected last dim {DIM} (standard HumanML3D), got {feat263.size(-1)}"
            )

        seg = self.movement(feat263[..., :_MOVEMENT_INPUT])
        seg_lengths = None

        if lengths is not None:
            seg_lengths = (lengths // _MOVEMENT_KERNEL).clamp(min=1, max=seg.size(1))

        return self.encoder(seg, seg_lengths)


############################################################
#       GuoTextEmbeddingEncoder
############################################################
class GuoTextEmbeddingEncoder(nn.Module):
    hidden: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.pos_emb = nn.Linear(_POS_DIM, _GLOVE_DIM)
        self.input_emb = nn.Linear(_GLOVE_DIM, _EMBEDDING_DIM)
        self.gru = nn.GRU(
            _EMBEDDING_DIM,
            _EMBEDDING_DIM,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_net = nn.Sequential(
            nn.Linear(_EMBEDDING_DIM * 2, _EMBEDDING_DIM),
            nn.LayerNorm(_EMBEDDING_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(_EMBEDDING_DIM, _EMBEDDING_DIM),
        )
        self.register_buffer("hidden", torch.zeros(2, 1, _EMBEDDING_DIM))

    def forward(
        self,
        word_embs: torch.Tensor,
        pos_onehots: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pos_onehots is None:
            pos_onehots = torch.zeros(*word_embs.shape[:2], _POS_DIM, device=word_embs.device)
        x = self.input_emb(word_embs + self.pos_emb(pos_onehots))
        final = bigru_final_state(self.gru, x, self.hidden, lengths)
        return self.output_net(final)
