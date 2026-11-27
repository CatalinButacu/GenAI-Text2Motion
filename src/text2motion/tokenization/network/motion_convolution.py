from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def downsample_stage_count(downsample: int) -> int:
    if downsample < 1 or downsample & (downsample - 1):
        raise ValueError(
            f"tokenizer downsample must be a power of two, got {downsample}; the encoder builds "
            f"stride-2 stages and any other value would silently change the frame rate per token"
        )
    return downsample.bit_length() - 1


############################################################
#       TemporalResidualBlock
############################################################
class CausalConvolution1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


def frame_convolution(
    in_channels: int, out_channels: int, kernel_size: int, stride: int, causal: bool
) -> nn.Module:
    if causal:
        return CausalConvolution1d(in_channels, out_channels, kernel_size, stride)
    return nn.Conv1d(
        in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2
    )


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, causal: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            frame_convolution(channels, channels, 3, 1, causal),
            nn.ReLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


############################################################
#       FrameSequenceConvolution
############################################################
class FrameSequenceConvolution(nn.Module):
    net: nn.Sequential

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames.transpose(1, 2)).transpose(1, 2)


############################################################
#       MotionFeatureEncoder
############################################################
class MotionFeatureEncoder(FrameSequenceConvolution):
    def __init__(
        self,
        feature_count: int,
        width: int,
        downsample: int,
        residual_block_count: int,
        causal: bool = False,
    ) -> None:
        super().__init__()
        downsample_stages = downsample_stage_count(downsample)
        layers: list[nn.Module] = [
            frame_convolution(feature_count, width, 3, 1, causal),
            nn.ReLU(),
        ]

        for _ in range(downsample_stages):
            layers += [frame_convolution(width, width, 4, 2, causal), nn.ReLU()]

        for _ in range(residual_block_count):
            layers.append(TemporalResidualBlock(width, causal))

        self.net = nn.Sequential(*layers)


############################################################
#       MotionFeatureDecoder
############################################################
class MotionFeatureDecoder(FrameSequenceConvolution):
    def __init__(
        self,
        feature_count: int,
        width: int,
        downsample: int,
        residual_block_count: int,
        causal: bool = False,
    ) -> None:
        super().__init__()
        upsample_stages = downsample_stage_count(downsample)
        layers: list[nn.Module] = []

        for _ in range(residual_block_count):
            layers.append(TemporalResidualBlock(width, causal))

        for _ in range(upsample_stages):
            if causal:
                layers += [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    frame_convolution(width, width, 3, 1, causal),
                    nn.ReLU(),
                ]
            else:
                layers += [nn.ConvTranspose1d(width, width, 4, stride=2, padding=1), nn.ReLU()]

        layers.append(frame_convolution(width, feature_count, 3, 1, causal))
        self.net = nn.Sequential(*layers)
