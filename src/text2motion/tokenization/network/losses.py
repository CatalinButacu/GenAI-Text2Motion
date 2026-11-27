from __future__ import annotations

import torch
from torch.nn import functional as F


def heading_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    frames = target.shape[1]
    if lengths is None:
        mask = torch.ones(target.shape[:2], device=target.device, dtype=target.dtype)
    else:
        mask = (
            torch.arange(frames, device=target.device)[None, :] < lengths.to(target.device)[:, None]
        ).to(target.dtype)
    turning_error = (reconstruction[..., 0] - target[..., 0]) * mask
    accumulated = torch.cumsum(turning_error, dim=1)
    return (accumulated.abs() * mask).sum() / mask.sum().clamp(min=1.0)


def feature_velocity_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    if reconstruction.shape != target.shape or target.shape[0] == 0 or target.shape[1] < 2:
        raise ValueError(
            f"the reconstruction loss needs equal shapes with at least 2 frames, got "
            f"{tuple(reconstruction.shape)} and {tuple(target.shape)}"
        )
    if lengths is None:
        feature_loss = F.l1_loss(reconstruction, target)
        reconstruction_velocity = reconstruction[:, 1:] - reconstruction[:, :-1]
        target_velocity = target[:, 1:] - target[:, :-1]
        return feature_loss + F.l1_loss(reconstruction_velocity, target_velocity)
    frame_mask = (
        torch.arange(target.shape[1], device=target.device)[None, :]
        < lengths.to(target.device)[:, None]
    ).to(target.dtype)
    feature_loss = masked_mean_abs(reconstruction - target, frame_mask)
    reconstruction_velocity = reconstruction[:, 1:] - reconstruction[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    velocity_mask = frame_mask[:, 1:] * frame_mask[:, :-1]
    velocity_loss = masked_mean_abs(reconstruction_velocity - target_velocity, velocity_mask)
    return feature_loss + velocity_loss


def masked_mean_abs(difference: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    per_frame = difference.abs().mean(-1)
    return (per_frame * frame_mask).sum() / frame_mask.sum().clamp(min=1.0)
