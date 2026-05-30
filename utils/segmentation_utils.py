from __future__ import annotations

import torch


def align_logits_to_masks(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    if masks.dim() != 4:
        return logits
    b, t = masks.shape[:2]
    if logits.dim() == 4:
        if logits.shape[:2] == (b, t):
            return logits
        raise ValueError(
            f'logits shape {tuple(logits.shape)} does not match masks batch/time {(b, t)}'
        )
    if logits.dim() != 3:
        return logits
    n_frames = b * t
    if logits.shape[0] == n_frames:
        return logits.view(b, t, *logits.shape[1:])
    raise ValueError(
        f'Cannot align logits with shape {tuple(logits.shape)} to masks {(b, t, *masks.shape[2:])}'
    )


def flatten_logits_and_masks(
    logits: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = align_logits_to_masks(logits, masks)
    if logits.dim() == 4:
        b, t, h, w = logits.shape
        logits = logits.reshape(b * t, h, w)
    if masks.dim() == 4:
        b, t, hm, wm = masks.shape
        masks = masks.reshape(b * t, hm, wm)
    return logits, masks
