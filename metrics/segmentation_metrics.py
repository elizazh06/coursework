"""Segmentation metrics for Ref-AVS-Bench.

Standard metrics used in audio-visual segmentation literature:
  - J (Jaccard / mean IoU): intersection over union at threshold 0.5
  - F (boundary F-score): harmonic mean of boundary precision / recall
  - J&F: average of J and F (primary Ref-AVS metric)
"""

from __future__ import annotations

import torch


def _threshold_masks(
    logits: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Convert raw logits to binary mask via threshold on sigmoid."""
    return (torch.sigmoid(logits) >= threshold).bool()


# ---------------------------------------------------------------------------
# J metric (Jaccard / mean IoU)
# ---------------------------------------------------------------------------

class MeanIoUMetric:
    """Mean Intersection-over-Union over the batch.

    Computes IoU per frame (after sigmoid + threshold), averages over all.
    """

    def __init__(self, name: str = "mean_iou", threshold: float = 0.5):
        self.name = name
        self.threshold = float(threshold)

    def __call__(
        self,
        logits: torch.Tensor,   # [B*T, H, W] or [B, T, H, W]
        masks: torch.Tensor,    # [B, T, H, W] or [B*T, H, W]
        **_,
    ) -> float:
        if logits.dim() == 4:
            b, t, h, w = logits.shape
            logits = logits.view(b * t, h, w)
        if masks.dim() == 4:
            b, t, hm, wm = masks.shape
            masks = masks.view(b * t, hm, wm)

        pred = _threshold_masks(logits, self.threshold)
        gt = masks.bool().to(pred.device)

        if pred.shape[-2:] != gt.shape[-2:]:
            import torch.nn.functional as F
            gt = F.interpolate(
                gt.float().unsqueeze(1),
                size=pred.shape[-2:],
                mode="nearest",
            ).squeeze(1).bool()

        intersection = (pred & gt).flatten(1).sum(dim=1).float()
        union = (pred | gt).flatten(1).sum(dim=1).float()
        iou = intersection / (union + 1e-6)
        return iou.mean().item()


# convenience alias used in paper
JMetric = MeanIoUMetric


# ---------------------------------------------------------------------------
# F metric (boundary F-score)
# ---------------------------------------------------------------------------

def _dilate(mask: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Fast morphological dilation using max-pooling."""
    pad = kernel // 2
    return torch.nn.functional.max_pool2d(
        mask.float().unsqueeze(0),
        kernel_size=kernel,
        stride=1,
        padding=pad,
    ).squeeze(0).bool()


def _boundary(mask: torch.Tensor) -> torch.Tensor:
    """Extract boundary pixels: mask XOR eroded(mask)."""
    eroded = ~_dilate(~mask)
    return mask ^ eroded


class BoundaryFMetric:
    """Boundary F-score (F-measure) as used in AVS benchmarks.

    Computes precision / recall on boundary pixels (extracted via erosion)
    and returns their harmonic mean.
    """

    def __init__(self, name: str = "boundary_f", threshold: float = 0.5):
        self.name = name
        self.threshold = float(threshold)

    def __call__(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        **_,
    ) -> float:
        if logits.dim() == 4:
            b, t, h, w = logits.shape
            logits = logits.view(b * t, h, w)
        if masks.dim() == 4:
            b, t, hm, wm = masks.shape
            masks = masks.view(b * t, hm, wm)

        pred = _threshold_masks(logits, self.threshold)
        gt = masks.bool().to(pred.device)

        if pred.shape[-2:] != gt.shape[-2:]:
            import torch.nn.functional as F
            gt = F.interpolate(
                gt.float().unsqueeze(1),
                size=pred.shape[-2:],
                mode="nearest",
            ).squeeze(1).bool()

        f_scores = []
        for p, g in zip(pred, gt):
            bp = _boundary(p)
            bg = _boundary(g)

            tp = (bp & bg).sum().float()
            fp = (bp & ~bg).sum().float()
            fn = (~bp & bg).sum().float()

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f = 2 * precision * recall / (precision + recall + 1e-6)
            f_scores.append(f.item())

        return float(sum(f_scores) / len(f_scores)) if f_scores else 0.0


# ---------------------------------------------------------------------------
# J&F combined (primary Ref-AVS metric)
# ---------------------------------------------------------------------------

class JAndFMetric:
    """J&F score = (J + F) / 2.  Primary metric for Ref-AVS-Bench."""

    def __init__(self, name: str = "j_and_f", threshold: float = 0.5):
        self.name = name
        self._j = MeanIoUMetric(threshold=threshold)
        self._f = BoundaryFMetric(threshold=threshold)

    def __call__(self, **batch) -> float:
        j = self._j(**batch)
        f = self._f(**batch)
        return (j + f) / 2.0
