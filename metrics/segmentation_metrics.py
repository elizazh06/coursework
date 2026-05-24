from __future__ import annotations
import torch
from utils.segmentation_utils import flatten_logits_and_masks

def _threshold_masks(logits: torch.Tensor, threshold: float=0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).bool()

class MeanIoUMetric:

    def __init__(self, name: str='mean_iou', threshold: float=0.5):
        self.name = name
        self.threshold = float(threshold)

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        logits, masks = flatten_logits_and_masks(logits, masks)
        pred = _threshold_masks(logits, self.threshold)
        gt = masks.bool().to(pred.device)
        if pred.shape[-2:] != gt.shape[-2:]:
            import torch.nn.functional as F
            gt = F.interpolate(gt.float().unsqueeze(1), size=pred.shape[-2:], mode='nearest').squeeze(1).bool()
        intersection = (pred & gt).flatten(1).sum(dim=1).float()
        union = (pred | gt).flatten(1).sum(dim=1).float()
        iou = intersection / (union + 1e-06)
        return iou.mean().item()
JMetric = MeanIoUMetric

def _dilate(mask: torch.Tensor, kernel: int=3) -> torch.Tensor:
    pad = kernel // 2
    return torch.nn.functional.max_pool2d(mask.float().unsqueeze(0), kernel_size=kernel, stride=1, padding=pad).squeeze(0).bool()

def _boundary(mask: torch.Tensor) -> torch.Tensor:
    eroded = ~_dilate(~mask)
    return mask ^ eroded

class BoundaryFMetric:

    def __init__(self, name: str='boundary_f', threshold: float=0.5):
        self.name = name
        self.threshold = float(threshold)

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        logits, masks = flatten_logits_and_masks(logits, masks)
        pred = _threshold_masks(logits, self.threshold)
        gt = masks.bool().to(pred.device)
        if pred.shape[-2:] != gt.shape[-2:]:
            import torch.nn.functional as F
            gt = F.interpolate(gt.float().unsqueeze(1), size=pred.shape[-2:], mode='nearest').squeeze(1).bool()
        f_scores = []
        for (p, g) in zip(pred, gt):
            bp = _boundary(p)
            bg = _boundary(g)
            tp = (bp & bg).sum().float()
            fp = (bp & ~bg).sum().float()
            fn = (~bp & bg).sum().float()
            precision = tp / (tp + fp + 1e-06)
            recall = tp / (tp + fn + 1e-06)
            f = 2 * precision * recall / (precision + recall + 1e-06)
            f_scores.append(f.item())
        return float(sum(f_scores) / len(f_scores)) if f_scores else 0.0

class JAndFMetric:

    def __init__(self, name: str='j_and_f', threshold: float=0.5):
        self.name = name
        self._j = MeanIoUMetric(threshold=threshold)
        self._f = BoundaryFMetric(threshold=threshold)

    def __call__(self, **batch) -> float:
        j = self._j(**batch)
        f = self._f(**batch)
        return (j + f) / 2.0
