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

class SoftMeanIoUMetric:

    def __init__(self, name: str='soft_mean_iou'):
        self.name = name

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        logits, masks = flatten_logits_and_masks(logits, masks)
        probs = torch.sigmoid(logits)
        gt = masks.float().to(probs.device)
        if probs.shape[-2:] != gt.shape[-2:]:
            import torch.nn.functional as F
            gt = F.interpolate(gt.unsqueeze(1), size=probs.shape[-2:], mode='nearest').squeeze(1)
        intersection = (probs * gt).flatten(1).sum(dim=1)
        union = (probs + gt - probs * gt).flatten(1).sum(dim=1)
        iou = intersection / (union + 1e-06)
        return iou.mean().item()

class PredPositiveRateMetric:

    def __init__(self, name: str='pred_pos_rate', threshold: float=0.5):
        self.name = name
        self.threshold = float(threshold)

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        del masks
        return _threshold_masks(logits, self.threshold).float().mean().item()

class TargetPositiveRateMetric:

    def __init__(self, name: str='target_pos_rate'):
        self.name = name

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        del logits
        return masks.float().mean().item()

class LogitMeanMetric:

    def __init__(self, name: str='logit_mean'):
        self.name = name

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        del masks
        return logits.detach().float().mean().item()

class LogitStdMetric:

    def __init__(self, name: str='logit_std'):
        self.name = name

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        del masks
        return logits.detach().float().std(unbiased=False).item()

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

def _aligned_pred_and_gt(logits: torch.Tensor, masks: torch.Tensor, threshold: float=0.5):
    logits, masks = flatten_logits_and_masks(logits, masks)
    pred = _threshold_masks(logits, threshold)
    gt = masks.bool().to(pred.device)
    if pred.shape[-2:] != gt.shape[-2:]:
        import torch.nn.functional as F
        gt = F.interpolate(gt.float().unsqueeze(1), size=pred.shape[-2:], mode='nearest').squeeze(1).bool()
    return (pred, gt)

def _foreground_binary_stats(logits: torch.Tensor, masks: torch.Tensor, threshold: float=0.5):
    pred, gt = _aligned_pred_and_gt(logits, masks, threshold)
    pred_flat = pred.flatten(1)
    gt_flat = gt.flatten(1)
    tp = (pred_flat & gt_flat).sum(dim=1).float()
    fp = (pred_flat & ~gt_flat).sum(dim=1).float()
    fn = (~pred_flat & gt_flat).sum(dim=1).float()
    return tp, fp, fn

class PixelAccuracyMetric:

    def __init__(self, name: str='pixel_accuracy', threshold: float=0.5):
        self.name = name
        self.threshold = float(threshold)

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        pred, gt = _aligned_pred_and_gt(logits, masks, self.threshold)
        return (pred == gt).float().mean().item()

class DiceCoefficientMetric:

    def __init__(self, name: str='dice_coefficient', threshold: float=0.5, eps: float=1.0):
        self.name = name
        self.threshold = float(threshold)
        self.eps = float(eps)

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        pred, gt = _aligned_pred_and_gt(logits, masks, self.threshold)
        pred_f = pred.float().flatten(1)
        gt_f = gt.float().flatten(1)
        intersection = (pred_f * gt_f).sum(dim=1)
        denom = pred_f.sum(dim=1) + gt_f.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)
        return dice.mean().item()

class ForegroundPrecisionMetric:

    def __init__(self, name: str='foreground_precision', threshold: float=0.5):
        self.name = name
        self.threshold = float(threshold)

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        tp, fp, _ = _foreground_binary_stats(logits, masks, self.threshold)
        precision = tp / (tp + fp + 1e-06)
        return precision.mean().item()

class ForegroundRecallMetric:

    def __init__(self, name: str='foreground_recall', threshold: float=0.5):
        self.name = name
        self.threshold = float(threshold)

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        tp, _, fn = _foreground_binary_stats(logits, masks, self.threshold)
        recall = tp / (tp + fn + 1e-06)
        return recall.mean().item()

class ForegroundF1Metric:

    def __init__(self, name: str='foreground_f1', threshold: float=0.5):
        self.name = name
        self.threshold = float(threshold)

    def __call__(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> float:
        tp, fp, fn = _foreground_binary_stats(logits, masks, self.threshold)
        precision = tp / (tp + fp + 1e-06)
        recall = tp / (tp + fn + 1e-06)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-06)
        return f1.mean().item()
