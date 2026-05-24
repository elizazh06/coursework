import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.segmentation_utils import flatten_logits_and_masks

def _dice_loss(probs: torch.Tensor, targets: torch.Tensor, eps: float=1.0) -> torch.Tensor:
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    return 1.0 - (2.0 * intersection + eps) / (denom + eps)

class SegmentationLoss(nn.Module):

    def __init__(self, bce_weight: float=1.0, dice_weight: float=1.0, pos_weight: float | None=None):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        pw = torch.tensor([float(pos_weight)]) if pos_weight is not None else None
        self.register_buffer('pos_weight', pw)

    def forward(self, logits: torch.Tensor, masks: torch.Tensor, **_) -> dict:
        logits, masks = flatten_logits_and_masks(logits, masks)
        targets = masks.float()
        if logits.shape[-2:] != targets.shape[-2:]:
            targets = F.interpolate(targets.unsqueeze(1), size=logits.shape[-2:], mode='nearest').squeeze(1)
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction='mean')
        probs = torch.sigmoid(logits)
        dice = _dice_loss(probs, targets).mean()
        loss = self.bce_weight * bce + self.dice_weight * dice
        return {'loss': loss, 'bce_loss': bce.detach(), 'dice_loss': dice.detach()}
