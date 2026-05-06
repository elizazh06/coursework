import torch

from metrics.base_metric import BaseMetric
from metrics import functional as Fm


class MusicAVQACompositeMetric(BaseMetric):
    def __init__(self):
        super().__init__("music_avqa")
        self._p, self._y, self._types = [], [], []

    def reset(self):
        self._p.clear()
        self._y.clear()
        self._types.clear()

    def update(self, logits=None, labels=None, valid_mask=None, raw_batch=None, **_):
        if labels is None or logits is None or labels.numel() == 0:
            return
        self._p.append(logits.argmax(dim=1).detach().cpu())
        self._y.append(labels.detach().cpu())
        if raw_batch is not None and valid_mask is not None:
            vm = valid_mask.detach().cpu().tolist()
            qt = raw_batch.get("question_type", [])
            self._types.extend(t for t, k in zip(qt, vm) if k)

    def compute(self, forward_pass=None, dataloader=None, num_classes=42, **_):
        if not self._y:
            return {"accuracy": 0.0, "macro_f1": 0.0, "top3_acc": 0.0, "by_type": {}}
        preds = torch.cat(self._p)
        labels = torch.cat(self._y)
        if preds.numel() != labels.numel():
            n = min(preds.numel(), labels.numel())
            preds, labels = preds[:n], labels[:n]
            types = self._types[:n]
        else:
            types = self._types
        out = {
            "accuracy": Fm.accuracy(preds, labels),
            "macro_f1": Fm.macro_f1_multiclass(preds, labels, num_classes),
            "by_type": Fm.by_question_type(preds, labels, types),
        }
        if forward_pass is not None and dataloader is not None:
            th, tt = Fm.top3_hits(forward_pass, dataloader)
            out["top3_acc"] = (th / tt) if tt else 0.0
        else:
            out["top3_acc"] = 0.0
        return out


class ClassificationCompositeMetric(BaseMetric):
    def __init__(self):
        super().__init__("classification")
        self._p, self._y, self._logits = [], [], []

    def reset(self):
        self._p.clear()
        self._y.clear()
        self._logits.clear()

    def update(self, logits=None, labels=None, **_):
        if labels is None or logits is None or labels.numel() == 0:
            return
        self._p.append(logits.argmax(dim=1).detach().cpu())
        self._y.append(labels.detach().cpu())
        self._logits.append(logits.detach().cpu())

    def compute(self, num_classes=42, **_):
        if not self._y:
            return {"accuracy": 0.0, "macro_f1": 0.0, "top3_acc": 0.0}
        preds = torch.cat(self._p)
        labels = torch.cat(self._y)
        logits = torch.cat(self._logits)
        if preds.numel() != labels.numel():
            n = min(preds.numel(), labels.numel())
            preds, labels = preds[:n], labels[:n]
            logits = logits[:n]
        k = min(3, logits.size(1))
        topk = logits.topk(k=k, dim=1).indices
        return {
            "accuracy": Fm.accuracy(preds, labels),
            "macro_f1": Fm.macro_f1_multiclass(preds, labels, num_classes),
            "top3_acc": (topk == labels.unsqueeze(1)).any(dim=1).float().mean().item(),
        }
