class AccuracyMetric:
    def __init__(self, name="accuracy"):
        self.name = name

    def __call__(self, logits, labels=None, label=None, **kwargs):
        del kwargs
        targets = labels if labels is not None else label
        if targets is None:
            return 0.0
        preds = logits.argmax(dim=1)
        return (preds == targets).float().mean().item()


class TopKAccuracyMetric:
    def __init__(self, k=3, name=None):
        self.k = int(k)
        self.name = name or f"top{self.k}_accuracy"

    def __call__(self, logits, labels=None, label=None, **kwargs):
        del kwargs
        targets = labels if labels is not None else label
        if targets is None:
            return 0.0
        k = min(self.k, int(logits.size(1)))
        topk = logits.topk(k=k, dim=1).indices
        return (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()


def _macro_prf_from_batch(logits, targets):
    preds = logits.argmax(dim=1)
    num_classes = int(logits.size(1))
    eps = 1e-8

    precisions = []
    recalls = []
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()

        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_p = float(sum(precisions) / len(precisions)) if precisions else 0.0
    macro_r = float(sum(recalls) / len(recalls)) if recalls else 0.0
    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0
    return macro_p, macro_r, macro_f1


class MacroPrecisionMetric:
    def __init__(self, name="macro_precision"):
        self.name = name

    def __call__(self, logits, labels=None, label=None, **kwargs):
        del kwargs
        targets = labels if labels is not None else label
        if targets is None:
            return 0.0
        macro_p, _, _ = _macro_prf_from_batch(logits, targets)
        return macro_p


class MacroRecallMetric:
    def __init__(self, name="macro_recall"):
        self.name = name

    def __call__(self, logits, labels=None, label=None, **kwargs):
        del kwargs
        targets = labels if labels is not None else label
        if targets is None:
            return 0.0
        _, macro_r, _ = _macro_prf_from_batch(logits, targets)
        return macro_r


class MacroF1Metric:
    def __init__(self, name="macro_f1"):
        self.name = name

    def __call__(self, logits, labels=None, label=None, **kwargs):
        del kwargs
        targets = labels if labels is not None else label
        if targets is None:
            return 0.0
        _, _, macro_f1 = _macro_prf_from_batch(logits, targets)
        return macro_f1


class BalancedAccuracyMetric:
    def __init__(self, name="balanced_accuracy"):
        self.name = name

    def __call__(self, logits, labels=None, label=None, **kwargs):
        del kwargs
        targets = labels if labels is not None else label
        if targets is None:
            return 0.0
        preds = logits.argmax(dim=1)
        num_classes = int(logits.size(1))
        eps = 1e-8
        recalls = []
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().item()
            fn = ((preds != c) & (targets == c)).sum().item()
            recalls.append(tp / (tp + fn + eps))
        return float(sum(recalls) / len(recalls)) if recalls else 0.0
