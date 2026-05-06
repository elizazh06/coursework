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
