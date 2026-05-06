import torch.nn as nn


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels=None, label=None, **kwargs):
        del kwargs
        targets = labels if labels is not None else label
        if targets is None:
            raise ValueError("Batch should contain 'label' or 'labels'.")
        return {"loss": self.loss(logits, targets)}
