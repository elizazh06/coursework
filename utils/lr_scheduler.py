import math

from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineLR(LambdaLR):

    def __init__(
        self,
        optimizer,
        max_epochs: int,
        warmup_epochs: int = 0,
        min_lr: float = 0.0,
        warmup_start_factor: float = 0.1,
        last_epoch: int = -1,
    ):
        self.max_epochs = max(1, int(max_epochs))
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.min_lr = float(min_lr)
        self.warmup_start_factor = float(warmup_start_factor)
        lr_lambdas = [self._make_lr_lambda(group['lr']) for group in optimizer.param_groups]
        super().__init__(optimizer, lr_lambdas, last_epoch=last_epoch)

    def _make_lr_lambda(self, base_lr: float):
        min_factor = min(1.0, max(0.0, self.min_lr / base_lr)) if base_lr > 0 else 0.0

        def lr_lambda(epoch: int) -> float:
            if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                progress = float(epoch + 1) / float(self.warmup_epochs)
                return self.warmup_start_factor + progress * (1.0 - self.warmup_start_factor)

            decay_epochs = max(1, self.max_epochs - self.warmup_epochs)
            progress = min(1.0, max(0.0, (epoch - self.warmup_epochs) / decay_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return lr_lambda
