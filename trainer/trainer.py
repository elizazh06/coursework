from metrics.tracker import MetricTracker
from trainer.base_trainer import BaseTrainer
import torch
from contextlib import nullcontext

class Trainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        amp_dtype_cfg = str(self.cfg_trainer.get('amp_dtype', 'fp16')).lower()
        amp_dtype = torch.float16 if amp_dtype_cfg == 'fp16' else torch.bfloat16
        self.use_amp = bool(self.cfg_trainer.get('use_amp', self.device == 'cuda'))
        self.amp_dtype = amp_dtype
        self._amp_enabled = self.use_amp and str(self.device).startswith('cuda')
        # Torch AMP API differs across versions:
        # - newer: torch.amp.GradScaler(device, ...)
        # - older: torch.cuda.amp.GradScaler(...)
        self.scaler = self._build_grad_scaler()
        self._last_batch_debug = {}

    def _record_batch_debug(self, batch):
        logits = batch.get('logits')
        if logits is None:
            return
        logits_f = logits.detach().float()
        finite = torch.isfinite(logits_f)
        self._last_batch_debug = {
            'loss': float(batch['loss'].item()) if 'loss' in batch and torch.isfinite(batch['loss']) else None,
            'logit_mean': float(logits_f[finite].mean().item()) if finite.any() else None,
            'logit_min': float(logits_f[finite].min().item()) if finite.any() else None,
            'logit_max': float(logits_f[finite].max().item()) if finite.any() else None,
            'pred_pos_rate': float((torch.sigmoid(logits_f) >= 0.5).float().mean().item()) if finite.any() else None,
        }

    def _build_grad_scaler(self):
        if not self._amp_enabled:
            return None
        try:
            return torch.amp.GradScaler(device='cuda', enabled=True)
        except Exception:
            try:
                return torch.cuda.amp.GradScaler(enabled=True)
            except Exception:
                # If scaler is unavailable in this torch build, fallback to fp32 path.
                self._amp_enabled = False
                return None

    def _autocast_context(self):
        if not self._amp_enabled:
            return nullcontext()
        try:
            return torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=True)
        except Exception:
            try:
                return torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=True)
            except Exception:
                return nullcontext()

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        metric_funcs = self.metrics['inference']
        if self.is_train:
            metric_funcs = self.metrics['train']
            self.optimizer.zero_grad(set_to_none=True)
        amp_ctx = self._autocast_context()
        with amp_ctx:
            outputs = self._model_forward(batch)
            if isinstance(outputs, dict):
                batch.update(outputs)
            else:
                batch['logits'] = outputs
        logits = batch.get('logits')
        if logits is None:
            raise RuntimeError('Model forward did not produce logits.')
        if not torch.isfinite(logits).all():
            self.logger.warning('Non-finite logits in batch; skipping loss/metrics for this batch.')
            return batch
        all_losses = self.criterion(logits=logits.float(), masks=batch['masks'].float())
        batch.update(all_losses)
        if self.is_train and not torch.isfinite(batch['loss']):
            self.logger.warning(f"Non-finite loss ({batch['loss'].item()}); skipping optimizer step.")
            self.optimizer.zero_grad(set_to_none=True)
            return batch
        if self.is_train:
            if self._amp_enabled:
                self.scaler.scale(batch['loss']).backward()
                if self.config['trainer'].get('max_grad_norm', None) is not None:
                    self.scaler.unscale_(self.optimizer)
                    self._clip_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch['loss'].backward()
                self._clip_grad_norm()
                self.optimizer.step()
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        self._record_batch_debug(batch)
        return batch

    def _log_batch(self, batch_idx, batch, mode='train'):
        if mode == 'train':
            pass
        else:
            pass
