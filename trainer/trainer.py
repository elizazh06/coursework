from metrics.tracker import MetricTracker
from trainer.base_trainer import BaseTrainer
import torch

class Trainer(BaseTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        amp_dtype_cfg = str(self.cfg_trainer.get('amp_dtype', 'fp16')).lower()
        amp_dtype = torch.float16 if amp_dtype_cfg == 'fp16' else torch.bfloat16
        self.use_amp = bool(self.cfg_trainer.get('use_amp', self.device == 'cuda'))
        self.amp_dtype = amp_dtype
        self._amp_enabled = self.use_amp and str(self.device).startswith('cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=self._amp_enabled)

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        metric_funcs = self.metrics['inference']
        if self.is_train:
            metric_funcs = self.metrics['train']
            self.optimizer.zero_grad(set_to_none=True)
        amp_ctx = torch.autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self._amp_enabled) if self._amp_enabled else torch.autocast(device_type='cpu', enabled=False)
        with amp_ctx:
            outputs = self._model_forward(batch)
            if isinstance(outputs, dict):
                batch.update(outputs)
            else:
                batch['logits'] = outputs
            all_losses = self.criterion(**batch)
        batch.update(all_losses)
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
        return batch

    def _log_batch(self, batch_idx, batch, mode='train'):
        if mode == 'train':
            pass
        else:
            pass
