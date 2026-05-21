from abc import abstractmethod
import inspect
import torch
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from datasets.data_utils import inf_loop
from metrics.tracker import MetricTracker
from utils.io_utils import ROOT_PATH, resolve_path

class BaseTrainer:

    def __init__(self, model, criterion, metrics, optimizer, lr_scheduler, config, device, dataloaders, logger, writer, epoch_len=None, skip_oom=True, batch_transforms=None):
        self.is_train = True
        self.config = config
        self.cfg_trainer = self.config.trainer
        self.device = device
        self.skip_oom = skip_oom
        self.logger = logger
        self.log_step = config.trainer.get('log_step', 50)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.batch_transforms = batch_transforms
        self.grad_norm_every = int(self.cfg_trainer.get('grad_norm_every', self.log_step))
        if self.grad_norm_every <= 0:
            self.grad_norm_every = self.log_step
        self.train_dataloader = dataloaders['train']
        if epoch_len is None:
            self.epoch_len = len(self.train_dataloader)
        else:
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.epoch_len = epoch_len
        self.evaluation_dataloaders = {k: v for (k, v) in dataloaders.items() if k != 'train'}
        self._last_epoch = 0
        self.start_epoch = 1
        self.epochs = self.cfg_trainer.n_epochs
        self.save_period = self.cfg_trainer.save_period
        self.monitor = self.cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            (self.mnt_mode, self.mnt_metric) = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = self.cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf
        self.writer = writer
        self.metrics = metrics
        val_metric_funcs = self.metrics.get('val', self.metrics['inference'])
        self.train_metrics = MetricTracker(*self.config.writer.loss_names, 'grad_norm', *[m.name for m in self.metrics['train']], writer=self.writer)
        self.evaluation_metrics = MetricTracker(*self.config.writer.loss_names, *[m.name for m in val_metric_funcs], writer=self.writer)
        self._epoch_log_keys = list(self.cfg_trainer.get('log_keys', []))
        self.checkpoint_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
        if config.trainer.get('resume_from') is not None:
            resume_path = self.checkpoint_dir / config.trainer.resume_from
            self._resume_checkpoint(resume_path)
        if config.trainer.get('from_pretrained') is not None:
            self._from_pretrained(config.trainer.get('from_pretrained'))

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info('Saving model on keyboard interrupt')
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)
            logs = {'epoch': epoch}
            logs.update(result)
            for (key, value) in self._select_epoch_logs(logs).items():
                self.logger.info(f'    {key:15s}: {value}')
            (best, stop_process, not_improved_count) = self._monitor_performance(logs, not_improved_count)
            if best:
                self._save_checkpoint(epoch, save_best=True, only_best=True)
            if stop_process:
                break
        if self.cfg_trainer.get('save_last', True):
            self._save_last_checkpoint(self._last_epoch)

    def _train_epoch(self, epoch):
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar('epoch', epoch)
        last_train_metrics = self.train_metrics.result()
        for (batch_idx, batch) in enumerate(tqdm(self.train_dataloader, desc='train', total=self.epoch_len)):
            try:
                batch = self.process_batch(batch, metrics=self.train_metrics)
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning('OOM on batch. Skipping batch.')
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx % self.grad_norm_every == 0:
                self.train_metrics.update('grad_norm', self._get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.epoch_len + batch_idx)
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(epoch, self._progress(batch_idx), batch['loss'].item()))
                self.writer.add_scalar('learning rate', self.lr_scheduler.get_last_lr()[0])
                self._log_scalars(self.train_metrics)
                self._log_batch(batch_idx, batch)
                last_train_metrics = self.train_metrics.result()
            if batch_idx + 1 >= self.epoch_len:
                break
        logs = self.train_metrics.result() or last_train_metrics
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        eval_every = int(self.cfg_trainer.get('eval_every', 1))
        if eval_every <= 0:
            eval_every = 1
        if epoch % eval_every == 0:
            eval_parts = self.cfg_trainer.get('eval_parts', None)
            for (part, dataloader) in self.evaluation_dataloaders.items():
                if eval_parts is not None and part not in eval_parts:
                    continue
                val_logs = self._evaluation_epoch(epoch, part, dataloader)
                logs.update(**{f'{part}_{name}': value for (name, value) in val_logs.items()})
        else:
            self.logger.info(f'Skipping evaluation at epoch {epoch} (eval_every={eval_every}).')
        return logs

    def _evaluation_epoch(self, epoch, part, dataloader):
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for (batch_idx, batch) in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.process_batch(batch, metrics=self.evaluation_metrics)
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(batch_idx, batch, part)
        return self.evaluation_metrics.result()

    def _monitor_performance(self, logs, not_improved_count):
        best = False
        stop_process = False
        if self.mnt_mode != 'off':
            try:
                if self.mnt_mode == 'min':
                    improved = logs[self.mnt_metric] <= self.mnt_best
                elif self.mnt_mode == 'max':
                    improved = logs[self.mnt_metric] >= self.mnt_best
                else:
                    improved = False
            except KeyError:
                self.logger.warning(f"Warning: Metric '{self.mnt_metric}' is not found. Skipping performance monitoring for this epoch.")
                return (best, stop_process, not_improved_count)
            if improved:
                self.mnt_best = logs[self.mnt_metric]
                not_improved_count = 0
                best = True
            else:
                not_improved_count += 1
            if not_improved_count >= self.early_stop:
                self.logger.info("Validation performance didn't improve for {} epochs. Training stops.".format(self.early_stop))
                stop_process = True
        return (best, stop_process, not_improved_count)

    def move_batch_to_device(self, batch):
        for tensor_for_device in self.cfg_trainer.device_tensors:
            batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def transform_batch(self, batch):
        transform_type = 'train' if self.is_train else 'inference'
        transforms = self.batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                batch[transform_name] = transforms[transform_name](batch[transform_name])
        return batch

    def _model_forward(self, batch):
        signature = inspect.signature(self.model.forward)
        if any((p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())):
            return self.model(**batch)
        accepted = set(signature.parameters.keys())
        model_inputs = {k: v for (k, v) in batch.items() if k in accepted}
        return self.model(**model_inputs)

    def _clip_grad_norm(self):
        if self.config['trainer'].get('max_grad_norm', None) is not None:
            clip_grad_norm_(self.model.parameters(), self.config['trainer']['max_grad_norm'])

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)
        return total_norm.item()

    def _select_epoch_logs(self, logs):
        if self._epoch_log_keys:
            selected = {}
            if 'epoch' in logs:
                selected['epoch'] = logs['epoch']
            for key in self._epoch_log_keys:
                if key in logs:
                    value = logs[key]
                    if value is not None and (not isinstance(value, (int, float)) or value == value):
                        selected[key] = value
            if len(selected) > 1 or (len(selected) == 1 and 'epoch' not in selected):
                return selected
        selected = {}
        for (key, value) in logs.items():
            if key == 'epoch':
                selected[key] = value
                continue
            if value is None:
                continue
            if isinstance(value, (int, float)) and value != value:
                continue
            selected[key] = value
        return selected

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.train_dataloader, 'n_samples'):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.epoch_len
        return base.format(current, total, 100.0 * current / total)

    @abstractmethod
    def _log_batch(self, batch_idx, batch, mode='train'):
        return NotImplementedError()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            value = metric_tracker.avg(metric_name)
            if value is not None:
                self.writer.add_scalar(f'{metric_name}', value)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        arch = type(self.model).__name__
        state = {'arch': arch, 'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'lr_scheduler': self.lr_scheduler.state_dict(), 'monitor_best': self.mnt_best, 'config': self.config}
        filename = str(self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth')
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f'Saving checkpoint: {filename} ...')
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info('Saving current best: model_best.pth ...')

    def _save_last_checkpoint(self, epoch):
        arch = type(self.model).__name__
        state = {'arch': arch, 'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'lr_scheduler': self.lr_scheduler.state_dict(), 'monitor_best': self.mnt_best, 'config': self.config}
        last_path = str(self.checkpoint_dir / 'model_last.pth')
        torch.save(state, last_path)
        if self.config.writer.log_checkpoints:
            self.writer.add_checkpoint(last_path, str(self.checkpoint_dir.parent))
        self.logger.info('Saving last checkpoint: model_last.pth ...')

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info(f'Loading checkpoint: {resume_path} ...')
        checkpoint = self._load_checkpoint_file(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        if checkpoint['config']['model'] != self.config['model']:
            self.logger.warning('Warning: Architecture configuration given in the config file is different from that of the checkpoint. This may yield an exception when state_dict is loaded.')
        self.model.load_state_dict(checkpoint['state_dict'])
        if checkpoint['config']['optimizer'] != self.config['optimizer'] or checkpoint['config']['lr_scheduler'] != self.config['lr_scheduler']:
            self.logger.warning('Warning: Optimizer or lr_scheduler given in the config file is different from that of the checkpoint. Optimizer and scheduler parameters are not resumed.')
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.logger.info(f'Checkpoint loaded. Resume training from epoch {self.start_epoch}')

    def _from_pretrained(self, pretrained_path):
        pretrained_path = str(pretrained_path)
        if hasattr(self, 'logger'):
            self.logger.info(f'Loading model weights from: {pretrained_path} ...')
        else:
            print(f'Loading model weights from: {pretrained_path} ...')
        checkpoint = self._load_checkpoint_file(pretrained_path)
        if checkpoint.get('state_dict') is not None:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def _load_checkpoint_file(self, checkpoint_path):
        path = resolve_path(checkpoint_path, must_exist=True)
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)
