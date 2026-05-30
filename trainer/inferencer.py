import torch
from tqdm.auto import tqdm
from metrics.tracker import MetricTracker
from trainer.base_trainer import BaseTrainer
from utils.segmentation_utils import align_logits_to_masks

class Inferencer(BaseTrainer):

    def __init__(self, model, config, device, dataloaders, save_path, metrics=None, batch_transforms=None, skip_model_load=False, criterion=None):
        assert skip_model_load or config.inferencer.get('from_pretrained') is not None, 'Provide checkpoint or set skip_model_load=True'
        self.config = config
        self.cfg_trainer = self.config.inferencer
        self.device = device
        self.model = model
        self.batch_transforms = batch_transforms
        self.criterion = criterion
        requested_parts = self.cfg_trainer.get('parts', ['test'])
        self.evaluation_dataloaders = {k: v for (k, v) in dataloaders.items() if k in requested_parts}
        if not self.evaluation_dataloaders:
            self.evaluation_dataloaders = {'test': dataloaders['test']}
        self.save_path = save_path
        self.save_predictions = bool(self.cfg_trainer.get('save_predictions', False))
        self.metrics = metrics
        self.loss_names = list(self.cfg_trainer.get('loss_names', []))
        if self.metrics is not None:
            tracker_keys = [m.name for m in self.metrics['inference']]
            if self.criterion is not None:
                tracker_keys.extend(self.loss_names or ['loss', 'bce_loss', 'dice_loss'])
            self.evaluation_metrics = MetricTracker(*tracker_keys, writer=None)
        else:
            self.evaluation_metrics = None
        if not skip_model_load:
            self._from_pretrained(config.inferencer.get('from_pretrained'))

    def run_inference(self):
        part_logs = {}
        for (part, dataloader) in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def _prepare_batch_for_metrics(self, batch):
        masks = batch.get('masks')
        logits = batch.get('logits')
        if masks is not None and logits is not None and logits.dim() >= 3:
            batch['logits'] = align_logits_to_masks(logits, masks)
        return batch

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)
        outputs = self._model_forward(batch)
        if isinstance(outputs, dict):
            batch.update(outputs)
        else:
            batch['logits'] = outputs
        batch = self._prepare_batch_for_metrics(batch)
        if metrics is not None:
            logits = batch.get('logits')
            masks = batch.get('masks')
            if self.criterion is not None and logits is not None and masks is not None:
                if torch.isfinite(logits).all():
                    loss_values = self.criterion(logits=logits.float(), masks=masks.float())
                    for loss_name in self.loss_names or ['loss', 'bce_loss', 'dice_loss']:
                        if loss_name in loss_values:
                            metrics.update(loss_name, loss_values[loss_name].item())
            for met in self.metrics['inference']:
                metrics.update(met.name, met(**batch))
        if self.save_predictions and self.save_path is not None:
            self._save_batch_predictions(batch_idx, batch, part)
        return batch

    def _save_batch_predictions(self, batch_idx, batch, part):
        logits = batch['logits']
        masks = batch.get('masks')
        label_tensor = batch.get('labels', batch.get('label'))
        if masks is not None:
            batch_size = masks.shape[0]
        else:
            batch_size = logits.shape[0]
        current_id = batch_idx * batch_size
        for i in range(batch_size):
            sample_logits = logits[i].clone()
            if masks is not None:
                label = masks[i].clone()
                pred_label = torch.sigmoid(sample_logits) >= 0.5
            else:
                label = label_tensor[i].clone()
                pred_label = sample_logits.argmax(dim=-1)
            output_id = current_id + i
            output = {'pred_label': pred_label, 'label': label}
            torch.save(output, self.save_path / part / f'output_{output_id}.pth')

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        if self.save_predictions and self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)
        with torch.no_grad():
            for (batch_idx, batch) in tqdm(enumerate(dataloader), desc=part, total=len(dataloader)):
                batch = self.process_batch(batch_idx=batch_idx, batch=batch, part=part, metrics=self.evaluation_metrics)
        return self.evaluation_metrics.result()
