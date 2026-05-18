from metrics.tracker import MetricTracker
from trainer.base_trainer import BaseTrainer
import torch


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        amp_dtype_cfg = str(self.cfg_trainer.get("amp_dtype", "fp16")).lower()
        amp_dtype = torch.float16 if amp_dtype_cfg == "fp16" else torch.bfloat16
        self.use_amp = bool(self.cfg_trainer.get("use_amp", self.device == "cuda"))
        self.amp_dtype = amp_dtype
        self._amp_enabled = self.use_amp and str(self.device).startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self._amp_enabled)
            if self._amp_enabled
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with amp_ctx:
            outputs = self._model_forward(batch)
            if isinstance(outputs, dict):
                batch.update(outputs)
            else:
                batch["logits"] = outputs
            all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            if self._amp_enabled:
                self.scaler.scale(batch["loss"]).backward()
                if self.config["trainer"].get("max_grad_norm", None) is not None:
                    self.scaler.unscale_(self.optimizer)
                    self._clip_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch["loss"].backward()  # sum of all losses is always called loss
                self._clip_grad_norm()
                self.optimizer.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass