import importlib
import time
import ast
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class AVSUPipeline:
    def __init__(self, config):
        self.config = config
        requested_device = config["training"].get("device", "cpu")
        self.device = (
            requested_device
            if requested_device == "cpu" or torch.cuda.is_available()
            else "cpu"
        )

        dataset_module = importlib.import_module(
            f"datasets.{config['dataset']['module']}"
        )
        dataset_class = getattr(dataset_module, config["dataset"]["name"])

        dataset_params = dict(config["dataset"]["params"])
        train_params = dict(dataset_params)
        train_params["split"] = config["dataset"].get("train_split", "train")
        self.train_dataset = dataset_class(**train_params)

        val_params = dict(dataset_params)
        val_params["split"] = config["dataset"].get("val_split", "val")
        val_params["answer_to_idx"] = self.train_dataset.answer_to_idx
        val_params["word_to_idx"] = self.train_dataset.word_to_idx
        self.val_dataset = dataset_class(**val_params)

        test_params = dict(dataset_params)
        test_params["split"] = config["dataset"].get("test_split", "test")
        test_params["answer_to_idx"] = self.train_dataset.answer_to_idx
        test_params["word_to_idx"] = self.train_dataset.word_to_idx
        self.test_dataset = dataset_class(**test_params)

        config["model"]["params"]["num_classes"] = self.train_dataset.num_classes

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=config["dataset"].get("batch_size", 4),
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=config["dataset"].get("num_workers", 0),
            pin_memory=self.device != "cpu",
        )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=config["dataset"].get("eval_batch_size", config["dataset"].get("batch_size", 4)),
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=config["dataset"].get("num_workers", 0),
            pin_memory=self.device != "cpu",
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=config["dataset"].get("eval_batch_size", config["dataset"].get("batch_size", 4)),
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=config["dataset"].get("num_workers", 0),
            pin_memory=self.device != "cpu",
        )

        model_module = importlib.import_module(
            f"models.{config['model']['module']}"
        )
        model_class = getattr(model_module, config["model"]["name"])

        self.model = model_class(**config["model"]["params"]).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config["training"].get("lr", 1e-4)),
            weight_decay=float(config["training"].get("weight_decay", 1e-5)),
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=int(config["training"].get("lr_step_size", 8)),
            gamma=float(config["training"].get("lr_gamma", 0.1)),
        )
        self.grad_clip_norm = float(config["training"].get("grad_clip_norm", 1.0))
        self.scaler = torch.amp.GradScaler(enabled=(self.device != "cpu"))

    def collate_fn(self, batch):
        if isinstance(batch[0], dict):
            labels = torch.stack([sample["label"] for sample in batch]).long()
            video = torch.stack([sample["video"] for sample in batch])
            audio = torch.stack([sample["audio"] for sample in batch])
            question = [sample["question"] for sample in batch]
            question_id = [sample.get("question_id", -1) for sample in batch]
            question_type = [sample.get("question_type", "[]") for sample in batch]
            video_id = [sample.get("video_id", "") for sample in batch]

            max_len = max(q.size(0) for q in question)
            padded_q = []
            for q in question:
                pad = max_len - q.size(0)
                padded_q.append(torch.cat([q, torch.zeros(pad, dtype=torch.long)]))

            return {
                "video": video,
                "audio": audio,
                "question": torch.stack(padded_q),
                "label": labels,
                "question_id": question_id,
                "question_type": question_type,
                "video_id": video_id,
            }

        if len(batch[0]) == 3:
            video, audio, label = zip(*batch)
            return (
                torch.stack(video),
                torch.stack(audio),
                torch.tensor(label)
            )

        elif len(batch[0]) == 4:
            video, audio, question, label = zip(*batch)

            max_len = max(q.size(0) for q in question)
            padded_q = []
            for q in question:
                pad = max_len - q.size(0)
                padded_q.append(torch.cat([q, torch.zeros(pad, dtype=torch.long)]))

            return (
                torch.stack(video),
                torch.stack(audio),
                torch.stack(padded_q),
                torch.tensor(label)
            )

        else:
            raise ValueError("Unsupported dataset format")

    def forward_pass(self, batch):
        if isinstance(batch, dict):
            video = batch["video"].to(self.device)
            audio = batch["audio"].to(self.device)
            question = batch["question"].to(self.device)
            labels = batch["label"].to(self.device)
            outputs = self.model(video, audio, question)
            valid_mask = labels >= 0
            return outputs[valid_mask], labels[valid_mask], valid_mask

        if len(batch) == 3:
            video, audio, labels = batch
            video = video.to(self.device)
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            return self.model(video, audio), labels

        elif len(batch) == 4:
            video, audio, question, labels = batch
            video = video.to(self.device)
            audio = audio.to(self.device)
            question = question.to(self.device)
            labels = labels.to(self.device)
            return self.model(video, audio, question), labels

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        start_time = time.time()

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            leave=False,
        )

        for batch in progress_bar:
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=(self.device != "cpu")):
                outputs, labels, _ = self.forward_pass(batch)
                if labels.numel() == 0:
                    continue
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

        epoch_time = time.time() - start_time

        print(f"Epoch time: {epoch_time:.2f} sec")

        return total_loss / len(self.train_dataloader)

    def _compute_macro_f1(self, preds, labels, num_classes):
        eps = 1e-8
        f1_scores = []
        for cls in range(num_classes):
            tp = ((preds == cls) & (labels == cls)).sum().item()
            fp = ((preds == cls) & (labels != cls)).sum().item()
            fn = ((preds != cls) & (labels == cls)).sum().item()
            if tp == 0 and fp == 0 and fn == 0:
                continue
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            f1_scores.append(f1)
        return float(sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0

    def evaluate(self, dataloader, split_name="val"):
        self.model.eval()
        all_preds = []
        all_labels = []
        type_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        with torch.no_grad():
            for batch in dataloader:
                outputs, labels, valid_mask = self.forward_pass(batch)
                if labels.numel() == 0:
                    continue

                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

                filtered_types = [
                    t for t, keep in zip(batch.get("question_type", []), valid_mask.cpu().tolist()) if keep
                ]
                for pred, label, type_str in zip(preds.cpu(), labels.cpu(), filtered_types):
                    try:
                        parsed = ast.literal_eval(type_str) if isinstance(type_str, str) else type_str
                        coarse = parsed[0] if len(parsed) > 0 else "Unknown"
                        fine = parsed[1] if len(parsed) > 1 else "Unknown"
                    except (ValueError, SyntaxError):
                        coarse, fine = "Unknown", "Unknown"

                    key = f"{coarse}/{fine}"
                    type_stats[key]["total"] += 1
                    type_stats[coarse]["total"] += 1
                    if pred.item() == label.item():
                        type_stats[key]["correct"] += 1
                        type_stats[coarse]["correct"] += 1

        if not all_labels:
            return {"accuracy": 0.0, "macro_f1": 0.0, "top3_acc": 0.0, "by_type": {}}

        labels = torch.cat(all_labels)
        preds = torch.cat(all_preds)
        if preds.numel() != labels.numel():
            min_len = min(preds.numel(), labels.numel())
            preds = preds[:min_len]
            labels = labels[:min_len]
        accuracy = (preds == labels).float().mean().item()

        top3_correct = 0
        total = 0
        with torch.no_grad():
            for batch in dataloader:
                outputs, labels_batch, _ = self.forward_pass(batch)
                if labels_batch.numel() == 0:
                    continue
                topk = outputs.topk(k=min(3, outputs.size(1)), dim=1).indices
                top3_correct += (topk == labels_batch.unsqueeze(1)).any(dim=1).sum().item()
                total += labels_batch.size(0)

        by_type = {
            key: (stats["correct"] / stats["total"]) if stats["total"] > 0 else 0.0
            for key, stats in sorted(type_stats.items())
        }
        metrics = {
            "accuracy": accuracy,
            "macro_f1": self._compute_macro_f1(preds, labels, self.train_dataset.num_classes),
            "top3_acc": (top3_correct / total) if total > 0 else 0.0,
            "by_type": by_type,
        }
        print(
            f"{split_name.upper()} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"top3={metrics['top3_acc']:.4f}"
        )
        return metrics

    def run(self):
        epochs = self.config["training"]["epochs"]
        self.epoch = 0

        best_val_acc = -1.0
        best_state = None

        while self.epoch < epochs:
            print(f"\nEpoch {self.epoch+1}/{epochs}")

            start = time.time()
            loss = self.train_epoch()
            end = time.time()
            self.scheduler.step()
            val_metrics = self.evaluate(self.val_dataloader, split_name="val")
            val_acc = val_metrics["accuracy"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            print(f"Loss: {loss:.4f}")
            print(f"Val acc: {val_acc:.4f}")
            print(f"Epoch duration: {(end - start):.2f} sec")
            self.epoch += 1

        if best_state is not None:
            self.model.load_state_dict(best_state)
        test_metrics = self.evaluate(self.test_dataloader, split_name="test")

        print("\nFinal test metrics:")
        print(f"- accuracy: {test_metrics['accuracy']:.4f}")
        print(f"- macro_f1: {test_metrics['macro_f1']:.4f}")
        print(f"- top3_acc: {test_metrics['top3_acc']:.4f}")
        for k, v in test_metrics["by_type"].items():
            print(f"- {k}: {v:.4f}")