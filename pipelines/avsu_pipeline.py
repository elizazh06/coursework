import torch
from torch.utils.data import DataLoader
import importlib
from tqdm import tqdm
import time


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
        self.val_dataset = dataset_class(**val_params)

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

        model_module = importlib.import_module(
            f"models.{config['model']['module']}"
        )
        model_class = getattr(model_module, config["model"]["name"])

        self.model = model_class(**config["model"]["params"]).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config["training"].get("lr", 1e-4))
        )

    def collate_fn(self, batch):
        if isinstance(batch[0], dict):
            labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
            video = torch.stack([sample["video"] for sample in batch])
            audio = torch.stack([sample["audio"] for sample in batch])
            question = [sample["question"] for sample in batch]

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
            return self.model(video, audio, question), labels

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

            outputs, labels = self.forward_pass(batch)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

        epoch_time = time.time() - start_time

        print(f"Epoch time: {epoch_time:.2f} sec")

        return total_loss / len(self.train_dataloader)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs, labels = self.forward_pass(batch)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def run(self):
        epochs = self.config["training"]["epochs"]
        self.epoch = 0

        while self.epoch < epochs:
            print(f"\nEpoch {self.epoch+1}/{epochs}")

            start = time.time()
            loss = self.train_epoch()
            end = time.time()
            val_acc = self.evaluate()

            print(f"Loss: {loss:.4f}")
            print(f"Val acc: {val_acc:.4f}")
            print(f"Epoch duration: {(end - start):.2f} sec")
            self.epoch += 1