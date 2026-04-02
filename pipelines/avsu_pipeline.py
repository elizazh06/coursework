import torch
from torch.utils.data import DataLoader
import importlib
from tqdm.notebook import tqdm
import time


class AVSUPipeline:
    def __init__(self, config):
        self.config = config
        self.device = config["training"].get("device", "cpu")

        dataset_module = importlib.import_module(
            f"datasets.{config['dataset']['module']}"
        )
        dataset_class = getattr(dataset_module, config["dataset"]["name"])

        self.dataset = dataset_class(**config["dataset"]["params"])
        config["model"]["params"]["num_classes"] = self.dataset.num_classes

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config["dataset"].get("batch_size", 4),
            shuffle=True,
            collate_fn=self.collate_fn
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

        if len(batch[0]) == 3:
            video, audio, label = zip(*batch)
            return (
                torch.stack(video),
                torch.stack(audio),
                torch.tensor(label)
            )

        elif len(batch[0]) == 4:
            video, audio, question, label = zip(*batch)

            # padding для question
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
        if len(batch) == 3:
            video, audio, labels = batch
            return self.model(video, audio), labels

        elif len(batch) == 4:
            video, audio, question, labels = batch
            return self.model(video, audio, question), labels

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        start_time = time.time()

        progress_bar = tqdm(
            self.dataloader,
            desc=f"Epoch {self.current_epoch}",
            leave=False
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

        return total_loss / len(self.dataloader)

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.dataloader:
                batch = [x.to(self.device) for x in batch]

                outputs, labels = self.forward_pass(batch)

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total

    def run(self):
        epochs = self.config["training"]["epochs"]

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            start = time.time()
            loss = self.train_epoch()
            end = time.time()

            print(f"Loss: {loss:.4f}")
            print(f"Epoch duration: {(end - start):.2f} sec")