from importlib import import_module
from itertools import repeat

from torch.utils.data import DataLoader


def inf_loop(data_loader):
    """Wrapper for endless dataloader."""
    for loader in repeat(data_loader):
        yield from loader


def _build_dataset(dataset_cfg, split, shared_state=None):
    module = import_module(f"datasets.{dataset_cfg.module}")
    dataset_cls = getattr(module, dataset_cfg.name)

    params = dict(dataset_cfg.params)
    params["split"] = split

    if shared_state is not None:
        for key in ("word_to_idx", "answer_to_idx"):
            if key in shared_state and shared_state[key] is not None:
                params[key] = shared_state[key]

    dataset = dataset_cls(**params)
    return dataset


def get_dataloaders(config, device):
    del device  # reserved for future batch transforms
    dataset_cfg = config.dataset

    train_dataset = _build_dataset(dataset_cfg, "train")
    shared_state = {
        "word_to_idx": getattr(train_dataset, "word_to_idx", None),
        "answer_to_idx": getattr(train_dataset, "answer_to_idx", None),
    }
    val_dataset = _build_dataset(dataset_cfg, "val", shared_state=shared_state)
    test_dataset = _build_dataset(dataset_cfg, "test", shared_state=shared_state)

    batch_size = int(dataset_cfg.batch_size)
    eval_batch_size = int(getattr(dataset_cfg, "eval_batch_size", batch_size))
    num_workers = int(getattr(dataset_cfg, "num_workers", 0))
    pin_memory = bool(getattr(dataset_cfg, "pin_memory", False))

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=train_dataset.collate_batch,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=val_dataset.collate_batch,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=test_dataset.collate_batch,
        ),
    }

    return dataloaders, {"train": None, "inference": None}
