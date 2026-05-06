import warnings
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from datasets.data_utils import get_dataloaders
from trainer.inferencer import Inferencer
from utils.config import ConfigNode
from utils.config_loader import apply_dotlist_overrides, load_composed_config
from utils.factory import instantiate
from utils.init_utils import set_random_seed
from utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


def main(config_path, overrides=None):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    raw_config = load_composed_config(config_path)
    raw_config = apply_dotlist_overrides(
        raw_config,
        overrides or [],
        config_dir=Path(config_path).parent,
    )
    config = ConfigNode(raw_config)

    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)

    # get metrics
    metrics = instantiate(config.metrics)

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")

    metrics_filename = config.inferencer.get("metrics_filename", "metrics.json")
    metrics_path = save_path / metrics_filename
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    _append_results_row(config, logs, save_path)


def _append_results_row(config, logs, save_path):
    dataset_name = str(config.dataset.module)
    model_target = str(config.model.get("_target_", "unknown_model"))
    checkpoint_path = str(config.inferencer.get("from_pretrained", ""))

    row = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds"),
        "dataset": dataset_name,
        "model": model_target,
        "checkpoint": checkpoint_path,
        "save_path": str(save_path),
    }

    for part, part_metrics in logs.items():
        for metric_name, value in part_metrics.items():
            row[f"{part}_{metric_name}"] = value

    results_dir = ROOT_PATH / "data" / "saved" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    table_path = results_dir / f"{dataset_name}_results.csv"

    existing_rows = []
    if table_path.exists():
        with table_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    all_rows = existing_rows + [row]
    base_cols = ["timestamp_utc", "dataset", "model", "checkpoint", "save_path"]
    dynamic_cols = sorted(
        {key for r in all_rows for key in r.keys() if key not in base_cols}
    )
    fieldnames = base_cols + dynamic_cols

    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Updated results table: {table_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    args, unknown = parser.parse_known_args()
    main(args.config, overrides=unknown)