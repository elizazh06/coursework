import warnings
import argparse

import torch
import yaml

from datasets.data_utils import get_dataloaders
from trainer.inferencer import Inferencer
from utils.config import ConfigNode
from utils.factory import instantiate
from utils.init_utils import set_random_seed
from utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


def main(config_path):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = ConfigNode(yaml.safe_load(f))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference.yaml")
    args = parser.parse_args()
    main(args.config)