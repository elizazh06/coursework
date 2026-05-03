import argparse
from pathlib import Path

import yaml

from config_paths import resolve_config_paths
from pipelines.avsu_pipeline import AVSUPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    config_path = str(Path(args.config).expanduser())
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resolve_config_paths(config, config_path)

    pipeline = AVSUPipeline(config)
    pipeline.run()