import argparse
import yaml
from pipelines.avsu_pipeline import AVSUPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    pipeline = AVSUPipeline(config)
    pipeline.run()