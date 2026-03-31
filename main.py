import yaml
from pipelines.avsu_pipeline import AVSUPipeline

if __name__ == "__main__":
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    pipeline = AVSUPipeline(config)
    pipeline.run()