from typing import Union

from metrics.composites import ClassificationCompositeMetric, MusicAVQACompositeMetric


def build_eval_metric(
    dataset_module: str,
) -> Union[MusicAVQACompositeMetric, ClassificationCompositeMetric]:
    if dataset_module == "music_avqa":
        return MusicAVQACompositeMetric()
    return ClassificationCompositeMetric()
