from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def __init__(self, name=None):
        self.name = name if name is not None else type(self).__name__

    def reset(self):
        pass

    @abstractmethod
    def update(self, **batch):
        pass

    @abstractmethod
    def compute(self, **kwargs):
        pass
