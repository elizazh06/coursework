import math


class MetricTracker:

    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._keys = list(keys)
        self.reset()

    def reset(self):
        self._totals = {k: 0.0 for k in self._keys}
        self._counts = {k: 0 for k in self._keys}

    def update(self, key, value):
        value = float(value)
        if not math.isfinite(value):
            return
        if key not in self._totals:
            self._totals[key] = 0.0
            self._counts[key] = 0
            self._keys.append(key)
        self._totals[key] += value
        self._counts[key] += 1

    def avg(self, key):
        count = self._counts.get(key, 0)
        if count == 0:
            return None
        avg = self._totals[key] / count
        return avg if math.isfinite(avg) else None

    def result(self):
        out = {}
        for k in self._keys:
            if self._counts.get(k, 0) <= 0:
                continue
            avg = self.avg(k)
            if avg is not None:
                out[k] = avg
        return out

    def keys(self):
        return self._keys
