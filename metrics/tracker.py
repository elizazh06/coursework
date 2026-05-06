class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._keys = list(keys)
        self.reset()

    def reset(self):
        self._totals = {k: 0.0 for k in self._keys}
        self._counts = {k: 0 for k in self._keys}

    def update(self, key, value):
        if key not in self._totals:
            self._totals[key] = 0.0
            self._counts[key] = 0
            self._keys.append(key)
        self._totals[key] += float(value)
        self._counts[key] += 1

    def avg(self, key):
        count = self._counts.get(key, 0)
        if count == 0:
            return 0.0
        return self._totals[key] / count

    def result(self):
        return {k: self.avg(k) for k in self._keys}

    def keys(self):
        return self._keys
