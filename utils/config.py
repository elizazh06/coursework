class ConfigNode(dict):
    """Dict with attribute access and recursive conversion."""

    def __init__(self, mapping=None, **kwargs):
        super().__init__()
        mapping = mapping or {}
        mapping.update(kwargs)
        for key, value in mapping.items():
            self[key] = self._convert(value)

    def _convert(self, value):
        if isinstance(value, dict):
            return ConfigNode(value)
        if isinstance(value, list):
            return [self._convert(v) for v in value]
        return value

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = self._convert(value)
