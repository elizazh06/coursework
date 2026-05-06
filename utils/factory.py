from importlib import import_module


def _locate(target):
    module_name, attr_name = target.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def instantiate(config, **extra_kwargs):
    if isinstance(config, list):
        return [instantiate(item) for item in config]

    if isinstance(config, dict) and "_target_" in config:
        params = {k: v for k, v in config.items() if k != "_target_"}
        params.update(extra_kwargs)
        target = _locate(config["_target_"])
        return target(**params)

    if isinstance(config, dict):
        return {k: instantiate(v) for k, v in config.items()}

    return config
