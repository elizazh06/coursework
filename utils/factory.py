from importlib import import_module


def _locate(target):
    module_name, attr_name = target.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def instantiate(obj_cfg, **extra_kwargs):
    if isinstance(obj_cfg, list):
        return [instantiate(item) for item in obj_cfg]

    if isinstance(obj_cfg, dict) and "_target_" in obj_cfg:
        params = {k: v for k, v in obj_cfg.items() if k != "_target_"}
        params.update(extra_kwargs)
        target = _locate(obj_cfg["_target_"])
        return target(**params)

    if isinstance(obj_cfg, dict):
        return {k: instantiate(v) for k, v in obj_cfg.items()}

    return obj_cfg
