import inspect
from importlib import import_module

def _locate(target):
    (module_name, attr_name) = target.rsplit('.', 1)
    module = import_module(module_name)
    return getattr(module, attr_name)

def _filter_params(target, params: dict) -> dict:
    try:
        sig = inspect.signature(target)
    except (ValueError, TypeError):
        return params
    accepts_var_keyword = any((p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()))
    if accepts_var_keyword:
        return params
    accepted = set(sig.parameters.keys()) - {'self'}
    return {k: v for (k, v) in params.items() if k in accepted}

def instantiate(obj_cfg, **extra_kwargs):
    if isinstance(obj_cfg, list):
        return [instantiate(item) for item in obj_cfg]
    if isinstance(obj_cfg, dict) and '_target_' in obj_cfg:
        params = {k: v for (k, v) in obj_cfg.items() if k != '_target_'}
        params.update(extra_kwargs)
        target = _locate(obj_cfg['_target_'])
        params = _filter_params(target, params)
        return target(**params)
    if isinstance(obj_cfg, dict):
        return {k: instantiate(v) for (k, v) in obj_cfg.items()}
    return obj_cfg
