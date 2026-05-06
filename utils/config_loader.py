from copy import deepcopy
from pathlib import Path

import yaml


def _deep_merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_composed_config(config_path):
    config_path = Path(config_path)
    root = _load_yaml(config_path)
    config_dir = config_path.parent

    defaults = root.pop("defaults", [])
    composed = {}

    for item in defaults:
        if item == "_self_":
            continue
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(f"Unsupported defaults entry: {item}")
        group, name = next(iter(item.items()))
        sub_path = config_dir / group / f"{name}.yaml"
        if not sub_path.exists():
            raise FileNotFoundError(f"Config group file not found: {sub_path}")
        composed = _deep_merge(composed, _load_yaml(sub_path))

    composed = _deep_merge(composed, root)
    return composed


def apply_dotlist_overrides(config_dict, overrides, config_dir=None):
    result = deepcopy(config_dict)
    config_dir = Path(config_dir) if config_dir is not None else None

    for override in overrides:
        if "=" not in override:
            continue
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)

        # Support group override style: model=..., dataset=...
        if key in {"model", "dataset"} and isinstance(value, str) and config_dir is not None:
            group_path = config_dir / key / f"{value}.yaml"
            if not group_path.exists():
                raise FileNotFoundError(f"Config group file not found: {group_path}")
            result = _deep_merge(result, _load_yaml(group_path))
            continue

        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    return result
