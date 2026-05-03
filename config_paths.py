"""Resolve dataset paths in YAML relative to the config file (any cwd / Colab)."""

from pathlib import Path
from typing import Any, Dict


def resolve_config_paths(config: Dict[str, Any], config_path: str) -> None:
    base = Path(config_path).expanduser().resolve().parent

    def _resolve(p: str) -> str:
        if not isinstance(p, str) or not p.strip():
            return p
        path = Path(p)
        if path.is_absolute():
            return str(path.resolve())
        return str((base / path).resolve())

    params = config.get("dataset", {}).get("params")
    if not isinstance(params, dict):
        return
    if "root_dir" in params:
        params["root_dir"] = _resolve(params["root_dir"])
    feats = params.get("features") or {}
    if isinstance(feats, dict):
        for key in ("audio_dir", "visual_dir"):
            if key in feats and feats[key]:
                feats[key] = _resolve(feats[key])
