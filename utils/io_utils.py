from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, *, must_exist: bool = False) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = None
        for base in (Path.cwd(), ROOT_PATH, ROOT_PATH.parent):
            candidate = (base / p).resolve()
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            resolved = (Path.cwd() / p).resolve()

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")

    return resolved
