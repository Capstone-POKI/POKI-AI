from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _state_dir() -> Path:
    return Path(os.getenv("AI_STATE_DIR", "data/output/runtime_state"))


def load_state(name: str) -> dict[str, Any]:
    path = _state_dir() / f"{name}.json"
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def save_state(name: str, payload: dict[str, Any]) -> None:
    state_dir = _state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    target = state_dir / f"{name}.json"

    fd, temp_name = tempfile.mkstemp(
        dir=state_dir,
        prefix=f".{name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
            json.dump(payload, temp_file, ensure_ascii=False, separators=(",", ":"))
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_name, target)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
