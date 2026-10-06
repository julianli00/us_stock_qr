from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from us_quant.config import QuantError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def digest_json(value: object) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def file_digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise QuantError(f"Expected a JSON object: {path}")
    return value


def implementation_fingerprint() -> str:
    root = Path(__file__).parent
    names = (
        "config.py",
        "calendar.py",
        "data.py",
        "strategy.py",
        "backtest.py",
        "metrics.py",
        "research.py",
        "storage.py",
    )
    return digest_json({name: file_digest(root / name) for name in names})


def new_output_directory(path: Path) -> None:
    if path.exists():
        raise QuantError(
            f"Refusing to overwrite evidence at {path}; choose a new output directory."
        )
    path.mkdir(parents=True)
