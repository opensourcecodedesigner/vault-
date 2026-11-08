"""Explicit local artifacts; callers must only persist synthetic/permitted data."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation keeps previous experiment records immutable.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def run_directory(root):
    path = Path(root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path.mkdir(parents=True, exist_ok=False)
    return path
