"""Small, atomic artifact helpers; completed files are content-checked on reuse."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

SCHEMA = 1


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def checked_manifest(path: Path, expected: dict | None = None) -> dict:
    manifest = read_json(path)
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported artifact schema: {path}")
    if expected is not None and manifest.get("signature") != expected:
        raise ValueError(f"Artifact configuration/source mismatch: {path}")
    for name, digest in manifest.get("files", {}).items():
        if file_hash(path.parent / name) != digest:
            raise ValueError(f"Artifact checksum mismatch: {path.parent / name}")
    return manifest


@contextmanager
def stage_lock(directory: Path):
    """Single-writer lock released by the OS on crash (not a stale PID file)."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another process is writing {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
