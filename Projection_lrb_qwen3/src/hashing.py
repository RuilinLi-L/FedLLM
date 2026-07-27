"""Canonical SHA-256 helpers used by the preregistration protocol."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class HashingError(RuntimeError):
    """Raised when a required file cannot be hashed deterministically."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return a stable UTF-8 JSON representation suitable for hashing."""
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise HashingError(f"Value is not canonical-JSON serializable: {error}") from error
    return text.encode("utf-8")


def sha256_text(value: str) -> str:
    """Hash one UTF-8 text value."""
    return sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON data."""
    return sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file without loading it fully into memory."""
    if not path.is_file():
        raise HashingError(f"Required hash target is not a regular file: {path}")
    digest = sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(chunk_size)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise HashingError(f"Unable to hash {path}: {error}") from error
    return digest.hexdigest()


def hash_file_map(paths: Iterable[Path], *, base_dir: Path) -> dict[str, str]:
    """Hash paths under ``base_dir`` using normalized relative-path keys."""
    result: dict[str, str] = {}
    base = base_dir.resolve()
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(base).as_posix()
        except ValueError as error:
            raise HashingError(f"Hash target is outside base directory: {path}") from error
        if relative in result:
            raise HashingError(f"Duplicate hash target: {relative}")
        result[relative] = sha256_file(path)
    if not result:
        raise HashingError("Refusing to create an empty key-file hash map.")
    return result


def sample_key(*, original_index: int, sentence: str, label: int) -> str:
    """Return the protocol-defined, public sample ordering key."""
    material = f"glue|sst2|validation|{original_index}|{sentence}|{label}"
    return sha256_text(material)


def hash_sample_list(samples: list[Mapping[str, Any]]) -> str:
    """Hash a stage's ordered sample records exactly as stored."""
    return sha256_json(samples)
