"""Atomic, conflict-detecting JSON and JSONL artifact writers."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

from .hashing import canonical_json_bytes


class ResultSchemaError(RuntimeError):
    """Raised when a preregistration artifact is malformed or conflicts."""


def _canonical_json_text(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8") + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        temporary.replace(path)
    except OSError as error:
        raise ResultSchemaError(f"Unable to write {path}: {error}") from error


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ResultSchemaError(f"Unable to read existing artifact {path}: {error}") from error


def write_or_verify_json(
    path: Path,
    document: Mapping[str, Any],
    *,
    identity_key: str,
    ignored_existing_keys: Sequence[str] = (),
) -> bool:
    """Write a JSON artifact once, or verify its immutable protocol identity.

    Returns ``True`` when a new file was written and ``False`` when an existing
    matching artifact was retained.  Only fields named in
    ``ignored_existing_keys`` may differ when an existing artifact is retained.
    """
    if identity_key not in document:
        raise ResultSchemaError(f"JSON document lacks identity key {identity_key!r}.")
    expected = _canonical_json_text(document)
    if not path.exists():
        _atomic_write_text(path, expected)
        return True
    try:
        existing = json.loads(_read_text(path))
    except json.JSONDecodeError as error:
        raise ResultSchemaError(f"Existing JSON artifact is invalid: {path}: {error}") from error
    if not isinstance(existing, dict) or existing.get(identity_key) != document[identity_key]:
        raise ResultSchemaError(
            f"Existing artifact conflicts with the requested preregistration: {path}"
        )
    existing_comparable = dict(existing)
    expected_comparable = dict(document)
    for key in ignored_existing_keys:
        existing_comparable.pop(key, None)
        expected_comparable.pop(key, None)
    if _canonical_json_text(existing_comparable) != _canonical_json_text(expected_comparable):
        raise ResultSchemaError(
            f"Existing artifact has the same identity but different immutable metadata: {path}"
        )
    return False


def jsonl_text(records: Iterable[Mapping[str, Any]]) -> str:
    """Encode records as canonical JSON Lines, rejecting an empty collection."""
    lines = [_canonical_json_text(record).rstrip("\n") for record in records]
    if not lines:
        raise ResultSchemaError("Refusing to write an empty JSONL preregistration artifact.")
    return "\n".join(lines) + "\n"


def write_or_verify_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> bool:
    """Write exact canonical JSONL once, or retain only an exact match."""
    expected = jsonl_text(records)
    if not path.exists():
        _atomic_write_text(path, expected)
        return True
    existing = _read_text(path)
    if existing != expected:
        raise ResultSchemaError(
            f"Existing JSONL artifact conflicts with the requested preregistration: {path}"
        )
    return False
