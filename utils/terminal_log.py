"""Mirror stdout/stderr to a UTF-8 file for experiment logs.

Notes:
- Captures Python-level print/logging/tqdm that use sys.stdout/sys.stderr.
- Output from some native libraries (e.g. parts of CUDA) may not go through
  these streams; use shell ``tee`` if you need a byte-perfect terminal dump.
"""
from __future__ import annotations

import atexit
import sys
from datetime import datetime
from pathlib import Path

_orig_out = None
_orig_err = None
_log_fp = None
_installed = False


class _TeeTextIO:
    __slots__ = ("_primary", "_file")

    def __init__(self, primary, file_):
        self._primary = primary
        self._file = file_

    def write(self, data):
        self._primary.write(data)
        self._file.write(data)
        self._primary.flush()
        self._file.flush()

    def flush(self):
        self._primary.flush()
        self._file.flush()

    def fileno(self):
        return self._primary.fileno()

    def isatty(self):
        return self._primary.isatty()

    def writable(self):
        return True

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")

    def __getattr__(self, name):
        return getattr(self._primary, name)


def _cleanup():
    global _orig_out, _orig_err, _log_fp, _installed
    if not _installed:
        return
    try:
        if _log_fp:
            _log_fp.write(f"\n===== terminal log ended {datetime.now().isoformat()} =====\n")
            _log_fp.flush()
            _log_fp.close()
    finally:
        if _orig_out is not None:
            sys.stdout = _orig_out
        if _orig_err is not None:
            sys.stderr = _orig_err
        _log_fp = None
        _orig_out = None
        _orig_err = None
        _installed = False


def install_terminal_log(
    path: str,
    *,
    append: bool = False,
    argv_for_banner=None,
) -> None:
    """Tee stdout and stderr to *path* (creates parent directories).

    *argv_for_banner* defaults to ``sys.argv``; pass explicit CLI tokens (e.g. from
    ``get_args(argv=...)``) so the header matches the parsed command.
    """
    global _orig_out, _orig_err, _log_fp, _installed
    if _installed:
        raise RuntimeError("install_terminal_log() is already active")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    _log_fp = open(p, mode, encoding="utf-8", errors="replace", newline="")
    _orig_out = sys.stdout
    _orig_err = sys.stderr
    av = sys.argv if argv_for_banner is None else argv_for_banner
    banner = f"\n===== terminal log started {datetime.now().isoformat()} =====\nargv: {av!r}\n"
    _log_fp.write(banner)
    _log_fp.flush()
    sys.stdout = _TeeTextIO(_orig_out, _log_fp)
    sys.stderr = _TeeTextIO(_orig_err, _log_fp)
    _installed = True
    atexit.register(_cleanup)
