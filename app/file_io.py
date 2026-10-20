"""Atomic replacement that tolerates brief Windows reader/antivirus locks."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Callable


WINDOWS = os.name == "nt"
REPLACE_ATTEMPTS = 12
REPLACE_INITIAL_DELAY = 0.01
REPLACE_MAX_DELAY = 0.1
_WINDOWS_LOCK_ERRORS = {5, 32, 33}


def replace_file(source: Path, destination: Path, *, check: Callable[[], object] | None = None) -> None:
    """Replace once atomically, retrying only transient Windows lock errors.

    The caller keeps the same closed temporary file throughout retries and
    removes it on failure. Recheck its path guards before every attempt,
    since another process can change paths while a reader releases its handle.
    Permanent/unknown errors propagate, preserving the previous destination.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        if check is not None:
            check()
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if (not WINDOWS or getattr(error, "winerror", None) not in _WINDOWS_LOCK_ERRORS
                    or attempt == REPLACE_ATTEMPTS - 1):
                raise
            time.sleep(min(REPLACE_INITIAL_DELAY * (2 ** attempt), REPLACE_MAX_DELAY))
