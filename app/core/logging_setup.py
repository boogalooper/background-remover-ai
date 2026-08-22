from __future__ import annotations

import faulthandler
import logging
from logging.handlers import RotatingFileHandler

from app.paths import CRASH_LOG_PATH, LOG_PATH


_crash_stream = None

def setup_logging() -> None:
    global _crash_stream
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler = RotatingFileHandler(LOG_PATH, maxBytes=4_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Native extensions (CUDA, image codecs, etc.) can terminate the process
    # before Python can raise an exception. Keep a separate low-level crash log
    # so a future access violation/abort leaves useful diagnostics.
    try:
        if _crash_stream is None:
            _crash_stream = open(CRASH_LOG_PATH, "a", encoding="utf-8")
        faulthandler.enable(_crash_stream, all_threads=True)
    except Exception:
        root.warning("Could not enable native crash log", exc_info=True)
