# =============================================================================
#  logger.py — event logging to logs/events.txt
# =============================================================================

import os
import logging
from datetime import datetime
import config

os.makedirs(config.LOGS_DIR, exist_ok=True)

_logger = None


def init():
    global _logger
    _logger = logging.getLogger("Proctor")
    _logger.setLevel(logging.DEBUG)
    if _logger.handlers:
        _logger.handlers.clear()

    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))

    _logger.addHandler(fh)
    _logger.addHandler(ch)
    _logger.info(f"=== SESSION STARTED {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")


def log(level: str, message: str):
    if _logger is None:
        init()
    if   level == "HIGH":     _logger.error(f"[{level}] {message}")
    elif level == "MEDIUM":   _logger.warning(f"[{level}] {message}")
    else:                     _logger.info(f"[{level}] {message}")


def log_session_end():
    if _logger:
        _logger.info("=== SESSION ENDED ===")
