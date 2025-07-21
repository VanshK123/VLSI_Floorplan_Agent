"""Project-wide logger configuration."""
from __future__ import annotations

import logging
import json


def configure_logger(name: str) -> logging.Logger:
    """Return a configured logger emitting JSON."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(json.dumps({"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}))
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
