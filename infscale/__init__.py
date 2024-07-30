"""Dunder init file."""

import logging
import os
from logging import Logger

from infscale.version import VERSION as __version__  # noqa: F401

logfile_prefix = os.path.join(os.getenv("HOME", "/tmp"), "infscale")
os.makedirs(logfile_prefix, exist_ok=True)

level = getattr(logging, os.getenv("INFSCALE_LOG_LEVEL", "WARNING"))
formatter = logging.Formatter(
    "%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(threadName)s | %(funcName)s | %(message)s"
)

logger_registry = dict()


def get_logger(name: str = __name__) -> Logger:
    """Return a logger with a given logger name.

    A log file is created with the name under $HOME/infscale folder.
    """
    global logger_registry

    if name in logger_registry:
        return logger_registry[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger_registry[name] = logger

    return logger
