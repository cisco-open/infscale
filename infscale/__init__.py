"""Dunder init file."""

import logging
import os
import sys
from logging import Logger

from infscale.version import VERSION as __version__  # noqa: F401

logfile_prefix = os.path.join(os.getenv("HOME", "/tmp"), "infscale")
os.makedirs(logfile_prefix, exist_ok=True)

level = getattr(logging, os.getenv("LOG_LEVEL", "NOTSET"))
formatter = logging.Formatter(
    "%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(threadName)s | %(funcName)s | %(message)s"
)

logger_registry = dict()


def get_logger(name: str = __name__) -> Logger:
    """Return a logger with a given logger name.

    A log file is created with the name under $HOME/infscale folder.
    """
    if name in logger_registry:
        return logger_registry[name]

    logfile_path = os.path.join(logfile_prefix, name + ".log")
    fileHandler = logging.FileHandler(logfile_path)
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)

    logger_registry[name] = logger

    return logger
