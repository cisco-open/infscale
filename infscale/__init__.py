# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Dunder init file."""

import logging
import os
import sys
from pathlib import Path

from infscale.version import VERSION as __version__  # noqa: F401


level = getattr(logging, os.getenv("INFSCALE_LOG_LEVEL", "WARNING"))

format = "%(process)d | %(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(threadName)s | %(funcName)s | %(message)s"

logging.basicConfig(level=level, format=format, stream=sys.stdout)


HOME_LOG_DIR = Path(os.getenv("HOME")) / ".infscale" / "log"
HOME_LOG_DIR.mkdir(parents=True, exist_ok=True)
logger_registry: dict[str | int, logging.Logger] = dict()


def get_logger(
    key: str = f"{os.getpid()}",
    log_file_path: str = None,
) -> logging.Logger:
    """Get a logger with a given key.

    If the logger doesn't exist, one will be created.
    """
    if key not in logger_registry:
        _create_logger(key, log_file_path)

    logger = logger_registry[key]

    return logger


def _create_logger(key: str | int, log_file_path: str = None) -> None:
    """Create a logger with a given key."""
    if key in logger_registry:
        raise ValueError(f"Logger '{key}' already exists.")

    logger = logging.getLogger(key)

    if log_file_path:
        log_file_path = HOME_LOG_DIR / log_file_path
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(format))

        logger.addHandler(file_handler)

    logger_registry[key] = logger
