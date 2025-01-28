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
from datetime import datetime

# Add custom PROFILE level
PROFILE_LEVEL = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(PROFILE_LEVEL, 'PROFILE')
setattr(logging, 'PROFILE', PROFILE_LEVEL)

def profile(self, message, *args, **kwargs):
    self.log(PROFILE_LEVEL, message, *args, **kwargs)
logging.Logger.profile = profile

# Get log level from env var, default to WARNING
level = getattr(logging, os.getenv("INFSCALE_LOG_LEVEL", "WARNING"))

format = "%(process)d | %(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(threadName)s | %(funcName)s | %(message)s"

# Create timestamp for log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
default_log_file = f"infscale_{timestamp}.log"

# Set up basic logging to current directory's log folder
LOG_DIR = Path.cwd() / "log"  # Changed from HOME_LOG_DIR to use current working directory
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file_path = LOG_DIR / default_log_file

# Configure root logger
logging.basicConfig(
    level=level,
    format=format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file_path)
    ]
)

logger_registry: dict[str | int, logging.Logger] = dict()

def get_logger(
    key: str = f"{os.getpid()}",
    log_file_path: str = None,
) -> logging.Logger:
    """Get a logger with a given key.

    If the logger doesn't exist, one will be created.
    All loggers will use the same log file configured at startup for easy parsing.
    """
    if key not in logger_registry:
        logger = logging.getLogger(key)
        logger_registry[key] = logger

    return logger_registry[key]
