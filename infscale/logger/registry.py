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

"""LoggerRegistry class."""

import logging
import os
from pathlib import Path
from logging import Logger

import sys

level = getattr(logging, os.getenv("INFSCALE_LOG_LEVEL", "WARNING"))

format = "%(process)d | %(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(threadName)s | %(funcName)s | %(message)s"

logging.basicConfig(level=level, format=format, stream=sys.stdout)


HOME_LOG_DIR = Path(os.getenv("HOME")) / ".infscale" / "log"
HOME_LOG_DIR.mkdir(parents=True, exist_ok=True)


class LoggerRegistry:
    """Pipeline class."""

    def __init__(
        self,
    ):
        """Initialize logger registry instance."""
        self.logger_registry: dict[str | int, Logger] = dict()
        print('LOG LEVEL', level, os.getenv("HOME"))

    def get_logger(self, name: int, log_file_path: str = None) -> Logger:
        """Get a logger with a given name.

        If the logger doesn't exist, one will be created.
        """
        if name not in self.logger_registry:
            self._create_logger(name, log_file_path)
        
        logger = self.logger_registry[name]

        return logger

    def _create_logger(self, name: str | int, log_file_path: str = None) -> None:
        """Create a logger with a given name."""
        if name in self.logger_registry:
            raise ValueError(f"Logger '{name}' already exists.")

        logger = logging.getLogger(name)

        if log_file_path:
            log_file_path = HOME_LOG_DIR / log_file_path
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file_path, "a")
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(format))

            logger.addHandler(file_handler)

        self.logger_registry[name] = logger


        

   