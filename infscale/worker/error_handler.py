# Copyright 2025 Cisco Systems, Inc. and its affiliates
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

"""error_handler.py."""

import asyncio
import os

from infscale import get_logger


logger = None


class WorkerErrorHandler:
    """WorkerErrorHandler class."""

    def __init__(self):
        """Initialize an instance."""
        global logger
        logger = get_logger()
        
        self._loop = asyncio.get_running_loop()
        self._err_q = asyncio.Queue()
        _ = asyncio.create_task(self._process_error_queue())
        
    async def _process_error_queue(self) -> None:
        """Process queue for incoming error messages."""
        while True:
            msg = await self._err_q.get()
            logger.error(msg)
            os._exit(0)


    def put(self, msg: str) -> None:
        """Put message into the err_q."""

        _ = asyncio.run_coroutine_threadsafe(self._err_q.put(msg), self._loop)


_error_handler: WorkerErrorHandler = None


def get_worker_error_handler() -> WorkerErrorHandler:
    """Get or create the global WorkerErrorHandler."""
    global _error_handler

    if _error_handler is None:
        _error_handler = WorkerErrorHandler()

    return _error_handler
