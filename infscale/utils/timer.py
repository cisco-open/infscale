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

"""Timer class."""

import asyncio
import os
from asyncio import Task
from typing import Any, Callable
from infscale import log_registry


class Timer:
    """Timer Class."""

    def __init__(
        self,
        timeout: int,
        callback: Callable,
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize timer instance.

        *args and **kwargs are parameters for callback
        """
        self._timeout = timeout
        self._callback = callback
        self._args = args
        self._kwargs = kwargs

        self.logger = log_registry.get_logger(f"{os.getpid()}")

        self._task = self._create()

    def _create(self) -> Task:
        return asyncio.create_task(self._wait())

    async def _wait(self) -> None:
        self.logger.debug(f"wait for {self._timeout} seconds")
        await asyncio.sleep(self._timeout)

        if asyncio.iscoroutinefunction(self._callback):
            self.logger.debug("call coroutine callback")
            await self._callback(*self._args, **self._kwargs)
        else:
            self.logger.debug("call normal callback")
            self._callback(*self._args, **self._kwargs)

    def cancel(self) -> None:
        """Cancel the timer."""
        if self._task is None or self._task.cancelled():
            return

        self._task.cancel()

    def renew(self) -> None:
        """Renew the timer."""
        self.cancel()

        self._task = self._create()
