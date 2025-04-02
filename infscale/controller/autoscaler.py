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

"""autoscaler.py."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from infscale import get_logger


if TYPE_CHECKING:
    from infscale.controller.controller import Controller

logger = None


class PerfMetrics:
    """PerfMetrics class."""

    def __init__(self):
        """Initialize an instance."""
        # length of queue, i.e., number of requests waiting to be served
        self.qlevel = 0.0
        # second to serve one request
        self.delay = 0.0
        # the number of requests served per second
        self.thp = 0.0

    def update(self, qlevel: float, delay: float, thp: float):
        """Update metric's values."""
        self.qlevel = qlevel
        self.delay = delay
        self.thp = thp

    def __str__(self) -> str:
        """Return string representation for the object."""
        return f"qlevel: {self.qlevel}, delay: {self.delay}, thp: {self.thp}"


class AutoScaler:
    """AutoScaler class."""

    def __init__(self, controller: Controller) -> None:
        """Initialize an instance."""
        global logger
        logger = get_logger()

        self._event_queue = asyncio.Queue()
        self._ctrl = controller

    async def run(self) -> None:
        """Run autoscaling functionality."""
        while True:
            job_id, wrkr_id = await self._event_queue.get()
            logger.debug(f"do dummy scaling work for {job_id}-{wrkr_id}")

    async def set_event(self, job_id: str, wrkr_id: str) -> None:
        """Set an autoscaling event for a given job and worker."""
        await self._event_queue.put((job_id, wrkr_id))
