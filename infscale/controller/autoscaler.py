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

    def __init__(
        self,
        qlevel: float = -1,
        thp: float = -1,
        rtt: float = -1,
    ):
        """Initialize an instance."""
        # length of queue, i.e., number of requests waiting to be served
        self.qlevel = qlevel

        # the number of requests served per second
        self.thp = thp

        # second to serve one request
        self.rtt = rtt

    def __str__(self) -> str:
        """Return string representation for the object."""
        return f"qlevel: {self.qlevel}, thp: {self.thp}, rtt: {self.rtt}"


class AutoScaler:
    """AutoScaler class."""

    def __init__(self, controller: Controller) -> None:
        """Initialize an instance."""
        global logger
        logger = get_logger()

        self.metrics_queue = asyncio.Queue()
        self._ctrl = controller

    async def run(self) -> None:
        """Run autoscaling functionality."""
        while True:
            job_id, metrics = await self.metrics_queue.get()
            logger.debug(f"do dummy scaling work for {job_id}")
            logger.debug(f"got metrics: {job_id}")

    async def set_metrics(self, job_id: str, metrics: PerfMetrics) -> None:
        """Set performance metrics for a given job."""
        await self.metrics_queue.put((job_id, metrics))
