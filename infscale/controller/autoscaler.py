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
from collections import deque
from typing import TYPE_CHECKING

from infscale import get_logger


if TYPE_CHECKING:
    from infscale.controller.controller import Controller

logger = None


class PerfMetrics:
    """PerfMetrics class."""

    def __init__(self, window_size: int = 5) -> None:
        """Initialize an instance."""
        # length of queue, i.e., number of requests waiting to be served
        self.qlevel = 0.0
        # second to serve one request
        self.delay = 0.0
        # the number of requests served per second
        self.thp = 0.0

        self.window_size = window_size
        self.qlevel_rate = 0.0
        self.thp_rate = 0.0

        self._qlevel_dq = deque()
        self._thp_dq = deque()

        self._suppress_factor = 2.0

    def update(self, qlevel: float, delay: float, thp: float) -> None:
        """Update metric's values."""
        self.qlevel = qlevel
        self.delay = delay
        self.thp = thp

    def update_rate(self) -> None:
        """Update rate of qlevel and thp changes over a window of samples.

        This method doesn't need to be called for every worker in a job. Instead,
        call this method for workers that need to monitor the changes of qlevel
        and throughput.
        This method should be only called in conjunction with update() so that
        the rate change can be accurately caculated.
        """
        self._qlevel_dq.append(self.qlevel)
        self._thp_dq.append(self.thp)

        if len(self._qlevel_dq) < self.window_size:
            return

        oldest_qlevel = self._qlevel_dq.popleft()
        oldest_thp = self._thp_dq.popleft()
        self.qlevel_rate = (self.qlevel - oldest_qlevel) / self.window_size
        self.thp_rate = (self.thp - oldest_thp) / self.window_size

    def is_congested(self) -> bool:
        """Return true if queue continues to build up while throughput is saturated."""
        # a negative rate change can be because of load reduction.
        # in such a case, multiplying a negative value with a suppress factor
        # can result in a false positive. To prevent that, we choose a small
        # non-zero value.
        effective_thp_rate = max(self.thp_rate, 0.000000001)
        return self.qlevel_rate > effective_thp_rate * self._suppress_factor

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

            job_ctx = self._ctrl.job_contexts.get(job_id)
            metrics = job_ctx.get_wrkr_metrics(wrkr_id)

            logger.debug(f"do dummy scaling work for {job_id}-{wrkr_id}")
            logger.debug(f"is congested: {metrics.is_congested()}")

    async def set_event(self, job_id: str, wrkr_id: str) -> None:
        """Set an autoscaling event for a given job and worker."""
        await self._event_queue.put((job_id, wrkr_id))
