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

    def __init__(self, window_size: int = 10) -> None:
        """Initialize an instance.

        Attributes:
            window_size (int): a sliding window size to calculate diff; default is 10.
        """
        # length of queue, i.e., number of requests waiting to be served
        self.qlevel = 0.0
        # second to serve one request
        self.delay = 0.0
        # the number of requests arrived per second
        self.input_rate = 0.0
        # the number of requests served per second
        self.output_rate = 0.0

        self._window_size = window_size
        self._qlevel_diff = 0.0
        self._output_rate_diff = 0.0

        self._qlevel_dq = deque()
        self._output_rate_dq = deque()

        self._suppress_factor = 1.5

    def update(
        self, qlevel: float, delay: float, input_rate: float, output_rate: float
    ) -> None:
        """Update metric's values."""
        self.qlevel = qlevel
        self.delay = delay
        self.input_rate = input_rate
        self.output_rate = output_rate

    def update_rate(self) -> None:
        """Update rate of qlevel and output rate changes over a window of samples.

        This method doesn't need to be called for every worker in a job. Instead,
        call this method for workers that need to monitor the changes of qlevel
        and throughput.
        This method should be only called in conjunction with update() so that
        the rate change can be accurately caculated.
        """
        self._qlevel_dq.append(self.qlevel)
        self._output_rate_dq.append(self.output_rate)

        if len(self._qlevel_dq) < self._window_size:
            return

        old_qlevel = self._qlevel_dq.popleft()
        old_output_rate = self._output_rate_dq.popleft()
        self._qlevel_diff = self.qlevel - old_qlevel
        self._output_rate_diff = self.output_rate - old_output_rate

    def is_congested(self) -> bool:
        """Return true if queue continues to build up while throughput is saturated."""
        # measure qlevel is larger than output rate * suppress factor
        cond1 = self.qlevel > self.output_rate * self._suppress_factor
        # measure if qlevel change is larger than output rate change
        cond2 = self._qlevel_diff > self._output_rate_diff
        logger.debug(f"cond1: {cond1}, cond2: {cond2}")

        return cond1 or cond2

    def __str__(self) -> str:
        """Return string representation for the object."""
        return f"qlevel: {self.qlevel}, delay: {self.delay}, input_rate: {self.input_rate}, output_rate: {self.output_rate}"


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
