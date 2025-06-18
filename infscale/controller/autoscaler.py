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
import time
from typing import TYPE_CHECKING

from infscale import get_logger
from infscale.common.metrics import PerfMetrics
from infscale.controller.job_context import JobContext, JobStateEnum


if TYPE_CHECKING:
    from infscale.controller.controller import Controller

logger = None


class AutoScaler:
    """AutoScaler class."""

    def __init__(self, controller: Controller) -> None:
        """Initialize an instance."""
        global logger
        logger = get_logger()

        self._ctrl = controller

        self._event_queue = asyncio.Queue()

        # to suppress frequent autoscaling; defaut: 30 sec
        self._interval = 30
        # the last time to run autoscale
        self._last_run = -1
        # to keep track of whether there is improvement after autoscaling
        self._last_output_rate = -1

    async def run(self) -> None:
        """Run autoscaling functionality."""
        while True:
            job_id, wrkr_id = await self._event_queue.get()

            job_ctx = self._ctrl.job_contexts.get(job_id)
            if job_ctx.state.enum_() != JobStateEnum.RUNNING:
                logger.debug("job not in running state; autoscaling disallowed")
                continue

            metrics = job_ctx.get_wrkr_metrics(wrkr_id)
            logger.debug(f"metrics: {metrics}, is_congested: {metrics.is_congested()}")

            if time.perf_counter() - self._last_run < self._interval:
                # to prevent too frequent autoscaling
                continue

            if not metrics.is_congested():
                # TODO: if not congested, check if scale-in is necessary
                continue

            if self._last_output_rate >= metrics.output_rate:
                # there seems no improvement even after autoscaling
                # so, don't try to scale out
                continue

            await self._scale_out(job_ctx, metrics)

    async def _scale_out(self, ctx: JobContext, metrics: PerfMetrics) -> None:
        rate = metrics.rate_to_decongest()
        ctx.set_desired_rate(rate)

        logger.debug(f"congested, desired rate = {rate}")

        try:
            await ctx.update()
        except Exception as e:
            logger.warning(f"exception: {e}")
            self._last_run = time.perf_counter()
            return

        self._last_run = time.perf_counter()
        self._last_output_rate = metrics.output_rate
        logger.debug("finished scaling-out")

    async def set_event(self, job_id: str, wrkr_id: str) -> None:
        """Set an autoscaling event for a given job and worker."""
        await self._event_queue.put((job_id, wrkr_id))
