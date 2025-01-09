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

"""Worker class."""
import asyncio
import os
from multiprocessing.connection import Connection

from infscale import get_logger
from infscale.actor.worker_comm import WorkerCommunicator
from infscale.execution.pipeline import Pipeline

logger = None


class Worker:
    """Worker class."""

    def __init__(self, local_rank: int, conn: Connection, job_id: str, wrk_id: str):
        """Initialize an instance."""
        global logger
        logger = get_logger(f"{os.getpid()}", f"job-{job_id}/worker-{wrk_id}.log")

        self.local_rank = local_rank
        self.wcomm = WorkerCommunicator(conn, job_id)

    def run(self) -> None:
        """Run worker."""
        asyncio.run(self._run())

    async def _run(self) -> None:
        """Run the worker."""
        logger.info(f"worker {self.local_rank}")
        self.wcomm.message_listener()
        pipeline = Pipeline(self.wcomm)
        await pipeline.run()
