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
from multiprocessing.connection import Connection

from infscale import get_logger
from infscale.actor.job_msg import Message, MessageType, WorkerStatus
from infscale.actor.worker_manager import WorkerManager
from infscale.execution.pipeline import Pipeline

logger = get_logger()


class Worker:
    """Worker class."""

    def __init__(self, local_rank: int, conn: Connection):
        """Initialize an instance."""
        self.local_rank = local_rank
        self.conn = conn
        self.worker_manager = WorkerManager(self.conn)
        self.worker_manager.send_message(
            Message(
                MessageType.STATUS,
                WorkerStatus.STARTED,
            )
        )

    def run(self) -> None:
        """Run worker."""
        asyncio.run(self._run())

    async def _run(self) -> None:
        """Run the worker."""
        logger.info(f"worker {self.local_rank}")
        self.worker_manager.message_listener()
        pipeline = Pipeline(self.worker_manager)
        await pipeline.run()
