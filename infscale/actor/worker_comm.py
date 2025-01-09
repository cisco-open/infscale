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

"""WorkerCommunicator class."""

import asyncio
import sys
from multiprocessing import connection

from infscale import get_logger
from infscale.actor.job_msg import Message, MessageType, WorkerStatus
from infscale.config import ServeConfig

logger = None


class WorkerCommunicator:
    """WorkerCommunicator class."""

    def __init__(self, pipe: connection.Connection, job_id: str):
        """Initialize an instance."""
        global logger
        logger = get_logger()

        self.pipe = pipe
        self.job_id = job_id
        self.config_q = asyncio.Queue()

    def send(self, message: Message) -> None:
        """Send a message to agent."""
        self.pipe.send(message)

    async def recv(self) -> ServeConfig:
        """Receive a config."""
        return await self.config_q.get()

    def message_listener(self) -> None:
        """Asynchronous worker listener to handle communication with agent."""
        loop = asyncio.get_event_loop()

        loop.add_reader(
            self.pipe.fileno(),
            self.on_read_ready,
            loop,
        )

    def on_read_ready(
        self,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Receive message from a pipe via callback."""
        if self.pipe.poll():
            try:
                message = self.pipe.recv()
                self._handle_message(message)
            except EOFError:
                # TODO: TBD on pipe failure case
                loop.remove_reader(self.pipe.fileno())

    def _handle_message(self, message: Message) -> None:
        match message.type:
            case MessageType.CONFIG:
                _ = asyncio.create_task(self.config_q.put(message.content))

            case MessageType.TERMINATE:
                # TODO: do the clean-up / caching before termination
                self.send(
                    Message(MessageType.STATUS, WorkerStatus.TERMINATED, self.job_id)
                )
                logger.info("worker is terminated")
                sys.exit()
