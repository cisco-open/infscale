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
from multiprocessing.connection import Connection

from infscale import get_logger
from infscale.common.job_msg import Message

logger = None


class WorkerCommunicator:
    """WorkerCommunicator class."""

    def __init__(self, pipe: Connection):
        """Initialize an instance."""
        global logger
        logger = get_logger()

        self.pipe = pipe
        self.msg_q = asyncio.Queue()

    def send(self, message: Message) -> None:
        """Send a message to agent."""
        self.pipe.send(message)

    async def recv(self) -> Message:
        """Receive a message."""
        return await self.msg_q.get()

    def message_listener(self) -> None:
        """Add a listener to handle communication with agent."""
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
            except EOFError:
                # TODO: TBD on pipe failure case
                loop.remove_reader(self.pipe.fileno())

            # put the message into the message queue
            # so that it can be comsumed in the pipeline
            _ = asyncio.run_coroutine_threadsafe(self.msg_q.put(message), loop)
