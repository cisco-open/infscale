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

import asyncio
from dataclasses import dataclass
from enum import Enum
from multiprocessing import connection

import torch.multiprocessing as mp
from infscale import get_logger

logger = get_logger()


class MessageType(Enum):
    """MessageType enum."""

    LOG = "log"
    TERMINATE = "terminate"


@dataclass
class Message:
    """WorkerMetaData dataclass."""

    type: MessageType
    content: str


@dataclass
class WorkerMetaData:
    """WorkerMetaData dataclass."""

    pipe: connection.Connection
    process: mp.Process


class JobManager:
    """JobManager class."""

    def __init__(self, metadata: WorkerMetaData):
        self.metadata = metadata

    def message_listener(self) -> None:
        """Asynchronous parent listener to handle communication with workers."""
        loop = asyncio.get_event_loop()

        for worker_data in self.metadata.values():
            loop.add_reader(
                worker_data.pipe.fileno(), self.on_read_ready, worker_data, loop
            )

    def on_read_ready(
        self, worker_data: WorkerMetaData, loop: asyncio.AbstractEventLoop
    ) -> None:
        if worker_data.pipe.poll():  # Check if there's data to read
            try:
                message = worker_data.pipe.recv()  # Receive the message
                self._handle_message(message, worker_data.process.pid)
            except EOFError:
                loop.remove_reader(worker_data.pipe.fileno())  # Clean up the reader

    def _handle_message(self, message: Message, process_id: int) -> None:
        if message.content:
            self._print_message(message.content, process_id)

        match message.type:
            case MessageType.TERMINATE:
                self._terminate_workers()

    def _terminate_workers(self) -> None:
        for worker_data in self.metadata.values():
            # TODO: update logic to terminate workers belonging to a terminated server only
            worker_data.process.terminate()

    def _print_message(self, content: str, process_id: int) -> None:
        print(f"Process ID: {process_id}, Message: {content}")
