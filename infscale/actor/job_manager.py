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
from multiprocessing import connection

import torch.multiprocessing as mp

from infscale import get_logger
from infscale.actor.job_msg import Message, MessageType, WorkerStatus

logger = get_logger()


@dataclass
class WorkerMetaData:
    """WorkerMetaData dataclass."""

    pipe: connection.Connection
    process: mp.Process
    status: WorkerStatus


class JobManager:
    """JobManager class."""

    def __init__(self):
        self._workers: dict[int, WorkerMetaData] = {}

    def add_worker(self, worker: WorkerMetaData) -> None:
        self._workers[worker.pipe.fileno()] = worker

    def send_message(self, worker: WorkerMetaData, message: Message) -> None:
        """Send message to worker."""
        worker.pipe.send(message)

    def message_listener(self) -> None:
        """Asynchronous parent listener to handle communication with workers."""
        loop = asyncio.get_event_loop()

        for worker_data in self._workers.values():
            loop.add_reader(
                worker_data.pipe.fileno(),
                self.on_read_ready,
                worker_data,
                loop,
                worker_data.pipe.fileno(),
            )

    def on_read_ready(
        self,
        worker_data: WorkerMetaData,
        loop: asyncio.AbstractEventLoop,
        descriptor: int,
    ) -> None:
        """Callback to wait for messages."""
        if worker_data.pipe.poll():  # Check if there's data to read
            try:
                message = worker_data.pipe.recv()  # Receive the message
                self._handle_message(message, worker_data, descriptor)
            except EOFError:
                self._handle_worker_failure(loop, worker_data)

    def _handle_worker_failure(self, loop, worker_data: WorkerMetaData) -> None:
        loop.remove_reader(worker_data.pipe.fileno())  # Clean up the reader
        self._terminate_workers()

    def _handle_message(
        self, message: Message, worker_data: WorkerMetaData, descriptor: int
    ) -> None:
        """Handle received messages."""
        match message.type:
            case MessageType.LOG:
                self._print_message(message.content, worker_data.process.pid)

            case MessageType.STATUS:
                self._handle_status(message, descriptor)

    def _handle_status(self, message: Message, descriptor: int) -> None:
        """Handle status update from Workers."""
        self._update_worker_status(message, descriptor)

        match message.content:
            case WorkerStatus.DONE:
                self._terminate_workers()

            case WorkerStatus.STARTED:
                pass

            case WorkerStatus.RUNNING:
                pass

            case WorkerStatus.TERMINATED:
                pass

            case WorkerStatus.FAILED:
                pass

    def _update_worker_status(self, message: Message, descriptor: int) -> None:
        """Update Worker status."""
        self._workers[descriptor].status = message.content

    def _terminate_workers(self) -> None:
        """Terminate Workers."""
        for worker_data in self._workers.values():
            # TODO: update logic to terminate workers belonging to a terminated server only
            worker_data.status = WorkerStatus.TERMINATED
            worker_data.process.terminate()

    def _print_message(self, content: str, process_id: int) -> None:
        """Print received messages."""
        print(f"Process ID: {process_id}, Message: {content}")
