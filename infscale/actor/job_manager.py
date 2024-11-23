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
from infscale.config import JobConfig

logger = get_logger()


@dataclass
class WorkerMetaData:
    """WorkerMetaData dataclass."""

    pipe: connection.Connection
    process: mp.Process
    status: WorkerStatus
    id: str
    job_id: str


class JobManager:
    """JobManager class."""

    def __init__(self):
        self._workers: dict[int, WorkerMetaData] = {}
        self.jobs: dict[str, JobConfig] = {}

    def add_worker(self, worker: WorkerMetaData) -> None:
        self._workers[worker.pipe.fileno()] = worker

    def _job_cleanup(self, job_id: str) -> None:
        for k, v in list(self._workers.items()):
            if v.job_id == job_id:
                del self._workers[k]

        if job_id in self.jobs:
            del self.jobs[job_id]

    def send_message_to_worker(self, worker: WorkerMetaData, message: Message) -> None:
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
                worker_data.pipe.fileno(),
            )

    def on_read_ready(
        self,
        worker_data: WorkerMetaData,
        descriptor: int,
    ) -> None:
        """Callback to wait for messages."""
        if worker_data.pipe.poll():
            try:
                message = worker_data.pipe.recv()
                self._handle_message(message, worker_data, descriptor)
            except EOFError:
                self._handle_worker_failure(worker_data)

    def _handle_worker_failure(self, worker_data: WorkerMetaData) -> None:
        self._terminate_worker(worker_data)

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
                self._terminate_workers(message.job_id)

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

    def _terminate_workers(self, job_id: str) -> None:
        """Terminate Workers."""
        loop = asyncio.get_event_loop()

        for worker_data in self._workers.values():
            if worker_data.job_id == job_id and worker_data.status in [WorkerStatus.STARTED, WorkerStatus.READY, WorkerStatus.RUNNING]:
                worker_data.status = WorkerStatus.TERMINATED
                self.send_message_to_worker(
                    worker_data, Message(MessageType.TERMINATE, "", worker_data.job_id)
                )

                loop.remove_reader(worker_data.pipe.fileno())

        logger.info(f"workers for job {job_id} terminated")

        self._job_cleanup(job_id)

    def _terminate_worker(self, worker_data: WorkerMetaData) -> None:
        """Terminate Worker."""
        loop = asyncio.get_event_loop()

        if worker_data.status in [WorkerStatus.STARTED, WorkerStatus.READY, WorkerStatus.RUNNING]:
            worker_data.status = WorkerStatus.TERMINATED

            self.send_message_to_worker(
                worker_data, Message(MessageType.TERMINATE, "", worker_data.job_id)
            )

            loop.remove_reader(worker_data.pipe.fileno())
            worker_data.pipe.close()

            
    def _print_message(self, content: str, process_id: int) -> None:
        """Print received messages."""
        print(f"Process ID: {process_id}, Message: {content}")

    def set_job_config(self, config: JobConfig) -> None:
        """Set job config."""
        self.jobs[config.job_id] = config

    def get_job_config(self, job_id: str) -> JobConfig | None:
        """Return job config."""
        return self.jobs[job_id] if self.job_exists(job_id) else None

    def job_exists(self, job_id) -> bool:
        """Check if job exists."""
        return job_id in self.jobs
