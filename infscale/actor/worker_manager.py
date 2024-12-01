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

"""worker_manager.py."""

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
    job_id: str
    id: str
    status: WorkerStatus


class WorkerManager:
    """WorkerManager class."""

    def __init__(self):
        """Initialize an instance."""
        self._workers: dict[int, WorkerMetaData] = {}

    def add(
        self,
        pipe: connection.Connection,
        process: mp.Process,
        job_id: str,
        stage_id: str,
        status: WorkerStatus = WorkerStatus.READY,
    ) -> WorkerMetaData:
        """Add a worker into the worker manager.

        This method creates a worker metadata and adds it to the worker manager
        as a dictionary. At the end it returns the worker metadata instance.
        """
        worker = WorkerMetaData(pipe, process, job_id, stage_id, status)
        self._workers[worker.pipe.fileno()] = worker

        return worker

    def get_workers(
        self, job_id: str, worker_ids: set[str]
    ) -> dict[str, WorkerMetaData]:
        """Return workers that match job_id and id in worker_ids."""
        results = {}
        for v in self._workers.values():
            if v.job_id == job_id and v.id in worker_ids:
                results[v.id] = v

        return results

    def _cleanup(
        self, job_id: str, by_worker_id: bool = False, worker_ids=set()
    ) -> None:
        """Clean up worker meta data information for a given job.

        If by_worker_id is true, then worker data in the worker_ids set
        are cleaned up.
        """
        for k, v in list(self._workers.items()):
            if v.job_id != job_id:
                continue

            if by_worker_id and v.id not in worker_ids:
                continue

            del self._workers[k]

    def send(self, worker: WorkerMetaData, message: Message) -> None:
        """Send message to worker."""
        worker.pipe.send(message)

    def initialize_listener(self, worker: WorkerMetaData) -> None:
        """Initialize a listener to handle communication with workers."""
        loop = asyncio.get_event_loop()

        fd = worker.pipe.fileno()
        loop.add_reader(fd, self.on_read_ready, worker, fd)

    def on_read_ready(self, worker: WorkerMetaData, fd: int) -> None:
        """Receive message from a pipe via callback."""
        if worker.pipe.poll():
            try:
                message = worker.pipe.recv()
                self._handle_message(message, worker, fd)
            except EOFError:
                self._terminate_worker(worker)

    def _handle_message(
        self, message: Message, worker: WorkerMetaData, fd: int
    ) -> None:
        """Handle received messages."""
        match message.type:
            case MessageType.LOG:
                self._print_message(message.content, worker.process.pid)

            case MessageType.STATUS:
                self._handle_status(message, fd)

    def _handle_status(self, message: Message, fd: int) -> None:
        """Handle status update from Workers."""
        self._update_worker_status(message, fd)

        match message.content:
            case WorkerStatus.DONE:
                self.terminate_workers(message.job_id)

            case WorkerStatus.STARTED:
                pass

            case WorkerStatus.RUNNING:
                pass

            case WorkerStatus.TERMINATED:
                pass

            case WorkerStatus.FAILED:
                pass

    def _update_worker_status(self, message: Message, fd: int) -> None:
        """Update Worker status."""
        self._workers[fd].status = message.content

    def terminate_workers(
        self, job_id: str, by_worker_id: bool = False, worker_ids=set()
    ) -> None:
        """Terminate workers of a given job id.

        If by_worker_id is true, then workers in the worker_ids set
        are terminated.
        """
        terminated = False
        for worker in self._workers.values():
            if worker.job_id != job_id:
                continue

            if by_worker_id and worker.id not in worker_ids:
                continue

            self._terminate_worker(worker)
            terminated = True

        self._cleanup(job_id, by_worker_id, worker_ids)

        if not terminated:
            # no worker is termianted; so no need to log below
            return

        if by_worker_id:
            logger.info(f"workers {worker_ids} for job {job_id} terminated")
        else:
            logger.info(f"all workers for job {job_id} terminated")

    def _terminate_worker(self, worker: WorkerMetaData) -> None:
        """Terminate a worker."""
        loop = asyncio.get_event_loop()

        valid = [WorkerStatus.STARTED, WorkerStatus.READY, WorkerStatus.RUNNING]
        if worker.status in valid:
            worker.status = WorkerStatus.TERMINATED

            msg = Message(MessageType.TERMINATE, "", worker.job_id)
            self.send(worker, msg)

            loop.remove_reader(worker.pipe.fileno())
            worker.pipe.close()

    def _print_message(self, content: str, process_id: int) -> None:
        """Print received messages."""
        print(f"Process ID: {process_id}, Message: {content}")
