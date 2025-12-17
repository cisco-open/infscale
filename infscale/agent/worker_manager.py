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
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import connection

import torch.multiprocessing as mp

from infscale import get_logger
from infscale.common.job_msg import (
    Message,
    MessageType,
    WorkerStatus,
    WorkerStatusMessage,
)
from infscale.common.metrics import PerfMetrics


logger = None


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
        global logger
        logger = get_logger()

        # agent can deploy workers from multiple jobs
        # we group workers by job id
        self._workers: defaultdict[str, dict[int, WorkerMetaData]] = defaultdict(dict)
        self.status_q: asyncio.Queue[WorkerStatusMessage] = asyncio.Queue()
        self.metrics_q: asyncio.Queue[tuple[str, str, PerfMetrics]] = asyncio.Queue()

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
        fd = pipe.fileno()

        self._workers[job_id][fd] = worker

        return worker

    def has_workers_for_job(self, job_id: str) -> bool:
        """Return True if there are any workers assigned to the given job ID."""
        return len(self._workers[job_id]) > 0

    def get_workers_by_job_id(self, job_id: str) -> dict[str, WorkerMetaData]:
        """Return workers that match job_id."""
        return {w.id: w for w in self._workers[job_id].values()}

    def get_workers(
        self, job_id: str, worker_ids: set[str]
    ) -> dict[str, WorkerMetaData]:
        """Return workers that match job_id and id in worker_ids."""
        results = {}
        for v in self._workers[job_id].values():
            if v.id in worker_ids:
                results[v.id] = v

        return results

    def send(self, worker: WorkerMetaData, message: Message) -> None:
        """Send message to worker."""
        try:
            worker.pipe.send(message)
        except OSError:
            return

    def initialize_listener(self, worker: WorkerMetaData) -> None:
        """Initialize a listener to handle communication with workers."""
        loop = asyncio.get_event_loop()

        fd = worker.pipe.fileno()
        loop.add_reader(fd, self.on_read_ready, worker, fd, loop)

    def on_read_ready(
        self, worker: WorkerMetaData, fd: int, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Receive message from a pipe via callback."""
        if worker.pipe.poll():
            try:
                message = worker.pipe.recv()
                self._handle_message(message, worker, fd)
            except EOFError:
                # When DONE or TERMINATED workers are being killed, EOF error is raised
                # Therefore, we need to skip these workers from marking as failed
                if worker.status not in [WorkerStatus.DONE, WorkerStatus.TERMINATED]:
                    msg = Message(
                        MessageType.STATUS, WorkerStatus.FAILED, worker.job_id
                    )
                    self._handle_status(msg, worker.pipe.fileno())

                self._remove_reader(worker)
            except ConnectionResetError:
                # When ConnectionResetError is raised, the pipe is already closed due to worker failure
                # so we only need to ignore this error.
                pass

    def remove_worker(self, wrk_id: str, job_id: str) -> None:
        """Remove worker related data."""
        for k, v in list(self._workers[job_id].items()):
            if wrk_id == v.id:
                del self._workers[job_id][k]

    def _handle_message(
        self, message: Message, worker: WorkerMetaData, fd: int
    ) -> None:
        """Handle received messages."""
        match message.type:
            case MessageType.LOG:
                self._print_message(message.content, worker.process.pid)

            case MessageType.STATUS:
                self._handle_status(message, fd)

            case MessageType.METRICS:
                self._handle_metrics(message, fd)

    def _handle_status(self, message: Message, fd: int) -> None:
        """Handle status update from Workers."""
        if fd not in self._workers[message.job_id]:
            return

        self._update_worker_status(message, fd)

    def _handle_metrics(self, message: Message, fd: int) -> None:
        """Handle metrics update from workers."""
        if fd not in self._workers[message.job_id]:
            return

        wrk = self._workers[message.job_id][fd]
        data = (message.job_id, wrk.id, message.content)

        loop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(self.metrics_q.put(data), loop)

    def _update_worker_status(self, message: Message, fd: int) -> None:
        """Update Worker status."""
        job_id, content = message.job_id, message.content
        wrk = self._workers[job_id][fd]
        msg = WorkerStatusMessage(wrk.id, job_id, content)

        loop = asyncio.get_running_loop()
        asyncio.run_coroutine_threadsafe(self.status_q.put(msg), loop)

        self._workers[job_id][fd].status = content

    def _remove_reader(self, worker: WorkerMetaData) -> None:
        """Remove and close pipe reader."""
        loop = asyncio.get_event_loop()
        loop.remove_reader(worker.pipe.fileno())
        worker.pipe.close()

    def _signal_terminate_wrkrs(
        self,
        job_id: str,
        by_worker_id: bool = False,
        worker_ids=set(),
        msg_type=MessageType.TERMINATE,
    ) -> None:
        """Signal workers that need to be terminated for a given job id.

        If by_worker_id is true, then workers in the worker_ids set
        are terminated.
        """
        terminated = False
        for worker in self._workers[job_id].values():
            if by_worker_id and worker.id not in worker_ids:
                continue

            self._signal_terminate_wrkr(worker, msg_type)
            terminated = True

        if not terminated:
            # no worker is terminated; so no need to log below
            return

        if by_worker_id:
            logger.info(f"workers {worker_ids} for job {job_id} terminated")
        else:
            logger.info(f"all workers for job {job_id} terminated")

    def _signal_terminate_wrkr(
        self, worker: WorkerMetaData, msg_type: MessageType
    ) -> None:
        """Signal a worker that needs to be terminated."""
        valid = [WorkerStatus.READY, WorkerStatus.RUNNING, WorkerStatus.UPDATED]
        if worker.status in valid:
            worker.status = WorkerStatus.TERMINATED

            msg = Message(msg_type, "", worker.job_id)
            self.send(worker, msg)

    def _print_message(self, content: str, process_id: int) -> None:
        """Print received messages."""
        print(f"Process ID: {process_id}, Message: {content}")
