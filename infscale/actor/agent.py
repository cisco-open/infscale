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

"""Agent class."""

import asyncio
import json

import grpc
import torch
import torch.multiprocessing as mp
from infscale import get_logger
from infscale.actor.config_diff import get_config_diff_ids
from infscale.actor.job_manager import JobManager, WorkerMetaData
from infscale.actor.job_msg import Message, MessageType, WorkerStatus
from infscale.actor.worker import Worker
from infscale.config import JobConfig
from infscale.constants import GRPC_MAX_MESSAGE_LENGTH, HEART_BEAT_PERIOD
from infscale.controller.apiserver import JobAction
from infscale.monitor.gpu import GpuMonitor
from infscale.proto import management_pb2 as pb2
from infscale.proto import management_pb2_grpc as pb2_grpc
from multiprocess.connection import Pipe

logger = get_logger()

service_config_json = json.dumps(
    {
        "methodConfig": [
            {
                # To apply retry to all methods, put [{}] in the "name" field
                "name": [
                    {
                        "service": "management.ManagementRoute",
                        "method": "register",
                    }
                ],
                "retryPolicy": {
                    "maxAttempts": 3,
                    "initialBackoff": "1s",
                    "maxBackoff": "10s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE"],
                },
            }
        ]
    }
)


class Agent:
    """Agent class manages workers in a node."""

    def __init__(
        self,
        id: str,
        endpoint: str,
    ):
        """Initialize the agent instance."""
        # TODO: there can be more than one worker per GPU
        #       if resource (gpu memory, gpu cycle) are available
        #       explore this possibility later
        # one worker per GPU

        self.id = id
        self.endpoint = endpoint
        self.job_config = None
        self.job_manager = JobManager()

        self._workers: dict[int, WorkerMetaData] = {}
        self.n_workers = torch.cuda.device_count()

        self.channel = grpc.aio.insecure_channel(
            endpoint,
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.service_config", service_config_json),
            ],
        )

        self.stub = pb2_grpc.ManagementRouteStub(self.channel)

        self.gpu_monitor = GpuMonitor()

    async def _init_controller_session(self) -> bool:
        try:
            reg_req = pb2.RegReq(id=self.id)  # register agent
            reg_res = await self.stub.register(reg_req)
        except grpc.aio.AioRpcError as e:
            logger.debug(f"can't register: {e}")
            return False

        if not reg_res.status:
            logger.error(f"registration failed: {reg_res.reason}")
            return False

        # create a task to send heart beat periodically
        _ = asyncio.create_task(self.heart_beat())

        # create a task to send status in an event-driven fashion
        _ = asyncio.create_task(self.report())

        # create a task to wait for fetch job action
        _ = asyncio.create_task(self.fetch())

        return True

    async def fetch(self) -> None:
        """Connect to the server and start the listening task."""
        try:
            await self._fetch()
        except Exception as e:
            logger.error(f"Error in connection: {e}")

    async def _fetch(self) -> None:
        """Listen for job action pushes from the ManagementRoute."""
        request = pb2.AgentID(id=self.id)

        async for action in self.stub.fetch(request):
            if not action:
                continue

            self._handle_job_action(action)

    def _handle_job_action(self, action: pb2.JobAction) -> None:
        """Handle job-related action."""
        match action.type:
            case JobAction.START | JobAction.UPDATE:
                config = JobConfig(**json.loads(action.manifest.decode("utf-8")))
                self._handle_config(config)

            case JobAction.STOP:
                self.job_manager._terminate_workers(action.job_id)

    def _handle_config(self, config: JobConfig) -> None:
        """Handle configuration file received from controller."""
        logger.debug(f"got new config: {config}")

        if self.job_manager.job_exists(config.job_id):
            old_config = self.job_manager.get_job_config(config.job_id)

            terminate_ids, start_ids, updated_ids = get_config_diff_ids(
                old_config, config
            )
            self.kill_workers(terminate_ids)
            self.reconfigure_job(config, start_ids, updated_ids)
            self.job_manager.set_job_config(config)

            return

        self.job_manager.set_job_config(config)

        self.launch(config)

    def kill_workers(self, workers) -> None:
        """Terminate workers whose IDs are in extra_in_a_ids."""
        for worker in self._workers.values():
            if worker.id in workers:
                print(f"Terminating worker with ID: {worker.id}")

    def reconfigure_job(
        self, job_config: JobConfig, start_ids: list[int], update_ids: list[int]
    ) -> None:
        """Reconfigure workers with new config."""
        for _, config in enumerate(job_config.get_serve_configs()):
            if config.stage.id in start_ids:
                print(f"worker {config.stage.id} needs to be started")

            if config.stage.id in update_ids:
                print(f"worker {config.stage.id} is updated")

    async def heart_beat(self):
        """Send a heart beat message periodically."""
        agent_id = pb2.AgentID(id=self.id)
        while True:
            self.stub.heartbeat(agent_id)
            await asyncio.sleep(HEART_BEAT_PERIOD)

    def launch(self, job_config: JobConfig):
        """Launch workers."""
        ctx = mp.get_context("spawn")

        for local_rank, config in enumerate(job_config.get_serve_configs()):
            pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(
                target=_run_worker,
                args=(
                    local_rank,
                    child_pipe,
                ),
                daemon=True,
            )
            process.start()

            pid, job_id, stage_id = process.pid, config.job_id, config.stage.id

            w = WorkerMetaData(pipe, process, WorkerStatus.READY, stage_id, job_id)
            self._workers[w.pipe.fileno()] = w
            self.job_manager.add_worker(w)

            msg = Message(MessageType.CONFIG, config, config.job_id)
            self.job_manager.send_message_to_worker(w, msg)

            print(f"PID: {pid} - Job ID: {job_id} - Worker: {stage_id}")

        self.job_manager.message_listener()

    def configure(self):
        """Configure workers."""
        pass

    async def report(self):
        """Report status about resources and workers to controller."""
        while True:
            gpu_stats, vram_stats = await self.gpu_monitor.metrics()
            gpu_msg_list = GpuMonitor.stats_to_proto(gpu_stats)
            vram_msg_list = GpuMonitor.stats_to_proto(vram_stats)

            status_msg = pb2.Status()
            status_msg.id = self.id
            status_msg.gpu_stats.extend(gpu_msg_list)
            status_msg.vram_stats.extend(vram_msg_list)
            # TODO: set cpu stat and ram stat into status message

            self.stub.update(status_msg)

    def monitor(self):
        """Monitor workers and resources."""
        _ = asyncio.create_task(self._monitor_gpu())
        # TODO: (priority: high) monitor workers
        # TODO: (priority: low) monitor cpu resources (cpu and ram)

    async def _monitor_gpu(self):
        await self.gpu_monitor.start()

    async def run(self):
        """Start the agent."""
        logger.info("run agent")

        if not await self._init_controller_session():
            return

        self.monitor()

        # wait forever
        await asyncio.Event().wait()


def _run_worker(local_rank: int, child_pipe: Pipe):
    w = Worker(local_rank, child_pipe)
    w.run()
