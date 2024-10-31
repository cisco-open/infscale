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
from infscale.controller.controller import Controller
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
        self, id: str, endpoint: str, use_controller: bool, controller: Controller
    ):
        """Initialize the agent instance."""
        # TODO: there can be more than one worker per GPU
        #       if resource (gpu memory, gpu cycle) are available
        #       explore this possibility later
        # one worker per GPU

        self.id = id
        self.endpoint = endpoint
        self.job_config = None
        self.use_controller = use_controller
        self.controller = controller
        self.job_manager = JobManager()
        self.cfg_event = asyncio.Event()
        self.controller = controller

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
        if not self.use_controller:
            return True

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

        return True

    async def run(self):
        """Start the agent."""
        logger.info("run agent")
        _ = asyncio.create_task(self.controller.run())

        if not await self._init_controller_session():
            return

        self.monitor()

        _ = asyncio.create_task(self.handle_config())
        await self.cfg_event.wait()
        # TODO: revisit launch later
        #       launch may need to be executed whenever manifest is fetched
        self.launch()

        # wait forever
        await asyncio.Event().wait()

    async def handle_config(self) -> None:
        """Handle configuration file received from controller."""
        while True:
            new_config = await self.controller.config_q.get()
            print(f"got new config: {new_config}")
            if new_config is None:
                continue

            if self.job_config:
                terminate_ids, start_ids, updated_ids = get_config_diff_ids(
                    self.job_config, new_config
                )
                self.kill_workers(terminate_ids)
                self.reconfigure_job(new_config, start_ids, updated_ids)
                self.job_config = new_config

            if self.job_config is None:
                self.job_config = new_config

            self.cfg_event.set()

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

    def launch(self):
        """Launch workers."""
        ctx = mp.get_context("spawn")

        for local_rank, config in enumerate(self.job_config.get_serve_configs()):
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
            w = WorkerMetaData(pipe, process, WorkerStatus.READY, config.stage.id)
            self._workers[w.pipe.fileno()] = w
            self.job_manager.add_worker(w)
            self.job_manager.send_message(w, Message(MessageType.CONFIG, config))

            print(f"Process ID: {process.pid} - Worker: {config.stage.id}")

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


def _run_worker(local_rank: int, child_pipe: Pipe):
    w = Worker(local_rank, child_pipe)
    w.run()
