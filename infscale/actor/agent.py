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

import grpc
import torch
import torch.multiprocessing as mp
from multiprocess.connection import Pipe

from infscale import get_logger
from infscale.actor.job_manager import JobManager, WorkerMetaData
from infscale.actor.job_msg import Message, MessageType, WorkerStatus
from infscale.actor.worker import Worker
from infscale.config import JobConfig, ServeConfig
from infscale.constants import GRPC_MAX_MESSAGE_LENGTH, HEART_BEAT_PERIOD
from infscale.monitor.gpu import GpuMonitor
from infscale.proto import management_pb2 as pb2
from infscale.proto import management_pb2_grpc as pb2_grpc

logger = get_logger()


class Agent:
    """Agent class manages workers in a node."""

    def __init__(
        self, id: str, endpoint: str, job_config: JobConfig, skip_controller: bool
    ):
        """Initialize the agent instance."""
        # TODO: there can be more than one worker per GPU
        #       if resource (gpu memory, gpu cycle) are available
        #       explore this possibility later
        # one worker per GPU

        self.id = id
        self.endpoint = endpoint
        self.job_config = job_config
        self.skip_controller = skip_controller

        self.n_workers = torch.cuda.device_count()

        self.channel = grpc.aio.insecure_channel(
            endpoint,
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
            ],
        )

        self.stub = pb2_grpc.ManagementRouteStub(self.channel)

        self.gpu_monitor = GpuMonitor()

    async def run(self):
        """Start the agent."""
        logger.info("run agent")

        if not self.skip_controller:
            reg_req = pb2.RegReq(id=self.id)  # register agent

            try:
                reg_res = await self.stub.register(reg_req)
            except grpc.aio.AioRpcError:
                logger.debug("can't proceed: no grpc channel available")
                return

            if not reg_res.status:
                logger.error(f"registration failed: {reg_res.reason}")
                return

        # create a task to send heart beat periodically
        _ = asyncio.create_task(self.heart_beat())

        # create a task to send status in an event-driven fashion
        _ = asyncio.create_task(self.report())

        self.monitor()

        # TODO: revisit launch later
        #       launch may need to be executed whenever manifest is fetched
        self.launch()

        # wait forever
        await asyncio.Event().wait()

    async def heart_beat(self):
        """Send a heart beat message periodically."""
        agent_id = pb2.AgentID(id=self.id)
        while True:
            self.stub.heartbeat(agent_id)
            await asyncio.sleep(HEART_BEAT_PERIOD)

    def launch(self):
        """Launch workers."""
        job_manager = JobManager()

        # create a task to monitor the job
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
            w = WorkerMetaData(pipe, process, WorkerStatus.READY)
            job_manager.add_worker(w)
            job_manager.send_message(w, Message(MessageType.CONFIG, config))

            print(f"Process ID: {process.pid}")

        job_manager.message_listener()

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
