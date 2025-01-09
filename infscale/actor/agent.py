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

"""agent.py."""

import asyncio
import json
import os
import socket

import grpc
import torch
import torch.multiprocessing as mp
from infscale import get_logger
from infscale.actor.job_manager import JobManager
from infscale.actor.job_msg import Message, MessageType, WorkerStatusMessage
from infscale.actor.worker import Worker
from infscale.actor.worker_manager import WorkerManager
from infscale.config import JobConfig, WorkerInfo
from infscale.constants import GRPC_MAX_MESSAGE_LENGTH, HEART_BEAT_PERIOD
from infscale.controller.ctrl_dtype import JobAction
from infscale.monitor.gpu import GpuMonitor
from infscale.proto import management_pb2 as pb2
from infscale.proto import management_pb2_grpc as pb2_grpc
from multiprocess.connection import Pipe

logger = None

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
        global logger
        logger = get_logger(f"{os.getpid()}", f"agent-{id}.log")

        self.id = id
        self.ip_address = self._get_ip_address()
        self.endpoint = endpoint
        self.job_mgr = JobManager()
        self.worker_mgr = WorkerManager()
        self.wrk_ports: dict[int, socket.socket] = dict()

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

    async def _get_worker_status(self) -> None:
        while True:
            status = await self.worker_mgr.status_q.get()
            if status is None:
                continue

            await self.update_worker_status(status)

    async def update_worker_status(self, message: WorkerStatusMessage) -> None:
        worker_status = {
            "job_id": message.job_id,
            "status": message.status.name.lower(),
            "worker_id": message.id,
        }

        req = pb2.Status(worker_status=worker_status)
        await self.stub.update(req)

    async def _init_controller_session(self) -> bool:
        try:
            reg_req = pb2.RegReq(id=self.id, ip=self.ip_address)  # register agent
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

    def _get_ip_address(self) -> str:
        """Get ip address of agent"""
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address

    def _reserve_ports(self, port_count: int) -> list[str]:
        """Reserve available ports based on number of workers."""
        available_ports = []
        while len(available_ports) < port_count:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind((self.ip_address, 0))
                    port = s.getsockname()[1]
                except OSError:
                    pass
                else:
                    self.wrk_ports[port] = s
                    available_ports.append(port)

        return available_ports

    def _release_ports(self, flow_graph: dict[str, list[WorkerInfo]]):
        for wrkr_info_list in flow_graph.values():
            for wrkr_info in wrkr_info_list:
                data_port, ctrl_port = wrkr_info.data_port, wrkr_info.ctrl_port

                if data_port in self.wrk_ports:
                    s = self.wrk_ports[data_port]
                    s.close()

                    del self.wrk_ports[data_port]

                if ctrl_port in self.wrk_ports:
                    s = self.wrk_ports[ctrl_port]
                    s.close()

                    del self.wrk_ports[ctrl_port]

    def _handle_job_action(self, action: pb2.JobAction) -> None:
        """Handle job-related action."""
        match action.type:
            case JobAction.START | JobAction.UPDATE:
                config = JobConfig(**json.loads(action.manifest.decode("utf-8")))
                logger.debug(f"got a new config for job {config.job_id}")

                self.job_mgr.process_config(config)

                self._release_ports(config.flow_graph)

                self._start_workers(config.job_id)

                self._update_workers(config.job_id)

                self._stop_workers(config.job_id)

            case JobAction.STOP:
                self.worker_mgr._signal_terminate_wrkrs(action.job_id)

            case JobAction.SETUP:
                port_count = int.from_bytes(action.manifest, byteorder="big")
                ports = self._reserve_ports(port_count)

                logger.debug(f"ports assigned: {ports} for {port_count / 2} workers")

                req = pb2.JobSetupReq(
                    ports=ports,
                    job_id=action.job_id,
                    agent_id=self.id,
                )
                self.stub.job_setup(req)

    async def heart_beat(self):
        """Send a heart beat message periodically."""
        agent_id = pb2.AgentID(id=self.id)
        while True:
            self.stub.heartbeat(agent_id)
            await asyncio.sleep(HEART_BEAT_PERIOD)

    def _start_workers(self, job_id: str) -> None:
        """Start workers."""
        ctx = mp.get_context("spawn")

        job_config = self.job_mgr.get_config(job_id)
        if not job_config:
            logger.debug(f"no worker to start for job {job_id}")
            return

        start_wrkrs = self.job_mgr.get_workers(job_id)

        for local_rank, config in enumerate(job_config.get_serve_configs()):
            if config.stage.id not in start_wrkrs:
                continue

            pipe, child_pipe = ctx.Pipe()
            process = ctx.Process(
                target=_run_worker,
                args=(local_rank, child_pipe, config.job_id, config.stage.id),
                daemon=True,
            )
            process.start()

            pid, job_id, stage_id = process.pid, config.job_id, config.stage.id

            w = self.worker_mgr.add(pipe, process, job_id, stage_id)

            msg = Message(MessageType.CONFIG, config, config.job_id)
            self.worker_mgr.send(w, msg)

            self.worker_mgr.initialize_listener(w)

            print(f"PID: {pid} - Job ID: {job_id} - Worker: {stage_id}")

    def _update_workers(self, job_id: str) -> None:
        job_config = self.job_mgr.get_config(job_id)
        if not job_config:
            logger.debug(f"no config for job {job_id}")
            return

        update_wrkrs = self.job_mgr.get_workers(job_id, JobAction.UPDATE)
        workers = self.worker_mgr.get_workers(job_id, update_wrkrs)

        logger.debug(f"workers to update: {update_wrkrs}")
        for config in job_config.get_serve_configs():
            if config.stage.id not in update_wrkrs:
                continue

            w = workers[config.stage.id]
            msg = Message(MessageType.CONFIG, config, config.job_id)
            self.worker_mgr.send(w, msg)

    def _stop_workers(self, job_id: str) -> None:
        stop_wrkrs = self.job_mgr.get_workers(job_id, JobAction.STOP)
        self.worker_mgr._signal_terminate_wrkrs(job_id, True, stop_wrkrs)

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

        _ = asyncio.create_task(self._get_worker_status())

        self.monitor()

        # wait forever
        await asyncio.Event().wait()


def _run_worker(local_rank: int, child_pipe: Pipe, job_id: str, wrk_id: str):
    w = Worker(local_rank, child_pipe, job_id, wrk_id)
    w.run()
