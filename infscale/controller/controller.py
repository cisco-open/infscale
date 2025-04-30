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

"""controller.py."""

import asyncio
import json
import os
from dataclasses import asdict
from typing import Any, AsyncIterable

import grpc
from fastapi import Request
from google.protobuf import empty_pb2
from grpc.aio import ServicerContext

from infscale import get_logger
from infscale.common.constants import GRPC_MAX_MESSAGE_LENGTH
from infscale.common.job_msg import WorkerStatus
from infscale.configs.controller import CtrlConfig
from infscale.configs.job import JobConfig, WorkerData
from infscale.controller.agent_context import AgentContext
from infscale.controller.apiserver import ApiServer
from infscale.controller.autoscaler import AutoScaler
from infscale.controller.ctrl_dtype import CommandAction, CommandActionModel, ReqType
from infscale.controller.deployment.factory import DeploymentPolicyFactory
from infscale.controller.job_context import AgentMetaData, JobContext, JobStateEnum
from infscale.controller.planner import Planner
from infscale.monitor.cpu import CpuMonitor
from infscale.monitor.gpu import GpuMonitor
from infscale.proto import management_pb2 as pb2, management_pb2_grpc as pb2_grpc


logger = None

CtrlRequest = Request | CommandActionModel


class Controller:
    """Controller class manages inference services via agents."""

    def __init__(self, config: CtrlConfig):
        """Initialize an instance."""
        global logger
        logger = get_logger(f"{os.getpid()}", "controller.log")

        self.reqgen_config = config.reqgen
        self.port = config.ctrl_port

        self.agent_contexts: dict[str, AgentContext] = dict()
        self.job_contexts: dict[str, JobContext] = dict()

        self.apiserver = ApiServer(self, config.api_port)

        self.deploy_policy = DeploymentPolicyFactory.get_deployment(
            config.deploy_policy
        )

        self.autoscaler = None
        if config.autoscale:
            self.autoscaler = AutoScaler(self)

        self.planner = Planner(config.job_plans, config.autoscale)

    async def _start_server(self):
        server_options = [
            ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
            ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
        ]
        server = grpc.aio.server(options=server_options)
        pb2_grpc.add_ManagementRouteServicer_to_server(ControllerServicer(self), server)

        endpoint = f"[::]:{self.port}"
        _ = server.add_insecure_port(endpoint)

        logger.info(f"serving on {endpoint}")
        await server.start()
        await server.wait_for_termination()

    async def run(self):
        """Run controller."""
        logger.info("starting controller")
        _ = asyncio.create_task(self._start_server())

        if self.autoscaler is not None:
            _ = asyncio.create_task(self.autoscaler.run())

        await self.apiserver.run()

    async def handle_register(self, req: pb2.RegReq) -> tuple[bool, str]:
        """Handle registration message."""
        logger.debug(f"recevied req = {req}")
        if req.id in self.agent_contexts:
            return False, f"{req.id} already registered"

        self.agent_contexts[req.id] = AgentContext(self, req.id, req.ip)
        # since registration is done, let's keep agent context alive
        self.agent_contexts[req.id].keep_alive()

        logger.debug(f"successfully registered {req.id}")

        return True, ""

    async def handle_heartbeat(self, id: str) -> None:
        """Handle heartbeat message."""
        if id not in self.agent_contexts:
            # nothing to do
            return

        self.agent_contexts[id].keep_alive()

    async def handle_resources(self, request: pb2.ResourceStats) -> None:
        """Handle agent resource stats."""
        gpu_stats = GpuMonitor.proto_to_stats(request.gpu_stats)
        logger.debug(f"gpu_stats = {gpu_stats}")

        vram_stats = GpuMonitor.proto_to_stats(request.vram_stats)
        logger.debug(f"vram_stats = {vram_stats}")

        cpu_stats = CpuMonitor.proto_to_stats(request.cpu_stats)
        logger.debug(f"cpu_stats = {cpu_stats}")

        dram_stats = CpuMonitor.proto_to_stats(request.dram_stats)
        logger.debug(f"dram_stats = {dram_stats}")

        agent_context = self.agent_contexts.get(request.id)
        agent_context.update_resource_statistics(
            gpu_stats, vram_stats, cpu_stats, dram_stats
        )

    def handle_job_status(self, request: pb2.JobStatus) -> None:
        """Handle job status message."""
        job_id, status, agent_id = request.job_id, request.status, request.agent_id

        job_ctx = self.job_contexts.get(job_id)
        job_ctx.handle_job_status(status, agent_id)

    async def handle_wrk_status(self, req: pb2.WorkerStatus) -> None:
        """Handle worker status."""
        job_id, status, wrk_id = (
            req.job_id,
            req.status,
            req.worker_id,
        )

        try:
            status_enum = WorkerStatus(status)
            job_ctx = self.job_contexts.get(job_id)
            job_ctx.set_wrk_status(wrk_id, status_enum)

            await job_ctx.do_wrk_cond(wrk_id, status_enum)
        except ValueError:
            logger.warning(f"'{status}' is not a valid WorkerStatus")

    async def handle_metrics(self, req: pb2.PerfMetrics) -> None:
        """Handle performance metrics of a worker."""
        job_ctx = self.job_contexts.get(req.job_id)

        metrics = job_ctx.get_wrkr_metrics(req.worker_id)
        metrics.update(req.qlevel, req.delay, req.thp)

        if not self.autoscaler or not job_ctx.is_server(req.worker_id):
            return

        # if the worker is server, we want to compute change rate of
        # performance metrics (qlevel and thp). So, update_rate() is
        # called.
        metrics.update_rate()

        await self.autoscaler.set_event(req.job_id, req.worker_id)

    def _cleanup_job_ctx(self, agent_id: str) -> None:
        """Cleanup job context by agent id."""
        for k, v in list(self.job_contexts.items()):
            if agent_id in v.agent_info:
                del self.job_contexts[k]

    def reset_agent_context(self, id: str) -> None:
        """Remove agent context from contexts dictionary."""
        if id not in self.agent_contexts:
            # nothing to do
            return

        context = self.agent_contexts[id]
        context.reset()
        del self.agent_contexts[id]
        self._cleanup_job_ctx(id)

    def get_job_status(self, job_id: str) -> JobStateEnum | None:
        """Return current job status."""
        job_ctx = self.job_contexts.get(job_id)

        if not job_ctx:
            return None

        return job_ctx.state_enum.name.lower()

    async def handle_fastapi_request(self, type: ReqType, req: CtrlRequest) -> Any:
        """Handle fastapi request."""
        if type != ReqType.JOB_ACTION:
            logger.debug(f"unknown fastapi request type: {type}")
            return None

        logger.debug(f"got {req.action} request")

        job_id = req.config.job_id if req.config else req.job_id

        if job_id not in self.job_contexts:
            self.job_contexts[job_id] = JobContext(self, job_id)

        job_ctx = self.job_contexts.get(job_id)
        await job_ctx.do(req)

    async def job_setup(self, job_id: str, agent_data: AgentMetaData) -> None:
        """Send a job setup request to agent."""
        agent_id, num_new_worlds, job_setup_event = (
            agent_data.id,
            agent_data.num_new_worlds,
            agent_data.job_setup_event,
        )

        if num_new_worlds <= 0:  # no new worlds to ask for ports
            job_setup_event.set()
            return

        # we need two ports  (channel and multiworld) for each world
        msg = (num_new_worlds * 2).to_bytes(1, byteorder="big")
        payload = pb2.Action(type=CommandAction.SETUP, job_id=job_id, manifest=msg)

        agent_context = self.agent_contexts[agent_id]
        context = agent_context.get_grpc_ctx()
        await context.write(payload)

    async def send_config_to_agent(
        self, agent_id: str, cfg: JobConfig, action: CommandActionModel
    ) -> None:
        """Send config to agent."""
        agent_context = self.agent_contexts[agent_id]
        context = agent_context.get_grpc_ctx()

        manifest_bytes = json.dumps(asdict(cfg)).encode("utf-8")

        payload = pb2.Action(
            type=action.action, job_id=action.job_id, manifest=manifest_bytes
        )

        await context.write(payload)

    async def _send_command_to_agent(
        self, agent_id: str, job_id: str, action: CommandActionModel
    ) -> None:
        """Send command to agent."""
        agent_context = self.agent_contexts[agent_id]
        context = agent_context.get_grpc_ctx()

        payload = pb2.Action(type=action.action, job_id=job_id)

        await context.write(payload)

    def handle_job_ports(self, req: pb2.JobSetupReq) -> JobConfig:
        """Patch config with connection info received from agent."""
        job_ctx = self.job_contexts.get(req.job_id)
        job_ctx.set_ports(req.agent_id, req.ports)

    def _get_deploy_worker_ids(self, workers: list[WorkerData]) -> list[str]:
        """Return a list of worker ids to be deployed."""
        return [w.id for w in workers if w.deploy]


class ControllerServicer(pb2_grpc.ManagementRouteServicer):
    """Controller Servicer class."""

    def __init__(self, ctrl: Controller):
        """Initialize controller servicer."""
        self.ctrl = ctrl

    async def job_setup(
        self, req: pb2.JobSetupReq, unused_context: ServicerContext
    ) -> None:
        """Handle job setup from agent."""
        self.ctrl.handle_job_ports(req)

        return empty_pb2.Empty()

    async def register(
        self, request: pb2.RegReq, unused_context: ServicerContext
    ) -> pb2.RegRes:
        """Register agent in the controller."""
        status, reason = await self.ctrl.handle_register(request)
        return pb2.RegRes(status=status, reason=reason)

    async def heartbeat(
        self, request: pb2.AgentID, unused_context: ServicerContext
    ) -> None:
        """Handle heart beat message from agent."""
        await self.ctrl.handle_heartbeat(request.id)

        return empty_pb2.Empty()

    async def update_wrk_status(
        self, request: pb2.WorkerStatus, unused_context: ServicerContext
    ) -> None:
        """Handle message with worker status."""
        await self.ctrl.handle_wrk_status(request)

        return empty_pb2.Empty()

    async def update_resources(
        self, request: pb2.ResourceStats, unused_context: ServicerContext
    ) -> None:
        """Handle message with agent resource stats."""
        await self.ctrl.handle_resources(request)

        return empty_pb2.Empty()

    async def job_status(
        self, request: pb2.JobStatus, unused_context: ServicerContext
    ) -> None:
        """Handle update message for job status."""
        self.ctrl.handle_job_status(request)

        return empty_pb2.Empty()

    async def command(
        self, request: pb2.AgentID, context: ServicerContext
    ) -> AsyncIterable[pb2.Action]:
        """Push CommandAction so that agent can take necessary actions for workers."""
        # since command() is used for command "push", this function shouldn't be
        # returned. For that, we create an asyncio event and let the event wait
        # forever. The event will be released only when agent is unreachable.
        if request.id not in self.ctrl.agent_contexts:
            logger.debug(f"{request.id} is not in controller'context")
            return
        agent_context = self.ctrl.agent_contexts[request.id]
        agent_context.set_grpc_ctx(context)
        event = agent_context.get_grpc_ctx_event()

        await event.wait()

    async def update_metrics(self, request: pb2.PerfMetrics, unused: ServicerContext):
        """Handle message with job's performance metrics."""
        await self.ctrl.handle_metrics(request)

        return empty_pb2.Empty()
