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

"""Controller class."""
import asyncio
import json
import os
from dataclasses import asdict
from typing import Any, AsyncIterable, Union

import grpc
from fastapi import Request
from google.protobuf import empty_pb2
from grpc.aio import ServicerContext
from infscale import get_logger
from infscale.config import JobConfig, WorkerData
from infscale.constants import (
    APISERVER_PORT,
    CONTROLLER_PORT,
    DEFAULT_DEPLOYMENT_POLICY,
    GRPC_MAX_MESSAGE_LENGTH,
)
from infscale.controller.agent_context import AgentContext, AgentResources
from infscale.controller.apiserver import ApiServer
from infscale.controller.ctrl_dtype import CommandAction, CommandActionModel, ReqType
from infscale.controller.deployment.factory import DeploymentPolicyFactory
from infscale.controller.job_context import AgentMetaData, JobContext
from infscale.controller.deployment.policy import (
    DeploymentPolicyEnum,
)
from infscale.monitor.gpu import GpuMonitor
from infscale.proto import management_pb2 as pb2
from infscale.proto import management_pb2_grpc as pb2_grpc

logger = None

CtrlRequest = Union[Request | CommandActionModel]


class Controller:
    """Controller class manages inference services via agents."""

    def __init__(
        self,
        port: int = CONTROLLER_PORT,
        apiport: int = APISERVER_PORT,
        policy: str = DEFAULT_DEPLOYMENT_POLICY,
    ):
        """Initialize an instance."""
        global logger
        logger = get_logger(f"{os.getpid()}", "controller.log")

        self.port = port

        self.agent_contexts: dict[str, AgentContext] = dict()
        self.job_contexts: dict[str, JobContext] = dict()

        self.apiserver = ApiServer(self, apiport)

        policy_fact = DeploymentPolicyFactory()

        try:
            policy_enum = DeploymentPolicyEnum(policy)
            self.deploy_policy = policy_fact.get_deployment(policy_enum)
        except ValueError:
            logger.warning(
                f"'{policy_enum}' is not a valid deployment policy, continuing with {DeploymentPolicyEnum.EVEN}"
            )
            self.deploy_policy = policy_fact.get_deployment(DeploymentPolicyEnum.EVEN)

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

    async def handle_status(self, request: pb2.Status) -> None:
        """Handle worker status message."""
        gpu_stats = GpuMonitor.proto_to_stats(request.gpu_stats)
        logger.debug(f"gpu_stats = {gpu_stats}")

        vram_stats = GpuMonitor.proto_to_stats(request.vram_stats)
        logger.debug(f"vram_stats = {vram_stats}")

        self.handle_wrk_status(request.worker_status)

        # TODO: use gpu and vram status to schedule deployment

    def handle_job_status(self, request: pb2.JobStatus) -> None:
        """Handle job status message."""
        job_id, status, agent_id = request.job_id, request.status, request.agent_id

        job_ctx = self.job_contexts.get(job_id)
        job_ctx.handle_job_status(status, agent_id)

    async def get_agents_resources(self) -> None:
        for agent_context in self.agent_contexts.values():
            context = agent_context.get_grpc_ctx()

            payload = pb2.Action(type=CommandAction.RESOURCE_STAT)
            await context.write(payload)

    def handle_agent_resources(self, req: pb2.ResourceStats) -> None:
        agent_id, gpu_stats, vram_stats, cpu_stats, dram_stats = (
            req.id,
            req.gpu_stats,
            req.vram_stats,
            req.cpu_stats,
            req.dram_stats,
        )

        resources = AgentResources(
           gpu_stats, vram_stats, cpu_stats, dram_stats
        )

        agent_context = self.agent_contexts.get(agent_id)
        agent_context.resources = resources

    def handle_wrk_status(self, worker_status: pb2.WorkerStatus) -> None:
        """Set worker status within job state."""
        if not worker_status.status:
            return

        job_id, status, wrk_id = (
            worker_status.job_id,
            worker_status.status,
            worker_status.worker_id,
        )
        job_ctx = self.job_contexts.get(job_id)

        job_ctx.set_wrk_status(wrk_id, status)

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

    async def _job_setup(self, agent_data: AgentMetaData) -> None:
        """Send job setup request to agent."""
        agent_id, config, num_new_workers, job_setup_event = (
            agent_data.id,
            agent_data.new_config,
            agent_data.num_new_workers,
            agent_data.job_setup_event,
        )

        if num_new_workers <= 0:  # no new workers to ask for ports
            job_setup_event.set()

            return

        # we need two sets of ports for each worker for channel connection and multiworld connection
        port_count_bytes = (num_new_workers * 2).to_bytes(1, byteorder="big")

        agent_context = self.agent_contexts[agent_id]
        context = agent_context.get_grpc_ctx()

        payload = pb2.Action(
            type=CommandAction.SETUP, job_id=config.job_id, manifest=port_count_bytes
        )

        await context.write(payload)

    async def _send_config_to_agent(
        self, agent_data: AgentMetaData, action: CommandActionModel
    ) -> None:
        """Send config to agent."""
        agent_id, config = agent_data.id, agent_data.config
        agent_context = self.agent_contexts[agent_id]
        context = agent_context.get_grpc_ctx()

        manifest_bytes = json.dumps(asdict(config)).encode("utf-8")

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

    async def update(
        self, request: pb2.Status, unused_context: ServicerContext
    ) -> None:
        """Handle update message for worker status."""
        await self.ctrl.handle_status(request)

        return empty_pb2.Empty()

    async def job_status(
        self, request: pb2.JobStatus, unused_context: ServicerContext
    ) -> None:
        """Handle update message for job status."""
        self.ctrl.handle_job_status(request)

        return empty_pb2.Empty()

    async def put_resource_stat(
        self, request: pb2.ResourceStats, unused_context: ServicerContext
    ) -> None:
        """Handle update message for job status."""
        self.ctrl.handle_agent_resources(request)

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
