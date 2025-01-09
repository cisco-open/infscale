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
from infscale.config import JobConfig
from infscale.constants import APISERVER_PORT, CONTROLLER_PORT, GRPC_MAX_MESSAGE_LENGTH
from infscale.controller.agent_context import AgentContext
from infscale.controller.apiserver import ApiServer
from infscale.controller.ctrl_dtype import JobAction, JobActionModel, ReqType
from infscale.controller.job_context import JobContext
from infscale.monitor.gpu import GpuMonitor
from infscale.proto import management_pb2 as pb2
from infscale.proto import management_pb2_grpc as pb2_grpc

logger = None

CtrlRequest = Union[Request | JobActionModel]


class Controller:
    """Controller class manages inference services via agents."""

    def __init__(
        self,
        port: int = CONTROLLER_PORT,
        apiport: int = APISERVER_PORT,
    ):
        """Initialize an instance."""
        global logger
        logger = get_logger(f"{os.getpid()}", "controller.log")

        self.port = port

        self.agent_contexts: dict[str, AgentContext] = dict()
        self.job_contexts: dict[str, JobContext] = dict()

        self.apiserver = ApiServer(self, apiport)

        self.job_setup_event = asyncio.Event()

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

        await self.handle_wrk_status(request.worker_status)

        # TODO: use gpu and vram status to schedule deployment

    async def handle_wrk_status(self, worker_status: pb2.WorkerStatus) -> None:
        """Set worker status within job state."""
        if not worker_status.status:
            return

        job_id, status, wrk_id = (
            worker_status.job_id,
            worker_status.status,
            worker_status.worker_id,
        )
        job_ctx = self.job_contexts.get(job_id)

        await job_ctx.set_wrk_status(wrk_id, status)

    def reset_agent_context(self, id: str) -> None:
        """Remove agent context from contexts dictionary."""
        if id not in self.agent_contexts:
            # nothing to do
            return

        context = self.agent_contexts[id]
        context.reset()
        del self.agent_contexts[id]

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

    async def _job_setup(self, agent_id: str, config: JobConfig) -> None:
        """Send job setup request to agent."""
        job_ctx = self.job_contexts.get(config.job_id)

        if job_ctx.num_new_workers <= 0:  # no new workers to ask for ports
            self.job_setup_event.set()

            return

        # we need two sets of ports for each worker for channel connection and multiworld connection
        port_count_bytes = (job_ctx.num_new_workers * 2).to_bytes(1, byteorder="big")

        agent_context = self.agent_contexts[agent_id]
        context = agent_context.get_grpc_ctx()

        payload = pb2.JobAction(
            type=JobAction.SETUP, job_id=config.job_id, manifest=port_count_bytes
        )

        await context.write(payload)

    async def _send_config_to_agent(
        self, agent_id: str, job_id: str, action: JobActionModel
    ) -> None:
        """Send config to agent."""

        agent_context = self.agent_contexts[agent_id]
        context = agent_context.get_grpc_ctx()
        config = self.job_contexts.get(job_id).config

        manifest_bytes = json.dumps(asdict(config)).encode("utf-8")

        payload = pb2.JobAction(
            type=action.action, job_id=job_id, manifest=manifest_bytes
        )

        await context.write(payload)

    async def _send_action_to_agent(
        self, agent_id: str, job_id: str, action: JobActionModel
    ) -> None:
        """Send job action to agent."""

        agent_context = self.agent_contexts[agent_id]
        context = agent_context.get_grpc_ctx()

        payload = pb2.JobAction(type=action.action, job_id=job_id)

        await context.write(payload)

    def handle_job_ports(self, req: pb2.JobSetupReq) -> JobConfig:
        """Patch config with connection info received from agent."""
        job_ctx = self.job_contexts.get(req.job_id)
        job_ctx.set_ports(req.ports)

        self.job_setup_event.set()

    async def _patch_job_cfg(self,agent_id: str, job_id: str) -> None:
        """Patch config for updated job."""
        await self.job_setup_event.wait()

        job_ctx = self.job_contexts.get(job_id)
        config, new_config, ports = job_ctx.config, job_ctx.new_config, job_ctx.ports

        port_iter = None
        if ports is not None:
            # there might be a case when new config has less number of workers
            port_iter = iter(ports)

        # step 1: save current workers from the old config
        curr_workers = {}
        if config is not None:
            for worker_list in config.flow_graph.values():
                for worker in worker_list:
                    curr_workers[worker.name] = worker

        # step 2: patch new config with existing workers ports and assign ports to new ones
        for worker_list in new_config.flow_graph.values():
            for worker in worker_list:
                worker.addr = self.agent_contexts[agent_id].ip
                if worker.name in curr_workers:
                    # keep existing ports
                    worker.data_port = curr_workers[worker.name].data_port
                    worker.ctrl_port = curr_workers[worker.name].ctrl_port
                else:
                    # assign new ports to new workers
                    worker.data_port = next(port_iter)
                    worker.ctrl_port = next(port_iter)

        job_ctx.config = new_config
        job_ctx.new_config = None
        job_ctx.num_new_workers = 0
        job_ctx.ports = None

        # block patch until new config is received
        self.job_setup_event.clear()


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

    async def fetch(
        self, request: pb2.AgentID, context: ServicerContext
    ) -> AsyncIterable[pb2.JobAction]:
        """Push JobAction so that agent can take necessary actions for workers."""
        # since fetch() is used for manifest "push", this function shouldn't be
        # returned. For that, we create an asyncio event and let the event wait
        # forever. The event will be released only when agent is unreachable.
        if request.id not in self.ctrl.agent_contexts:
            logger.debug(f"{request.id} is not in controller'context")
            return
        agent_context = self.ctrl.agent_contexts[request.id]
        agent_context.set_grpc_ctx(context)
        event = agent_context.get_grpc_ctx_event()

        await event.wait()
