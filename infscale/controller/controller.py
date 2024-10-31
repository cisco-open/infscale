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
from typing import Any, AsyncIterable, Union

import grpc
from fastapi import Request
from google.protobuf import empty_pb2
from grpc.aio import ServicerContext
from infscale import get_logger
from infscale.constants import (APISERVER_PORT, CONTROLLER_PORT,
                                GRPC_MAX_MESSAGE_LENGTH)
from infscale.controller.agent_context import AgentContext
from infscale.controller.apiserver import (ApiServer, JobAction,
                                           JobActionModel, ReqType)
from infscale.monitor.gpu import GpuMonitor
from infscale.proto import management_pb2 as pb2
from infscale.proto import management_pb2_grpc as pb2_grpc

logger = get_logger()

CtrlRequest = Union[Request | JobActionModel]


class Controller:
    """Controller class manages inference services via agents."""

    def __init__(
        self,
        port: int = CONTROLLER_PORT,
        apiport: int = APISERVER_PORT,
    ):
        """Initialize an instance."""
        self.port = port

        self.contexts: dict[str, AgentContext] = dict()

        self.apiserver = ApiServer(self, apiport)

        self.config_q = asyncio.Queue()

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

    async def handle_register(self, id: str) -> tuple[bool, str]:
        """Handle registration message."""
        logger.debug(f"recevied id = {id}")
        if id in self.contexts:
            return False, f"{id} already registered"

        self.contexts[id] = AgentContext(self, id)
        # since registration is done, let's keep agent context alive
        self.contexts[id].keep_alive()

        logger.debug(f"successfully registered {id}")

        return True, ""

    async def handle_heartbeat(self, id: str) -> None:
        """Handle heartbeat message."""
        if id not in self.contexts:
            # nothing to do
            return

        self.contexts[id].keep_alive()

    async def handle_status(self, request: pb2.Status) -> None:
        """Handle worker status message."""
        gpu_stats = GpuMonitor.proto_to_stats(request.gpu_stats)
        logger.debug(f"gpu_stats = {gpu_stats}")

        vram_stats = GpuMonitor.proto_to_stats(request.vram_stats)
        logger.debug(f"vram_stats = {vram_stats}")

        # TODO: use gpu and vram status to schedule deployment

    def reset_agent_context(self, id: str) -> None:
        """Remove agent context from contexts dictionary."""
        if id not in self.contexts:
            # nothing to do
            return

        context = self.contexts[id]
        context.reset()
        del self.contexts[id]

    async def handle_fastapi_request(self, type: ReqType, req: CtrlRequest) -> Any:
        """Handle fastapi request."""
        match type:
            case ReqType.SERVE:
                logger.debug("got request serve")
                return await self._handle_fastapi_serve(req)

            case ReqType.JOB_ACTION:
                logger.debug("got start job request")
                return await self._handle_fastapi_job_action(req)

            case _:
                logger.debug(f"unknown fastapi request type: {type}")
                return None

    async def _handle_fastapi_job_action(self, req: JobActionModel) -> None:
        """Handle fastapi job action request."""
        match req.action:
            case JobAction.UPDATE | JobAction.START:
                await self.config_q.put(req.config)

            case JobAction.STOP:
                # TODO: notify agent to stop the job
                print("stopping job")

    async def _handle_fastapi_serve(self, req: Request):
        logger.debug(f"req = {req}")


class ControllerServicer(pb2_grpc.ManagementRouteServicer):
    """Controller Servicer class."""

    def __init__(self, ctrl: Controller):
        """Initialize controller servicer."""
        self.ctrl = ctrl

    async def register(
        self, request: pb2.RegReq, unused_context: ServicerContext
    ) -> pb2.RegRes:
        """Register agent in the controller."""
        status, reason = await self.ctrl.handle_register(request.id)
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
    ) -> AsyncIterable[pb2.Manifest]:
        """Push manifest to agent to configure worker for inference service."""
        # since fetch() is used for manifest "push", this function shouldn't be
        # returned. For that, we create an asyncio event and let the event wait
        # forever. The event will be released only when agent is unreachable.
        if request.id not in self.ctrl.contexts:
            logger.debug(f"{request.id} is not in controller'context")
            return
        agent_context = self.ctrl.contexts[request.id]
        agent_context.set_grpc_ctx(context)
        event = agent_context.get_grpc_ctx_event()

        # blocked until event is set until set_grpc_ctx_event() is called
        await event
