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
from dataclasses import asdict
from typing import Any, AsyncIterable, Union

import grpc
from fastapi import HTTPException, Request, status
from google.protobuf import empty_pb2
from grpc.aio import ServicerContext
from infscale import get_logger
from infscale.constants import (APISERVER_PORT, CONTROLLER_PORT,
                                GRPC_MAX_MESSAGE_LENGTH)
from infscale.controller.agent_context import AgentContext
from infscale.controller.apiserver import ApiServer
from infscale.controller.ctrl_dtype import JobAction, JobActionModel, ReqType
from infscale.controller.job_state import JobState
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

        self.jobs_state = JobState()

        self.job_action_q = asyncio.Queue()

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
        self.jobs_state.set_agent(id)

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
        if type != ReqType.JOB_ACTION:
            logger.debug(f"unknown fastapi request type: {type}")
            return None

        logger.debug(f"got {req.action} request")

        job_id = req.config.job_id if req.config else req.job_id

        if not self.jobs_state.job_exists(job_id) and req.action in [
            JobAction.STOP,
            JobAction.UPDATE,
        ]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"job with id: '{job_id}' was not found",
            )

        if not self.jobs_state.can_update_job_state(job_id, req.action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Action '{req.action}' on job '{job_id}' is not allowed",
            )

        self.jobs_state.set_job_state(job_id, req.action)

        await self.job_action_q.put(req)

        match req.action:
            case JobAction.UPDATE | JobAction.START:
                pass

            case JobAction.STOP:
                self.jobs_state.remove_job(job_id)


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
    ) -> AsyncIterable[pb2.JobAction]:
        """Push JobAction so that agent can take necessary actions for workers."""
        # since fetch() is used for manifest "push", this function shouldn't be
        # returned. For that, we create an asyncio event and let the event wait
        # forever. The event will be released only when agent is unreachable.
        if request.id not in self.ctrl.contexts:
            logger.debug(f"{request.id} is not in controller'context")
            return
        agent_context = self.ctrl.contexts[request.id]
        agent_context.set_grpc_ctx(context)
        event = agent_context.get_grpc_ctx_event()

        while True:
            action: JobActionModel = await self.ctrl.job_action_q.get()
            if not action:
                continue

            job_id = action.config.job_id if action.config else action.job_id

            if action.config:
                manifest = json.dumps(asdict(action.config)).encode("utf-8")
                payload = pb2.JobAction(
                    type=action.action, job_id=job_id, manifest=manifest
                )
            else:
                payload = pb2.JobAction(type=action.action, job_id=job_id)

            yield payload
