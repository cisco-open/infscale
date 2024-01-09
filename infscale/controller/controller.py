"""Controller class."""

from typing import AsyncIterable

import grpc
from grpc.aio import ServicerContext
from infscale import get_logger
from infscale.constants import CONTROLLER_PORT, GRPC_MAX_MESSAGE_LENGTH
from infscale.controller.agent_context import AgentContext
from infscale.proto import management_pb2 as pb2
from infscale.proto import management_pb2_grpc as pb2_grpc

logger = get_logger()


class Controller:
    """Controller class manages inference services via agents."""

    def __init__(self, port: int = CONTROLLER_PORT):
        """Initialize an instance."""
        self.port = port

        self.contexts: dict[str, AgentContext] = dict()

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
        await self._start_server()

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
        # TODO: implement this
        logger.info("not implemented")

    def reset_agent_context(self, id: str) -> None:
        """Remove agent context from contexts dictionary."""
        if id not in self.contexts:
            # nothing to do
            return

        context = self.contexts[id]
        context.reset()
        del self.contexts[id]


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

    async def update(
        self, request: pb2.Status, unused_context: ServicerContext
    ) -> None:
        """Handle update message for worker status."""
        await self.ctrl.handle_status(request)

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
