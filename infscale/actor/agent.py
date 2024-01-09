"""Agent class."""

import asyncio
import multiprocessing as mp
import os
from dataclasses import dataclass
from multiprocessing import connection

import grpc
import torch
from infscale import get_logger
from infscale.actor.worker import Worker
from infscale.constants import GRPC_MAX_MESSAGE_LENGTH, HEART_BEAT_PERIOD
from infscale.proto import management_pb2 as pb2
from infscale.proto import management_pb2_grpc as pb2_grpc

ENV_CUDA_VIS_DEVS = "CUDA_VISIBLE_DEVICES"

logger = get_logger()


@dataclass
class WorkerMetaData:
    """WorkerMetaData dataclass."""

    pipe: connection.Connection
    process: mp.Process


class Agent:
    """Agent class manages workers in a node."""

    def __init__(self, id: str, endpoint: str):
        """Initialize the agent instance."""
        # TODO: there can be more than one worker per GPU
        #       if resource (gpu memory, gpu cycle) are available
        #       explore this possibility later
        # one worker per GPU
        self.id = id
        self.endpoint = endpoint

        self.n_workers = torch.cuda.device_count()
        self._workers: dict[int, WorkerMetaData] = {}

        self.channel = grpc.aio.insecure_channel(
            endpoint,
            options=[
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
            ],
        )

        self.stub = pb2_grpc.ManagementRouteStub(self.channel)

    async def run(self):
        """Start the agent."""
        logger.info("run agent")

        # register agent
        reg_req = pb2.RegReq(id=self.id)
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

        # TODO: revisit launch later; currently it doesn't work
        # self.launch()

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
        ctx = mp.get_context("spawn")

        for local_rank in range(self.n_workers):
            os.environ[ENV_CUDA_VIS_DEVS] = str(local_rank)

            pipe, child_pipe = ctx.Pipe()
            w = Worker(local_rank, child_pipe)
            process = mp.Process(target=w.run, args=(), daemon=True)
            self._workers[local_rank] = WorkerMetaData(pipe, process)

        os.environ.pop(ENV_CUDA_VIS_DEVS)

    def configure(self):
        """Configure workers."""
        pass

    def terminate(self):
        """Terminate workers."""
        logger.info("terminate workers")

        for rank, wmd in self._workers.items():
            logger.info(f"terminate worker {rank}")
            wmd.process.terminate()

    def monitor(self):
        """Monitor workers."""
        pass
