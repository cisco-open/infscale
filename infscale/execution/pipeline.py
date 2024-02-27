"""Pipeline class."""

import asyncio
import os
import random
from collections.abc import Iterator

import torch
import torch.distributed as dist
from infscale import get_logger
from infscale.config import ServeConfig
from infscale.execution.comm import TensorReceiver, TensorSender
from infscale.execution.stage import Stage
from infscale.module.dataset import HuggingFaceDataset
from infscale.module.modelir import ModelIR
from torch.utils.data import DataLoader

logger = get_logger()

MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

DEFAULT_ROUTER_QUEUE_SIZE = 3


class Pipeline:
    """Pipeline class."""

    def __init__(
        self,
        spec: ServeConfig,
        modelir: ModelIR,
        dataset: HuggingFaceDataset,
    ):
        """Initialize pipeline instance."""
        self.stage: Stage = None
        self.spec = spec

        self._initialize_distributed()

        if "s" in spec.stage.id:  # it's server
            self.dataset = dataset
            self._run = self._run_server
            logger.info("I am server and master")
        else:
            logger.info("I am a worker")
            self._run = self._run_worker
            self._initialize_worker(spec, modelir)

    def _initialize_distributed(self, backend: str = "gloo"):
        # TODO: revise the hard-coded values and configure it dynamically
        os.environ["MASTER_ADDR"] = MASTER_ADDR
        os.environ["MASTER_PORT"] = MASTER_PORT

        # TODO: using torch.distributed is not fault-tolerant;
        #       replace it with elatic horovod's functionality
        rank = self.spec.rank_map[self.spec.stage.id]
        size = len(self.spec.rank_map)
        dist.init_process_group(backend, rank=rank, world_size=size)
        logger.info("initializing distributed: done")

    def _initialize_worker(self, spec: ServeConfig, modelir: ModelIR):
        (start, end, my_id) = (
            spec.stage.start,
            spec.stage.end,
            spec.stage.id,
        )

        layers = modelir.layers[start : end + 1]

        self.stage = Stage(my_id, layers)

    async def _run_server(self):
        async def _send_request(router: Router, data_iter: Iterator):
            seqno = 0
            while True:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break

                # send batch to the first stage
                await router.tx_q.put((batch, seqno))
                seqno += 1

        async def _recv_resp(router: Router, max_seqno: int = -1):
            """
            Receive inference results from the last stage.

            max_seqno: if it's -1, run forever;
                       come out of loop if seqno is max_seqno
            """
            seqno = -1
            while max_seqno == -1 or max_seqno != seqno:
                outputs, seqno = await router.rx_q.get()
                logger.info(f"{seqno}: {outputs}")

        # create router
        router = Router(self.spec)
        router.prepare()

        # TODO: we read data directly from a dataset right now.
        #       in the future, we need to take dataset from stream as well.
        dataset_size = len(self.dataset.dataset)
        dataloader = DataLoader(
            self.dataset.dataset,
            self.spec.micro_batch_size,
            collate_fn=self.dataset.data_collator,
        )
        data_iter = iter(dataloader)

        # send and recv asynchronously
        send_task = asyncio.create_task(_send_request(router, data_iter))
        recv_task = asyncio.create_task(_recv_resp(router, dataset_size - 1))
        await asyncio.wait([send_task, recv_task])

    async def _run_worker(self):
        router = Router(self.spec)
        router.prepare()
        while True:
            inputs, index = await router.rx_q.get()
            outputs = self.stage(inputs)
            await router.tx_q.put((outputs, index))

    async def run(self):
        """Run pipeline."""
        await self._run()


class Router:
    """Router class."""

    def __init__(self, spec: ServeConfig):
        """Initialize Router instance."""
        self._rx_q = asyncio.Queue(DEFAULT_ROUTER_QUEUE_SIZE)  # used in pipeline
        self._tx_q = asyncio.Queue(DEFAULT_ROUTER_QUEUE_SIZE)  # used in pipeline

        # a collection of ranks that receive data from me
        self.receiver_ranks: dict[int] = []
        self.__tx_qs: dict[int, asyncio.Queue] = {}

        # a collection of ranks that send data to me
        self.sender_ranks: list[int] = []
        self.__rx_q = asyncio.Queue(DEFAULT_ROUTER_QUEUE_SIZE)

        my_id = spec.stage.id
        self.rank = spec.rank_map[my_id]
        for k, v in spec.flow_graph.items():
            if my_id == k:  # I am a sender to v
                for other_id in v:
                    rank = spec.rank_map[other_id]
                    self.receiver_ranks.append(rank)
                    self.__tx_qs[rank] = asyncio.Queue(DEFAULT_ROUTER_QUEUE_SIZE)
            elif my_id in v:  # I am a receiver from k
                self.sender_ranks.append(spec.rank_map[k])

    @property
    def rx_q(self):
        """Return receiver queue."""
        return self._rx_q

    @property
    def tx_q(self):
        """Return transmit queue."""
        return self._tx_q

    def prepare(self):
        """Create asyncio tasks for sending and receiving."""
        for rank in self.receiver_ranks:
            # TODO: revise hard-coded device
            _ = asyncio.create_task(self._send(rank, torch.device("cpu")))

        for rank in self.sender_ranks:
            # TODO: revise hard-coded device
            _ = asyncio.create_task(self._recv(rank, torch.device("cpu")))

    async def _recv(self, rank: int, device: torch.device):
        receiver = TensorReceiver(rank, device)
        while True:
            tensor, index = receiver.recv()
            await self.__rx_q.put((tensor, index))

    async def _send(self, rank: int, device: torch.device):
        sender = TensorSender(rank, device)
        tx_q = self.__tx_qs[rank]

        while True:
            tensor, index = await tx_q.get()
            sender.send(tensor, index)

    async def _recv_arbiter(self):
        while True:
            tensor, index = await self.__rx_q.get()
            # TODO: introduce a prioritization policy
            await self._rx_q.put((tensor, index))

    async def _send_arbiter(self):
        while True:
            tensor, index = await self._tx_q.get()
            # TODO: introduce a prioritization policy
            #       current default policy is to choose receiving rank randomly

            # TODO: choosing a rank randomly by converting dictionary keys into
            #       a list can be a performance bottleneck; look into it later.
            rank = random.choice(list(self.__tx_qs.keys()))

            self.__tx_qs[rank].put((tensor, index))
