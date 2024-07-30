"""Server class."""

import random

import torch
from infscale import get_logger
from infscale.config import ServeConfig
from infscale.execution.comm import TensorReceiver, TensorSender
from infscale.module.dataset import HuggingFaceDataset
from torch.utils.data import DataLoader

logger = get_logger()


class Server:
    """Server class.

    This class is a leader node in the pipeline and
    acts as a proxy between input to the server and model.
    """

    def __init__(
        self, spec: ServeConfig, dataset: HuggingFaceDataset, device: torch.device
    ):
        """Initialize server instance."""
        self.next_stages: set[str] = set()
        self.prev_stages: set[str] = set()

        for stage in spec.flow_graph[spec.stage.id]:
            self.next_stages.add(stage)
        for src, stages in spec.flow_graph.items():
            for stage in stages:
                if stage != spec.stage.id:
                    continue
                self.prev_stages.add(src)

        self.rank_map = spec.rank_map

        self.dataset = dataset
        self.micro_batch_size = spec.micro_batch_size
        self.device = device

        self.senders: dict[str, TensorSender] = dict()
        self.receivers: dict[str, TensorReceiver] = dict()

        for stage in self.next_stages:
            rank = self.rank_map[stage]
            self.senders[stage] = TensorSender(rank, self.device)

        for stage in self.prev_stages:
            rank = self.rank_map[stage]
            self.receivers[stage] = TensorReceiver(rank, self.device)

    def run(self):
        """Serve inference requests."""
        dataloader = DataLoader(self.dataset.dataset, self.micro_batch_size)
        data_iter = iter(dataloader)

        index = 0
        while True:
            try:
                batch = next(data_iter)

                # send a batch to one of next stages
                # TODO: need more forwarding strategies / logic
                # TODO: make send and receive ops asynchronous
                dst = random.sample(self.next_stages, 1)
                self.senders[dst].send(batch, index)
                index += 1

                # TODO: this only works when there is only one last stage.
                src = random.sample(self.prev_stages, 1)
                res_idx, result = self.receivers[src].recv()
                logger.info(f">>> received tensors {result}")
            except StopIteration:
                logger.debug(f"done: processed {res_idx+1} batches")
                break
