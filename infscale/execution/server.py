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

"""Server class."""

import random

import torch
from infscale import get_logger
from infscale.config import ServeConfig
from infscale.execution.comm import TensorReceiver, TensorSender
from infscale.module.dataset import HuggingFaceDataset
from torch.utils.data import DataLoader

logger = None


class Server:
    """Server class.

    This class is a leader node in the pipeline and
    acts as a proxy between input to the server and model.
    """

    def __init__(
        self, spec: ServeConfig, dataset: HuggingFaceDataset, device: torch.device
    ):
        """Initialize server instance."""
        global logger
        logger = get_logger()

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
