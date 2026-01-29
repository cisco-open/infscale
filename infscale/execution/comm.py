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

"""Communication module."""
import torch
from multiworld.communicator import WorldCommunicator

from infscale.execution.control import MSG_MODE_ACK, Channel, ControlMessage


class TensorSender:
    """Tensor sender class."""

    def __init__(
        self,
        communicator: WorldCommunicator,
        channel: Channel,
        world_name: str,
        rank: int,
        device: torch.device,
    ):
        """Initialize tensor sender instance."""
        self.communicator = communicator
        self.channel = channel
        self.world_name = world_name
        self.rank = rank  # destination's rank
        self.device = device

    async def send(self, tensors: dict[str, torch.Tensor], seqno: int) -> None:
        """Send tensors to destination rank.

        tensors: represented as dictionary where key is string and value is tensor
        seqno: the seqno of a tensor; will be used to keep track of tensors
        traversing a pipeline.
        """
        # to minimize the overhead of busy-waiting by communicator's operations
        # we coordinate send/recv via control channel
        _ = await self.channel.sync(self.rank, seqno=seqno, tensors=tensors)

        for _, tensor in tensors.items():
            await self.communicator.send(tensor, self.rank, self.world_name)

    def is_broken(self) -> bool:
        """Check if world is broken or not."""
        return self.communicator.is_broken(self.world_name)


class TensorReceiver:
    """TensorReceiver class."""

    def __init__(
        self,
        communicator: WorldCommunicator,
        channel: Channel,
        world_name: str,
        rank: int,
        device: torch.device,
    ):
        """Initialize communication instance."""
        self.communicator = communicator
        self.channel = channel
        self.world_name = world_name
        self.rank = rank  # source's rank
        self.device = device

        self.buffer: dict[str, torch.Tensor] = None

    async def recv(self) -> tuple[dict[str, torch.Tensor], int]:
        """Receive tensors from source rank.

        seqno: the seqno of a tensor; will be used to keep track of tensors
        traversing a pipeline
        """
        # to minimize the overhead of busy-waiting by communicator's operations
        # we coordinate send/recv via control channel
        ctrl_msg: ControlMessage = await self.channel.sync(self.rank, mode=MSG_MODE_ACK)
        if not ctrl_msg.ditto:
            # since there is change in tensor format, reallocate buffer
            self.buffer = {}
            for k, v in ctrl_msg.metas.items():
                tensor = torch.zeros(
                    v.shape,
                    device=self.device,
                    dtype=v.dtype,
                    requires_grad=False,
                )
                self.buffer[k] = tensor

        for _, tensor in self.buffer.items():
            await self.communicator.recv(tensor, self.rank, self.world_name)

        seqno = ctrl_msg.seqno

        recvd = {}
        for k, v in self.buffer.items():
            recvd[k] = v.detach().clone()

        return recvd, seqno

    def is_broken(self) -> bool:
        """Check if world is broken or not."""
        return self.communicator.is_broken(self.world_name)
