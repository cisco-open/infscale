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

"""Control channel class."""

import asyncio
import pickle
from asyncio import StreamReader, StreamWriter
from dataclasses import dataclass
from typing import Union

import torch
from infscale import get_logger
from torch import Tensor

logger = None


MSG_SIZE = 10000
MSG_MODE_SEND = "send"
MSG_MODE_ACK = "ack"

WAIT_DURATION = 3  # 3 seconds
NUM_OF_RETRIES = 100  # in total, wait for 5 minutes to set up a control channel


@dataclass
class MetaData:
    """MetaMessage dataclass."""

    shape: torch.Size
    dtype: torch.dtype


class ControlMessage:
    """ControlMessage dataclass."""

    def __init__(self):
        """Initialize ControlMessage class."""
        self.seqno: int = 0
        self.ditto: bool = False
        self.metas: dict[str, MetaData] = {}


class Channel:
    """Control Channel class."""

    def __init__(self, rank: int, world_size: int, addr: str, port: int):
        """Initialize an instance."""
        global logger
        logger = get_logger()

        self.rank = rank
        self.world_size = world_size
        self.addr = addr
        self.port = port

        self.peers: dict[int, tuple[StreamReader, StreamWriter]] = {}
        self.prev_ctrl_msg: ControlMessage = ControlMessage()

    async def _setup_server(self, setup_done: asyncio.Event) -> None:
        server = await asyncio.start_server(self._handle_client, self.addr, self.port)

        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        logger.info(f"Serving on {addrs}")

        setup_done.set()
        logger.info("set the setup_done event")

        async with server:
            await server.serve_forever()

    async def _handle_client(self, reader: StreamReader, writer: StreamWriter) -> None:
        data = await reader.read(MSG_SIZE)
        message = data.decode()
        peer_rank = int(message)

        logger.info(f"peer rank: {peer_rank}")
        # save reader and writer streams for peer rank
        self.peers[peer_rank] = (reader, writer)

    async def _setup_client(self, setup_done: asyncio.Event) -> None:
        logger.info(f"setting up a client: {self.addr}:{self.port}")
        for i in range(NUM_OF_RETRIES):
            try:
                reader, writer = await asyncio.open_connection(self.addr, self.port)
            except Exception as e:
                if i + 1 == NUM_OF_RETRIES:
                    logger.warn(f"max number ({i+1}) of tries reached")
                    raise e

                logger.info(f"({i+1}): exception occurred: {e}; retrying...")
                await asyncio.sleep(WAIT_DURATION)
        logger.info("done with setting up a client")

        # send my rank to rank 0
        message = f"{self.rank}"
        writer.write(message.encode())
        await writer.drain()
        logger.info(f"sent rank info({message}) to server")
        # server is always rank 0
        self.peers[0] = (reader, writer)

        setup_done.set()
        logger.info("set the setup_done event")

    async def wait_readiness(self):
        """Wait until control channel is fully configured."""
        if self.rank != 0:
            # this is client, configuration is done during calling self.setup()
            # nothing to do
            return

        while len(self.peers) != self.world_size - 1:
            await asyncio.sleep(1)

    async def setup(self) -> None:
        """Set up the channel."""
        setup_done = asyncio.Event()

        if self.rank == 0:
            _ = asyncio.create_task(self._setup_server(setup_done))
        else:
            _ = asyncio.create_task(self._setup_client(setup_done))

        # wait until setting up either server or client is done
        await setup_done.wait()

        logger.info("channel setup is done")

    async def send_ctrl_msg(
        self, rank: int, tensors: dict[str, Tensor], seqno: int = 0
    ) -> None:
        """Send control information to a receiver."""
        assert tensors is not None, "tensors can't be none"

        msg = ControlMessage()
        msg.seqno = seqno
        for k, tensor in tensors.items():
            shape = tensor.shape
            dtype = tensor.dtype
            msg.metas[k] = MetaData(*[shape, dtype])

        if self.prev_ctrl_msg.metas == msg.metas:
            # no need to duplicate the same control infomation
            # set ditto and send the control message
            msg.metas = {}
            msg.ditto = True
        else:
            self.prev_ctrl_msg = msg

        msg_bytes = pickle.dumps(msg)
        _, writer = self.peers[rank]
        writer.write(msg_bytes)
        await writer.drain()

    async def recv_ctrl_msg(self, rank: int) -> ControlMessage:
        """Receive control information from a sender."""
        reader, _ = self.peers[rank]
        data_bytes = await reader.read(MSG_SIZE)

        msg: ControlMessage = pickle.loads(data_bytes)

        return msg

    async def sync(
        self,
        rank: int,
        mode=MSG_MODE_SEND,
        seqno: int = 0,
        tensors: dict[str, Tensor] = None,
    ) -> Union[ControlMessage, None]:
        """Synchronize send/recv."""
        if mode == MSG_MODE_SEND:
            await self.send_ctrl_msg(rank, tensors, seqno)

            # wait for acknowledgment message
            reader, _ = self.peers[rank]
            _ = await reader.read(MSG_SIZE)

            return None
        else:  # ack
            msg = await self.recv_ctrl_msg(rank)

            # send acknowledgment message
            _, writer = self.peers[rank]
            writer.write(mode.encode())
            await writer.drain()

            return msg
