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

"""Router class."""
import asyncio
from collections import deque

import torch
from multiworld.manager import WorldManager
from torch import Tensor

from infscale import get_logger
from infscale.configs.job import ServeConfig
from infscale.execution.comm import TensorReceiver, TensorSender
from infscale.execution.metrics_collector import MetricsCollector
from infscale.execution.world import WorldInfo
from infscale.fwding import random, rr, shortest, static


DEFAULT_QUEUE_SIZE = 100
DEFAULT_SLEEP_TIME = 0.1  # 100ms
QUEUE_WAIT_PERIOD = 0.1  # 100ms


logger = None


class Router:
    """Router class."""

    def __init__(self, world_manager: WorldManager, mc: MetricsCollector):
        """Initialize Router instance."""
        global logger
        logger = get_logger()

        self.world_manager = world_manager
        self.mc = mc
        self.requests_count = 0

        self._rx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)  # used in pipeline
        self._tx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)  # used in pipeline

        # maintains the tasks of send / recv for worlds
        self._tasks: dict[str, tuple[asyncio.Task, asyncio.Event]] = {}
        self.__tx_qs: dict[int, list[tuple[WorldInfo, asyncio.Queue]]] = {}
        self.__rx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)

        self.orphan_dq: deque = deque()

        _ = asyncio.create_task(self._send_arbiter())
        _ = asyncio.create_task(self._recv_arbiter())

    def _select_forwarding_policy(self, fwd_policy: str) -> None:
        match fwd_policy:
            case "random":
                logger.info("random forwarding policy selected")
                self._select = random.select

            case "rr":
                logger.info("round-robin forwarding policy selected")
                self._select = rr.select

            case "shortest":
                logger.info("shortest queue length forwarding policy selected")
                self._select = shortest.select

            case "static":
                logger.info("static forwarding policy selected")
                self._select = static.select

            case _:
                raise NotImplementedError(f"{fwd_policy}")

    @property
    def rx_q(self) -> asyncio.Queue:
        """Return receiver queue."""
        return self._rx_q

    @property
    def tx_q(self) -> asyncio.Queue:
        """Return transmit queue."""
        return self._tx_q

    async def configure(
        self,
        spec: ServeConfig,
        device=torch.device("cpu"),
        worlds_to_add: list[WorldInfo] = [],
        worlds_to_remove: list[WorldInfo] = [],
    ) -> None:
        """(Re)configure router."""
        self.device = device

        self._select_forwarding_policy(spec.fwd_policy)

        for world_info in worlds_to_add:
            logger.info(f"world info: {world_info}")
            cancellable = asyncio.Event()
            if world_info.me == 0:  # I am a receiver from other
                task = asyncio.create_task(self._recv(world_info, cancellable))
                self._tasks[world_info.name] = (task, cancellable)
            else:  # I am a sender to other
                task = asyncio.create_task(self._send(world_info, cancellable))
                self._tasks[world_info.name] = (task, cancellable)

                stage_cfg = spec.workers_stage_info[world_info.other_id]

                if stage_cfg.start not in self.__tx_qs:
                    self.__tx_qs[stage_cfg.start] = []

                tpl = (world_info, asyncio.Queue(DEFAULT_QUEUE_SIZE))
                self.__tx_qs[stage_cfg.start].append(tpl)

        for world_info in worlds_to_remove:
            # do not remove recv tasks yet
            if world_info.me == 0:
                continue

            await self._cleanup_world(world_info)

    async def _recv(self, world_info: WorldInfo, cancellable: asyncio.Event) -> None:
        logger.debug(
            f"start to receive tensor from {world_info.other} in world {world_info.name}"
        )
        recv_dev = torch.device("cpu") if world_info.backend == "gloo" else self.device
        receiver = TensorReceiver(
            self.world_manager.communicator,
            world_info.channel,
            world_info.name,
            world_info.other,
            recv_dev,
        )
        logger.debug("created tensor receiver")

        while True:
            try:
                # task can be cancelled before it receives tensors
                cancellable.set()
                tensors, seqno = await receiver.recv()
                # task can't be cancelled until the tensors are processed
                cancellable.clear()
                self.requests_count += 1
            except Exception as e:
                logger.warn(f"{world_info.name} error: {e}")
                break

            if self.mc.can_collect_in_router():
                # update metrics for request
                self.mc.update(seqno)

            if recv_dev != self.device:
                for k in tensors.keys():
                    tensors[k] = tensors[k].to(self.device)

            logger.debug(f"received tensors of seqno {seqno}")
            await self.__rx_q.put((tensors, seqno))
            logger.debug(f"put tensors of seqno {seqno} into __rx_q")

        await self._cleanup_world(world_info, cancel_task=False)
        logger.warn(f"done with recv task for {world_info.name}")

    async def wait_on_term_ready(self) -> None:
        """Wait for pending requests to be processed."""
        while True:
            # wait a second to check the req status
            await asyncio.sleep(1)

            if self.requests_count == 0:
                break

    async def _cleanup_world(
        self, world_info: WorldInfo, cancel_task: bool = True
    ) -> None:
        """Cleanup state info on world."""
        # reset tx q related to a given world info
        self._cleanup_tx_q(world_info)

        name = world_info.name
        task, cancellable = self._tasks.pop(name, (None, None))

        if not cancel_task or task is None:
            return

        try:
            await cancellable.wait()
            task.cancel()
            logger.info(f"canceled task for world {name}")
        except Exception as e:
            logger.error(f"failed to cancel task for world {name}: {e}")

    def _find_tx_q(self, world_info: WorldInfo) -> asyncio.Queue:
        for _, v in self.__tx_qs.items():
            for wi, q in v:
                if wi != world_info:
                    continue

                return q

        return None

    def _cleanup_tx_q(self, world_info: WorldInfo) -> None:
        for _, v in self.__tx_qs.items():
            for i, (wi, q) in enumerate(v):
                if wi != world_info:
                    continue

                del v[i]
                return

    async def _send(self, world_info: WorldInfo, cancellable: asyncio.Event) -> None:
        logger.debug(f"start to send tensor to {world_info.other} in {world_info.name}")
        send_dev = torch.device("cpu") if world_info.backend == "gloo" else self.device
        sender = TensorSender(
            self.world_manager.communicator,
            world_info.channel,
            world_info.name,
            world_info.other,
            send_dev,
        )
        logger.debug("created tensor sender")

        tx_q = self._find_tx_q(world_info)
        assert tx_q is not None, f"no tx queqe found for {world_info}"
        logger.debug("acquired tx q")

        while True:
            try:
                # task can be cancelled before it gets tensors from queue
                cancellable.set()
                seqno, tensors, _ = await asyncio.wait_for(
                    tx_q.get(), QUEUE_WAIT_PERIOD
                )
                # task can't be cancelled until the tensors are processed
                # (until they are sent to other side, i.e., peer)
                cancellable.clear()
            except asyncio.TimeoutError:
                if not sender.is_broken():
                    continue
                logger.warn(f"{world_info.name} is broken")
                break

            if send_dev != self.device:
                for k in tensors.keys():
                    tensors[k] = tensors[k].to(send_dev)

            logger.debug(f"got tensors of seqno {seqno} from __tx_q")
            try:
                await sender.send(tensors, seqno)
                self.requests_count -= 1
            except Exception as e:
                logger.warn(f"{world_info.name} error: {e}")
                break
            logger.debug(f"sent tensors of seqno {seqno}")

            if self.mc.can_collect_in_router():
                # update metrics for request
                self.mc.update(seqno)

        # remove tx queue for the world
        self._cleanup_tx_q(world_info)

        await self._handle_orphan_data(tx_q)

        logger.warn(f"done with send task for {world_info.name}")

    async def _handle_orphan_data(self, queue: asyncio.Queue) -> None:
        async def _drain_and_put_inner():
            while not queue.empty():
                # drain queue
                data = await queue.get()
                # put data into orphan deque so that data can be resent
                self.orphan_dq.append(data)

        await _drain_and_put_inner()

        # we give some time for _send_arbiter to finish any blocked "put" call
        await asyncio.sleep(DEFAULT_SLEEP_TIME)

        # we call this one more time
        await _drain_and_put_inner()

    async def _recv_arbiter(self) -> None:
        logger.debug("start recv_arbiter")
        while True:
            tensor, seqno = await self.__rx_q.get()
            logger.debug(f"fetched tensor {seqno} from __rx_q")
            # TODO: introduce a prioritization policy
            await self._rx_q.put((tensor, seqno))
            logger.debug("put tensor to _rx_q")

    async def _send_arbiter(self) -> None:
        logger.debug("start send_arbiter")
        while True:
            try:
                seqno, tensor, next_layer = self.orphan_dq.popleft()
                logger.debug(f"fetched tensor {seqno} from orhpan_dq")
            except IndexError:
                # In case of this error, we know that there is no orphan
                # data, so we can proceed to fetch data from _tx_q
                seqno, tensor, next_layer = await self._tx_q.get()
                logger.debug(f"fetched tensor {seqno} from _tx_q")

            while len(self.__tx_qs[next_layer]) == 0:
                logger.warn("no worker to send data")
                await asyncio.sleep(DEFAULT_SLEEP_TIME)

            tx_qs = self.__tx_qs[next_layer]
            world_info, tx_q = self._select(tx_qs)

            logger.debug(f"world name: {world_info.name}")
            logger.debug(f"receiver rank: {world_info.other}")

            await tx_q.put((seqno, tensor, next_layer))
            logger.debug(
                f"put tensor {seqno} to __tx_q for {world_info.other} in {world_info.name}"
            )

    async def send(self, seqno: int, data: dict[str, Tensor], next_layer: int) -> None:
        """Send outputs to an appropriate destination."""
        await self._tx_q.put((seqno, data, next_layer))

    async def recv(self) -> tuple[dict[str, Tensor], int]:
        """Receive a dictionary of tensors and sequence number."""
        return await self._rx_q.get()
