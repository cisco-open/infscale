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
from infscale.fwding.factory import Forwarder, get_forwarder
from infscale.worker.error_handler import get_worker_error_handler


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
        self._mc = mc
        self.requests_count = 0

        self._rx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)  # used in pipeline
        self._tx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)  # used in pipeline

        # maintains the tasks of send / recv for worlds
        self._tasks: dict[str, tuple[asyncio.Task, asyncio.Event]] = {}
        self.__tx_qs: dict[int, list[tuple[WorldInfo, asyncio.Queue]]] = {}
        self.__suspended_tx_qs: dict[int, list[tuple[WorldInfo, asyncio.Queue]]] = {}
        self.__rx_q = asyncio.Queue(DEFAULT_QUEUE_SIZE)

        self.orphan_dq: deque = deque()

        _ = asyncio.create_task(self._send_arbiter())
        _ = asyncio.create_task(self._recv_arbiter())

        self._fwder: Forwarder = None
        
        self._error_handler = get_worker_error_handler()

    @property
    def rx_q(self) -> asyncio.Queue:
        """Return receiver queue."""
        return self._rx_q

    @property
    def tx_q(self) -> asyncio.Queue:
        """Return transmit queue."""
        return self._tx_q
    
    def handle_suspended_worlds(self, suspended_worlds: list[str]) -> None:
        """Handle suspended worlds by adding or removing them from the pipeline."""
        if len(suspended_worlds) == 0:
            # revert all suspended connections
            for stage, tuples in self.__suspended_tx_qs.items():
                self.__tx_qs[stage].extend(tuples)

            # reset suspended dict after the recovery is done
            self.__suspended_tx_qs.clear()
            
            logger.info(f"suspended __tx_qs {self.__suspended_tx_qs}")

            return

        for stage, q_list in list(self.__tx_qs.items()):
            remaining = []
            for tpl in q_list:
                world_info, _ = tpl
                if world_info.name in suspended_worlds:
                    if stage not in self.__suspended_tx_qs:
                         self.__suspended_tx_qs[stage] = []

                    self.__suspended_tx_qs[stage].append(tpl)
                else:
                    remaining.append(tpl)

            self.__tx_qs[stage] = remaining

        logger.info(f"suspended __tx_qs {self.__suspended_tx_qs}")

    async def configure(
        self,
        spec: ServeConfig,
        device=torch.device("cpu"),
        worlds_to_add: list[WorldInfo] = [],
        worlds_to_remove: list[WorldInfo] = [],
    ) -> None:
        """(Re)configure router."""
        self.device = device

        if self._fwder is None:
            self._fwder = get_forwarder(spec.fwd_policy)

        # In case of llm, decoding token efficiently needs kv cache.
        # When request is routed to a receiver (worker) for the first time,
        # the destination should be remembered (and reused) so that kv cache
        # can be utilized. Otherwise, the decoding causes error since no
        # kv cache associated with the corresponding request.
        sticky = spec.kv_cache_needed()
        self._fwder.set_stickiness(sticky)

        for world_info in worlds_to_add:
            cancellable = asyncio.Event()
            if world_info.me == 0:  # I am a receiver from other
                task = asyncio.create_task(self._recv(world_info, cancellable))
                self._tasks[world_info.multiworld_name] = (task, cancellable)
            else:  # I am a sender to other
                task = asyncio.create_task(self._send(world_info, cancellable))
                self._tasks[world_info.multiworld_name] = (task, cancellable)

                stage_cfg = spec.workers_stage_info[world_info.other_id]

                if stage_cfg.start not in self.__tx_qs:
                    self.__tx_qs[stage_cfg.start] = []
                    self._fwder.configure(stage_cfg.start)

                tpl = (world_info, asyncio.Queue(DEFAULT_QUEUE_SIZE))
                self.__tx_qs[stage_cfg.start].append(tpl)

        for world_info in worlds_to_remove:
            # do not remove recv tasks yet
            if world_info.me == 0:
                continue

            await self.cleanup_world(world_info)

    async def _recv(self, world_info: WorldInfo, cancellable: asyncio.Event) -> None:
        recv_dev = torch.device("cpu") if world_info.backend == "gloo" else self.device
        receiver = TensorReceiver(
            self.world_manager.communicator,
            world_info.channel,
            world_info.multiworld_name,
            world_info.other,
            recv_dev,
        )

        while True:
            try:
                # task can be cancelled before it receives tensors
                cancellable.set()
                tensors, seqno = await receiver.recv()
                # task can't be cancelled until the tensors are processed
                cancellable.clear()
                self.requests_count += 1
            except Exception as e:
                logger.warning(f"{world_info.multiworld_name} error: {e}")
                break

            if self._mc.can_collect_in_router():
                # update metrics for request
                self._mc.update(seqno)

            if recv_dev != self.device:
                for k in tensors.keys():
                    tensors[k] = tensors[k].to(self.device)

            await self.__rx_q.put((tensors, seqno))

        await self.cleanup_world(world_info, cancel_task=False)

    async def wait_on_term_ready(self) -> None:
        """Wait for pending requests to be processed."""
        while True:
            # wait a second to check the req status
            await asyncio.sleep(1)

            if self.requests_count == 0:
                break

    async def cleanup_world(
        self, world_info: WorldInfo, cancel_task: bool = True
    ) -> None:
        """Cleanup state info on world."""
        # reset tx q related to a given world info
        self._cleanup_tx_q(world_info)

        name = world_info.multiworld_name
        task, cancellable = self._tasks.pop(name, (None, None))

        if not cancel_task or task is None:
            return

        try:
            await cancellable.wait()
            task.cancel()
        except Exception as e:
            logger.warning(f"failed to cancel task for world {name}: {e}")

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
        send_dev = torch.device("cpu") if world_info.backend == "gloo" else self.device
        sender = TensorSender(
            self.world_manager.communicator,
            world_info.channel,
            world_info.multiworld_name,
            world_info.other,
            send_dev,
        )

        tx_q = self._find_tx_q(world_info)
        assert tx_q is not None, f"no tx queqe found for {world_info}"

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
                logger.warning(f"{world_info.multiworld_name} is broken")
                break
            except Exception as e:
                self._error_handler.put(e)

            if send_dev != self.device:
                for k in tensors.keys():
                    tensors[k] = tensors[k].to(send_dev)

            try:
                await sender.send(tensors, seqno)
                self.requests_count -= 1
            except Exception as e:
                cancellable.set()
                logger.warning(f"{world_info.multiworld_name} error: {e}")
                break

            if self._mc.can_collect_in_router():
                # update metrics for request
                self._mc.update(seqno)

        # remove tx queue for the world
        self._cleanup_tx_q(world_info)

        await self._handle_orphan_data(tx_q)

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
        while True:
            try:
                tensor, seqno = await self.__rx_q.get()
                # TODO: introduce a prioritization policy
                await self._rx_q.put((tensor, seqno))
            except Exception as e:
                # this is very likely to be a no-op due to the simple
                # get and put operations we do on the asyncio queues.
                self._error_handler.put(e)

    async def _send_arbiter(self) -> None:
        while True:
            try:
                seqno, tensor, next_layer = self.orphan_dq.popleft()
            except IndexError:
                # In case of this error, we know that there is no orphan
                # data, so we can proceed to fetch data from _tx_q
                seqno, tensor, next_layer = await self._tx_q.get()

            while len(self.__tx_qs[next_layer]) == 0:
                await asyncio.sleep(DEFAULT_SLEEP_TIME)

            tx_qs = self.__tx_qs[next_layer]

            try:
                world_info, tx_q = self._fwder.select(tx_qs, next_layer, seqno)
            except Exception as e:
                self._error_handler.put(e)

            await tx_q.put((seqno, tensor, next_layer))

    async def send(self, seqno: int, data: dict[str, Tensor], next_layer: int) -> None:
        """Send outputs to an appropriate destination."""
        await self._tx_q.put((seqno, data, next_layer))

    async def recv(self) -> tuple[dict[str, Tensor], int]:
        """Receive a dictionary of tensors and sequence number."""
        return await self._rx_q.get()
