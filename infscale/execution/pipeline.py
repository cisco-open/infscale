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

"""Pipeline class."""

import asyncio
import time

import torch
from infscale import get_logger
from infscale.actor.job_msg import Message, MessageType, WorkerStatus
from infscale.actor.worker_manager import WorkerManager
from infscale.config import ServeConfig, WorkerInfo
from infscale.execution.control import Channel as CtrlCh
from infscale.execution.router import Router
from infscale.execution.stage import Stage
from infscale.execution.world import WorldInfo
from infscale.module.dataset import HuggingFaceDataset
from infscale.module.modelir import ModelIR
from infscale.module.zoo import Zoo
from multiworld.manager import WorldManager

logger = get_logger()


class Pipeline:
    """Pipeline class."""

    def __init__(
        self,
        worker_manager: WorkerManager,
    ):
        """Initialize pipeline instance."""
        self.stage: Stage = None
        self.world_manager = WorldManager()
        self.worker_manager = worker_manager
        self.spec: ServeConfig = None
        self.device = None
        self.world_info_list: list[WorldInfo] = list()
        self.cfg_event = asyncio.Event()

        # TODO: these variables are only for a server (i.e., dispatcher)
        #       need to consider refactoring pipeline such that server code
        #       and worker code are managed in a separate file.
        self.n_inflight = 0
        self.tx_allow_evt = asyncio.Event()
        self.tx_allow_evt.set()

    async def _initialize_multiworld(self):
        my_id = self.spec.stage.id

        for k, v in self.spec.flow_graph.items():
            for wrk_info in v:
                assert len(wrk_info.peers) == 1

                if my_id == k:
                    my_rank = 0
                elif my_id in wrk_info.peers:
                    my_rank = wrk_info.peers.index(my_id) + 1
                else:
                    continue

                name, backend, addr, port = (
                    wrk_info.name,
                    wrk_info.backend,
                    wrk_info.addr,
                    wrk_info.port,
                )
                world_size = len(wrk_info.peers) + 1

                logger.info(f"initializing world {name} with my rank {my_rank}")
                logger.info(f"leader addr={addr}, port={port}")
                await self.world_manager.initialize_world(
                    name,
                    my_rank,
                    world_size,
                    backend=backend,
                    addr=addr,
                    port=port,
                )
                logger.debug(f"done initializing multiworld {name}")

    async def _initialize_control_channel(self):
        async def _inner(
            my_id: str,
            my_rank: int,
            other_id: str,
            other_rank: int,
            wrk_info: WorkerInfo,
        ):
            name, backend, addr, port = (
                wrk_info.name,
                wrk_info.backend,
                wrk_info.addr,
                wrk_info.port,
            )
            world_size = len(wrk_info.peers) + 1

            # increment port number by 1
            port = port + 1
            ctrl_ch = CtrlCh(my_rank, world_size, addr, port)
            await ctrl_ch.setup()
            data = {
                "name": name,
                "my_id": my_id,
                "me": my_rank,
                "other_id": other_id,
                "other": other_rank,
                "backend": backend,
                "channel": ctrl_ch,
            }
            world_info = WorldInfo(**data)
            self.world_info_list.append(world_info)
            logger.debug(f"done initializing control channel for {name}")

        my_id = self.spec.stage.id

        # initialize server first
        for k, v in self.spec.flow_graph.items():
            if my_id != k:
                continue

            my_rank = 0
            other_rank = 1
            for wrk_info in v:
                assert len(wrk_info.peers) == 1
                other_id = wrk_info.peers[0]
                await _inner(my_id, my_rank, other_id, other_rank, wrk_info)

        # initialize client next
        for k, v in self.spec.flow_graph.items():
            for wrk_info in v:
                assert len(wrk_info.peers) == 1
                if my_id not in wrk_info.peers:
                    continue

                my_rank = wrk_info.peers.index(my_id) + 1
                other_rank = 0
                other_id = k
                await _inner(my_id, my_rank, other_id, other_rank, wrk_info)

        for world_info in self.world_info_list:
            await world_info.channel.wait_readiness()
            logger.info(f"control channel for {world_info.name} is ready")

    async def _initialize_pipeline(self):
        await self._initialize_multiworld()
        await self._initialize_control_channel()

    def _initialize_worker(self, modelir: ModelIR):
        self.stage = Stage(
            self.spec.stage.id,
            modelir=modelir,
            start=self.spec.stage.start,
            end=self.spec.stage.end,
            device=self.device,
            max_inflight=self.max_inflight,
        )

    async def _wait_tx_permission(self):
        await self.tx_allow_evt.wait()
        self.n_inflight += 1
        if self.n_inflight == self.max_inflight:
            self.tx_allow_evt.clear()

    async def _check_n_enable_tx_permission(self):
        self.n_inflight -= 1
        if self.n_inflight < self.max_inflight:
            self.tx_allow_evt.set()

    async def _server_send(self, router: Router):
        logger.info("start to send requests")
        seqno = 0
        while True:
            batch = self.dataset.next_batch(self.device)
            if batch is None:
                break

            await self._wait_tx_permission()

            logger.info(f"sending batch {seqno}")
            # send batch to the first stage
            await router.send(seqno, batch, 0)
            seqno += 1
        logger.info("_server_send task done")

    async def _server_recv(self, router: Router, max_count: int = -1):
        """
        Receive inference results from the last stage.

        max_count: if it's -1, run forever;
                   get out of loop if the number of responses becomes max_count
        """
        logger.info("start to receive responses")
        seqno = -1
        idx = 0
        start_time = None
        self.worker_manager.send_message(
            Message(
                MessageType.STATUS,
                WorkerStatus.RUNNING,
            )
        )
        while max_count == -1 or max_count > idx:
            logger.debug("waiting for response")
            outputs, seqno = await router.recv()
            results = self._predict_fn(outputs)
            logger.info(f"response for {seqno}: {results}")

            await self._check_n_enable_tx_permission()

            if idx % 100 == 0:
                if start_time is None:
                    start_time = time.perf_counter()

            idx += 1

        end_time = time.perf_counter()
        print(f"Server recv done, elapsed time: {end_time - start_time}")
        self.worker_manager.send_message(
            Message(
                MessageType.STATUS,
                WorkerStatus.DONE,
            )
        )

        logger.info("_server_recv task done")

    async def _run_server(self):
        # create router
        router = Router(
            self.world_manager, self.world_info_list, self.spec, self.device
        )
        router.prepare()

        # TODO: we read data directly from a dataset right now.
        #       in the future, we need to take dataset from stream as well.
        self.dataset.set_micro_batch_size(self.spec.micro_batch_size)
        max_count = self.dataset.num_of_batches()

        # send and recv asynchronously
        send_task = asyncio.create_task(self._server_send(router))
        recv_task = asyncio.create_task(self._server_recv(router, max_count))
        logger.debug("created _send_request and _recv_resp tasks")

        await asyncio.gather(*[send_task, recv_task])
        logger.info("inference serving is done")

    async def _run_worker(self):
        logger.debug("start to run worker")
        router = Router(
            self.world_manager, self.world_info_list, self.spec, self.device
        )
        router.prepare()
        while True:
            inputs, seqno = await router.recv()
            logger.debug(f"received input {seqno} from router")

            with torch.inference_mode():
                outputs, next_layer = self.stage.predict(seqno, **inputs)

            logger.debug("got output from stage and put output into router")
            await router.send(seqno, outputs, next_layer)
            logger.debug("put output into router")

    async def handle_config(self) -> None:
        """Handle a config sent by the controller."""
        while True:
            spec = await self.worker_manager.config_q.get()

            if spec is None:
                continue

            self.spec = spec
            # TODO: this is not the best place to initialize an instance
            #       variable for readability; let's handle this later
            self.max_inflight: int = self.spec.max_inflight

            self._init_assets()
            self._prepare_worker()
            await self._initialize_pipeline()
            self.cfg_event.set()

    def _init_assets(self) -> None:
        # load model meta info from zoo
        mmd = Zoo.get_model_metadata(self.spec.model)
        (path, name, split) = (
            self.spec.dataset.path,
            self.spec.dataset.name,
            self.spec.dataset.split,
        )

        # load dataset
        self.dataset = HuggingFaceDataset(mmd, path, name, split)
        self.device = torch.device(self.spec.device)

        # load model intermediate representation
        self.modelir = ModelIR(mmd)

    def _prepare_worker(self) -> None:
        if self.spec.stage.is_server:
            logger.info("I am server and leader")
            self.dataset = self.dataset
            self._predict_fn = self.modelir.predict_fn
        else:
            logger.info("I am a worker")
            self._initialize_worker(self.modelir)

    async def _run(self) -> None:
        await self.cfg_event.wait()

        if self.spec.stage.is_server:
            await self._run_server()
        else:
            await self._run_worker()

    async def run(self):
        """Run pipeline."""
        self.worker_manager.send_message(
            Message(
                MessageType.STATUS,
                WorkerStatus.RUNNING,
            )
        )
        _ = asyncio.create_task(self.handle_config())
        await self._run()
