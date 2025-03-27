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

"""pipeline.py."""

import asyncio
import os
import sys
import time

import torch
from infscale import get_logger
from infscale.common.job_msg import Message, MessageType, WorkerStatus
from infscale.config import ServeConfig
from infscale.execution.control import Channel as CtrlCh
from infscale.execution.metrics_collector import MetricsCollector
from infscale.execution.router import Router
from infscale.execution.stage import Stage
from infscale.execution.world import WorldInfo
from infscale.module.dataset import HuggingFaceDataset
from infscale.module.modelir import ModelIR
from infscale.module.zoo import Zoo
from infscale.worker.worker_comm import WorkerCommunicator
from multiworld.manager import WorldManager

logger = None

# a global variable to store start time of the first request
start_time = None


class Pipeline:
    """Pipeline class."""

    def __init__(
        self,
        job_id: str,
        wcomm: WorkerCommunicator,
    ):
        """Initialize pipeline instance."""
        global logger
        logger = get_logger()

        self.stage: Stage = None
        self.mc = MetricsCollector()
        self.world_manager = WorldManager()
        self.router = Router(self.world_manager, self.mc)
        self.job_id = job_id
        self.wcomm = wcomm
        self.spec: ServeConfig = None
        self.device = None
        self.world_infos: dict[str, WorldInfo] = {}
        self.cfg_event = asyncio.Event()
        self._initialized = False

        # TODO: these variables are only for a server (i.e., dispatcher)
        #       need to consider refactoring pipeline such that server code
        #       and worker code are managed in a separate file.
        self.n_inflight = 0
        self.tx_allow_evt = asyncio.Event()
        self.tx_allow_evt.set()

    async def _configure_multiworld(self, world_info: WorldInfo) -> None:
        (name, world_size, addr, port, backend, my_rank) = (
            world_info.name,
            world_info.size,
            world_info.addr,
            world_info.port,
            world_info.backend,
            world_info.me,
        )

        logger.info(f"configuring world {name} of size {world_size}")
        logger.info(f"my rank: {my_rank} backend: {backend}")
        logger.info(f"leader addr={addr}, port={port}")
        try:
            await self.world_manager.initialize_world(
                name,
                my_rank,
                world_size,
                backend=backend,
                addr=addr,
                port=port,
                device=self.device,
            )
        except Exception as e:
            logger.error(f"failed to initialize a multiworld {name}: {e}")
            return

        logger.debug(f"done initializing multiworld {name}")

    def _send_status_message(self, status: WorkerStatus) -> None:
        msg = Message(MessageType.STATUS, status, self.spec.job_id)
        self.wcomm.send(msg)

    async def _configure_control_channel(self, world_info: WorldInfo) -> None:
        await world_info.channel.setup()
        logger.debug(f"done configuring control channel for {world_info.name}")

        await world_info.channel.wait_readiness()
        logger.info(f"control channel for {world_info.name} is ready")

    def _reset_multiworld(self, world_info: WorldInfo) -> None:
        # TODO: implement this
        logger.info(f"remove world {world_info.name} from multiworld")

    def _reset_control_channel(self, world_info: WorldInfo) -> None:
        # TODO: implement this
        logger.info(f"remove world {world_info.name} from control channel")

    async def _configure(self) -> None:
        """(Re)configure multiworld, control channel and router."""
        new_world_infos = self._build_world_infos()
        new = new_world_infos.keys()
        cur = self.world_infos.keys()

        worlds_to_add = [new_world_infos[name] for name in new - cur]
        worlds_to_remove = [self.world_infos[name] for name in cur - new]

        # handle new worlds
        tasks = []
        # 1. set up control channel
        for world_info in worlds_to_add:
            task = self._configure_control_channel(world_info)
            tasks.append(task)

        # TODO: this doesn't handle partial success
        #       a mechanism to handle a failure is left as a todo
        await asyncio.gather(*tasks)

        tasks = []
        # 2. set up multiworld
        for world_info in worlds_to_add:
            task = self._configure_multiworld(world_info)
            tasks.append(task)

        # TODO: this doesn't handle partial success
        #       a mechanism to handle a failure is left as a todo
        await asyncio.gather(*tasks)

        # update world_info for added worlds
        for world_info in worlds_to_add:
            self.world_infos[world_info.name] = world_info

        # configure router with worlds to add and remove
        self.router.configure(self.spec, self.device, worlds_to_add, worlds_to_remove)

        # handle unnecessary world
        # remove is executed in the reverse order of add
        for world_info in worlds_to_remove:
            logger.info(f"remove world {world_info.name}")
            # 1. remove unnecessary world from control channel
            self._reset_control_channel(world_info)
            # 2. remove unnecessary world from multiworld
            self._reset_multiworld(world_info)

            del self.world_infos[world_info.name]

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
        global start_time

        logger.info("start to send requests")
        seqno = 0
        start_time = time.perf_counter()
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
        global start_time

        logger.info("start to receive responses")
        seqno = -1
        idx = 0
        while max_count == -1 or max_count > idx:
            logger.debug("waiting for response")
            outputs, seqno = await router.recv()
            results = self._predict_fn(outputs)
            logger.info(f"response for {seqno}: {results}")

            await self._check_n_enable_tx_permission()

            idx += 1

        end_time = time.perf_counter()
        print(
            f"Server recv done, Job: {self.spec.job_id} elapsed time: {end_time - start_time}"
        )

        self._send_status_message(WorkerStatus.DONE)

        logger.info("_server_recv task done")

    async def _run_server(self):
        # TODO: we read data directly from a dataset right now.
        #       in the future, we need to take dataset from stream as well.
        self.dataset.set_micro_batch_size(self.spec.micro_batch_size)
        max_count = self.dataset.num_of_batches()

        # send and recv asynchronously
        send_task = asyncio.create_task(self._server_send(self.router))
        recv_task = asyncio.create_task(self._server_recv(self.router, max_count))

        await asyncio.gather(*[send_task, recv_task])
        logger.info("inference serving is done")

    async def _run_worker(self):
        logger.debug("start to run worker")
        while True:
            inputs, seqno = await self.router.recv()

            with torch.inference_mode():
                outputs, next_layer = self.stage.predict(seqno, **inputs)

            await self.router.send(seqno, outputs, next_layer)

    async def _collect_metrics(self):
        # TODO: more changes will follow
        while True:
            await asyncio.sleep(1)
            qlevel, delay, thp = self.mc.retrieve()
            logger.info(f"qlevel = {qlevel}, delay = {delay}, thp = {thp}")

    async def _handle_message(self) -> None:
        """Handle a message from an agent."""
        while True:
            msg = await self.wcomm.recv()

            match msg.type:
                case MessageType.CONFIG:
                    await self._handle_config(msg.content)

                case MessageType.TERMINATE:
                    # TODO: do the clean-up / caching before termination
                    status = WorkerStatus.TERMINATED
                    resp = Message(MessageType.STATUS, status, self.job_id)
                    self.wcomm.send(resp)
                    logger.info("worker is terminated")

                    sys.stdout.flush()
                    # TODO: This forcibly terminates the entire process.
                    #       This is not graceful. Revisit this later.
                    os._exit(0)

                case MessageType.FINISH_JOB:
                    # TODO: do the clean-up before transitioning to DONE
                    status = WorkerStatus.DONE
                    resp = Message(MessageType.STATUS, status, self.job_id)
                    self.wcomm.send(resp)
                    logger.info("worker is done")

                    sys.stdout.flush()
                    # TODO: This forcibly terminates the entire process.
                    #       This is not graceful. Revisit this later.
                    os._exit(0)

    async def _handle_config(self, spec: ServeConfig) -> None:
        """Handle a config."""
        if spec is None:
            return

        self._configure_variables(spec)

        self._initialize_once()

        # (re)configure the pipeline
        await self._configure()

        self.cfg_event.set()

        self._send_status_message(WorkerStatus.RUNNING)

    def _build_world_infos(self) -> dict[str, WorldInfo]:
        world_infos: dict[str, WorldInfo] = {}

        my_id = self.spec.stage.id
        for k, v in self.spec.flow_graph.items():
            for cfg_world_info in v:
                # NOTE: no. of peers is always 1 for now
                assert len(cfg_world_info.peers) == 1

                if my_id == k:
                    my_rank = 0
                    other_rank = 1
                    other_id = cfg_world_info.peers[0]
                elif my_id in cfg_world_info.peers:
                    # NOTE: this is always 1 for now
                    my_rank = cfg_world_info.peers.index(my_id) + 1
                    other_rank = 0
                    other_id = k
                else:
                    continue

                name, backend, addr, data_port, ctrl_port = (
                    cfg_world_info.name,
                    cfg_world_info.backend,
                    cfg_world_info.addr,
                    cfg_world_info.data_port,
                    cfg_world_info.ctrl_port,
                )

                world_size = len(cfg_world_info.peers) + 1
                ctrl_ch = CtrlCh(my_rank, world_size, addr, ctrl_port)

                data = {
                    "name": name,
                    "size": world_size,
                    "addr": addr,
                    "port": data_port,
                    "backend": backend,
                    "channel": ctrl_ch,
                    "my_id": my_id,
                    "me": my_rank,
                    "other_id": other_id,
                    "other": other_rank,
                }
                world_info = WorldInfo(**data)
                world_infos[name] = world_info

        return world_infos

    def _configure_variables(self, spec: ServeConfig) -> None:
        """Set variables that need to be updated."""
        self.spec = spec
        self.max_inflight = spec.max_inflight

    def _initialize_once(self) -> None:
        if self._initialized:
            return

        self._init_assets()
        self._prepare_worker()

        self._initialized = True

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
        if self.spec.is_server:
            logger.info("I am server and leader")
            self.dataset = self.dataset
            self._predict_fn = self.modelir.predict_fn
        else:
            logger.info("I am a worker")
            self._initialize_worker(self.modelir)

    async def run(self) -> None:
        """Run pipeline."""
        _ = asyncio.create_task(self._collect_metrics())
        _ = asyncio.create_task(self._handle_message())
        await self.cfg_event.wait()

        if self.spec.is_server:
            await self._run_server()
        else:
            await self._run_worker()
