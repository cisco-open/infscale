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
from multiworld.manager import WorldManager

from infscale import get_logger
from infscale.common.job_msg import Message, MessageType, WorkerStatus
from infscale.configs.job import ServeConfig
from infscale.execution.config_manager import ConfigManager
from infscale.execution.metrics_collector import MetricsCollector
from infscale.execution.router import Router
from infscale.execution.stage import Stage
from infscale.execution.world import WorldInfo
from infscale.module.dataset import HuggingFaceDataset
from infscale.module.modelir import ModelIR
from infscale.module.zoo import Zoo
from infscale.request.generator import GeneratorFactory
from infscale.worker.fatal import kill_worker
from infscale.worker.worker_comm import WorkerCommunicator


METRICS_INTERVAL = 1  # one second

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

        self._stage: Stage = None
        self._mc = MetricsCollector()
        self._world_manager = WorldManager()
        self._router = Router(self._world_manager, self._mc)
        self._config_manager = ConfigManager()
        self._job_id = job_id
        self._wcomm = wcomm
        self._device = None
        self._initial_cfg_event = asyncio.Event()
        self._micro_batch_size = 1
        self._initialized = False
        self._status: WorkerStatus = WorkerStatus.READY

        # TODO: these variables are only for a server (i.e., dispatcher)
        #       need to consider refactoring pipeline such that server code
        #       and worker code are managed in a separate file.
        self.n_inflight = 0
        self.tx_allow_evt = asyncio.Event()
        self.tx_allow_evt.set()

    async def _configure_multiworld(self, world_info: WorldInfo) -> None:
        (name, world_size, addr, port, backend, my_rank) = (
            world_info.multiworld_name,
            world_info.size,
            world_info.addr,
            world_info.port,
            world_info.backend,
            world_info.me,
        )

        try:
            await self._world_manager.initialize_world(
                name,
                my_rank,
                world_size,
                backend=backend,
                addr=addr,
                port=port,
                device=self._device,
            )
        except asyncio.CancelledError:
            logger.warning(f"multiworld configuration cancelled for {world_info.name}")
        except Exception as e:
            logger.error(f"failed to initialize multiworld {name}: {e}")
            condition = self._status != WorkerStatus.UPDATING
            kill_worker(e, condition)

            return

        logger.info(f"initialized multiworld {name}")

    def _set_worker_status(self, status: WorkerStatus) -> None:
        """Set worker status in pipeline and channel."""
        self._status = status

        world_infos = self._config_manager.get_curr_world_infos()

        for world_info in world_infos.values():
            world_info.channel.set_worker_status(status)

    def _set_n_send_worker_status(self, status: WorkerStatus) -> None:
        """Set and send worker status."""
        self._set_worker_status(status)
        msg = Message(MessageType.STATUS, status, self._job_id)
        self._wcomm.send(msg)

    async def _configure_control_channel(self, world_info: WorldInfo) -> None:
        try:
            await world_info.channel.setup()

            await world_info.channel.wait_readiness()
        except asyncio.CancelledError:
            logger.warning(f"channel configuration cancelled for {world_info}")

    async def _cleanup_recovered_worlds(self) -> None:
        """Clean up world infos for recovered worlds."""
        world_infos = self._config_manager.get_curr_world_infos()

        # if I'm the recovered worker, return
        if len(world_infos) == 0:
            return

        recover_worlds = self._config_manager.get_worlds_to_recover()

        # no worlds to recover
        if len(recover_worlds) == 0:
            return

        for world_info in recover_worlds:
            wi = world_infos.get(world_info.name, None)

            await self._router.cleanup_world(wi)

            self._config_manager.remove_world_info(wi.name)

    async def _configure(self) -> None:
        """(Re)configure multiworld, control channel and router."""
        await self._cleanup_recovered_worlds()

        is_first_run = self._config_manager.is_first_run()

        if not is_first_run:
            self._set_worker_status(WorkerStatus.UPDATING)

        world_names_to_add, world_names_to_remove = (
            self._config_manager.get_worlds_to_add_and_remove()
        )

        tasks = []
        # 1. set up control channel
        for world_name in world_names_to_add - self._config_manager.worlds_to_cancel:
            world_info = self._config_manager.get_new_world_info(world_name)

            task = self._config_manager.schedule_world_cfg(
                world_info, self._configure_control_channel
            )
            tasks.append(task)

        # TODO: this doesn't handle partial success
        #       a mechanism to handle a failure is left as a todo
        await asyncio.gather(*tasks)

        tasks = []
        # 2. set up multiworld
        for world_name in world_names_to_add - self._config_manager.worlds_to_cancel:
            world_info = self._config_manager.get_new_world_info(world_name)
            task = self._config_manager.schedule_world_cfg(
                world_info, self._configure_multiworld
            )
            tasks.append(task)

        # TODO: this doesn't handle partial success
        #       a mechanism to handle a failure is left as a todo
        await asyncio.gather(*tasks)

        # update world_info for added worlds
        self._config_manager.update_world_infos(
            world_names_to_add - self._config_manager.worlds_to_cancel
        )

        worlds_to_add = self._config_manager.get_worlds_to_add(
            world_names_to_add - self._config_manager.worlds_to_cancel
        )

        worlds_to_remove = self._config_manager.get_worlds_to_remove(
            world_names_to_remove
        )

        spec = self._config_manager.get_spec()
        # configure router with worlds to add and remove
        await self._router.configure(
            spec,
            self._device,
            worlds_to_add,
            worlds_to_remove,
        )

        # handle unnecessary world
        # remove is executed in the reverse order of add
        for world_info in worlds_to_remove:
            # cleanup of control channel and multiworld was moved into router
            # since we need to do async world cleanup based on certain scenarios
            # sender can do the cleanup when new config is processed to stop
            # sending requests to failed / removed worker
            # received needs to keep waiting for requests until an exception is raised

            self._config_manager.remove_world_info(world_info.name)

        if not self._config_manager.has_pending_cfg:
            # only update and send worker status if there is no pending request
            worker_status = WorkerStatus.RUNNING if is_first_run else WorkerStatus.UPDATED
            self._set_n_send_worker_status(worker_status)

        self._config_manager.unblock_next_config()

        self._initial_cfg_event.set()

    def _initialize_worker(self, modelir: ModelIR, spec: ServeConfig):
        self._stage = Stage(
            spec.stage.id,
            modelir=modelir,
            start=spec.stage.start,
            end=spec.stage.end,
            device=self._device,
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

    def _reset_inflight_and_tx_event(self) -> None:
        """Reset inflight and tx event.

        For recovery to work properly, when a new config is received,
        we need to reset the n_inflight count and un-bock the send event.
        This happens due to requests loss during recovery, when the server
        continues to send requests to the failed worker / pipeline, before it
        gets notified about the failure, blocking any further requests sending
        due to the maximum number of inflight requests.
        """

        self.n_inflight = 0
        self.tx_allow_evt.set()

    async def _server_send(self, router: Router):
        global start_time

        self._seqno = 0
        self._end_of_send = False

        async def _inner_send(batches: list[torch.Tensor | None]) -> None:
            for batch, is_last in batches:
                if is_last:
                    self._end_of_send = True

                await self._wait_tx_permission()

                # send batch to the first stage
                await router.send(self._seqno, batch, 0)
                self._seqno += 1

        start_time = time.perf_counter()
        while True:
            try:
                batches = await self.req_generator.get()

                await _inner_send(batches)
                if self._end_of_send:
                    break
            except Exception as e:
                # this is very likely a no-op due to the actions that are happening
                # either in inner_send or generator get, but we keep it as a safety net
                kill_worker(e)

    async def _server_recv(self, router: Router):
        """Receive inference results from the last stage."""
        global start_time

        count = 0
        while not self._end_of_send or self._seqno > count:
            outputs, seqno = await router.recv()
            results = self._predict_fn(outputs)
            logger.info(f"response for {seqno}: {results}")

            self._mc.update(seqno)

            await self._check_n_enable_tx_permission()

            count += 1

        end_time = time.perf_counter()
        print(
            f"Server recv done, Job: {self._job_id} elapsed time: {end_time - start_time}"
        )

        self._set_n_send_worker_status(WorkerStatus.SERVING_DONE)

    async def _run_server(self):
        # we disable metrics collection in router in case the worker is server
        # so that we can collect metrics at _server_send and _server_recv tasks
        self._mc.enable_in_router(False)

        spec = self._config_manager.get_spec()

        # TODO: we read data directly from a dataset right now.
        #       in the future, we need to take dataset from stream as well.
        # Loading dataset with some settings might take some time and block
        # the main thread until is done, making coroutines blocked as well,
        # blocking worker to receive messages and run other async processes.
        # For this we need to run configure() in a thread so the event loop stays responsive
        await asyncio.to_thread(
            self.dataset.configure,
            self._device,
            spec.reqgen_config.params.in_memory,
            spec.reqgen_config.params.replay,
        )

        self.req_generator = GeneratorFactory.get(spec.reqgen_config.sort)
        self.req_generator.initialize(
            self.dataset,
            spec.reqgen_config.params,
            self._micro_batch_size,
            self._mc,
        )

        # send and recv asynchronously
        send_task = asyncio.create_task(self._server_send(self._router))
        recv_task = asyncio.create_task(self._server_recv(self._router))

        await asyncio.gather(*[send_task, recv_task])

        logger.info("inference serving is done")

        # wait forever
        await asyncio.Event().wait()

    async def _run_worker(self):
        while True:
            inputs, seqno = await self._router.recv()
            with torch.inference_mode():
                outputs, next_layer = self._stage.predict(seqno, **inputs)
            await self._router.send(seqno, outputs, next_layer)

    async def _collect_metrics(self):
        while True:
            metrics = self._mc.retrieve()
            msg = Message(MessageType.METRICS, metrics, self._job_id)
            self._wcomm.send(msg)

            # wait for an interval
            await asyncio.sleep(METRICS_INTERVAL)

    def _terminate_worker(self) -> None:
        """Terminate worker."""
        status = WorkerStatus.TERMINATED
        resp = Message(MessageType.STATUS, status, self._job_id)
        self._wcomm.send(resp)

        sys.stdout.flush()
        # TODO: This forcibly terminates the entire process.
        #       This is not graceful. Revisit this later.
        os._exit(0)

    async def _handle_message(self) -> None:
        """Handle a message from an agent."""
        while True:
            msg = await self._wcomm.recv()

            match msg.type:
                case MessageType.CONFIG:
                    await self._handle_config(msg.content)

                case MessageType.FORCE_TERMINATE:
                    self._terminate_worker()

                case MessageType.TERMINATE:
                    await self._router.wait_on_term_ready()
                    self._terminate_worker()

                case MessageType.CHECK_LOOP:
                    failed_wids = msg.content
                    suspended_worlds = self._config_manager.get_suspended_worlds(
                        failed_wids
                    )
                    self._router.handle_suspended_worlds(suspended_worlds)

                    # if failed wids is empty, the job is recovered
                    # and we can reset inflight requests and tx event
                    if len(failed_wids) == 0:
                        self._reset_inflight_and_tx_event()

                case MessageType.FINISH_JOB:
                    # TODO: do the clean-up before transitioning to DONE
                    status = WorkerStatus.DONE
                    resp = Message(MessageType.STATUS, status, self._job_id)
                    self._wcomm.send(resp)

                    sys.stdout.flush()
                    # TODO: This forcibly terminates the entire process.
                    #       This is not graceful. Revisit this later.
                    os._exit(0)

    async def _handle_config(self, spec: ServeConfig) -> None:
        """Handle a config."""
        if spec is None:
            return

        await self._config_manager.handle_new_spec(spec)

        self._configure_variables(spec)

        self._initialize_once(spec)

        # run configure as a separate task since we need to unblock receiving
        # a new config to be processed when current configuration is finished
        _ = asyncio.create_task(self._configure())

    def _configure_variables(self, spec: ServeConfig) -> None:
        """Set variables that need to be updated."""
        self.max_inflight = spec.max_inflight

    def _initialize_once(self, spec: ServeConfig) -> None:
        if self._initialized:
            return

        # specify batch size once
        self._micro_batch_size = spec.micro_batch_size
        self._mc.set_batch_size(self._micro_batch_size)

        self._init_assets(spec)
        self._prepare_worker(spec)

        self._initialized = True

    def _init_assets(self, spec: ServeConfig) -> None:
        # load model meta info from zoo
        mmd = Zoo.get_model_metadata(spec.model)
        (path, name, split) = (
            spec.dataset.path,
            spec.dataset.name,
            spec.dataset.split,
        )

        # load dataset
        self.dataset = HuggingFaceDataset(
            mmd,
            path,
            dataset_name=name,
            split=split,
            micro_batch_size=self._micro_batch_size,
        )
        self._device = torch.device(spec.device)

        # load model intermediate representation
        self.modelir = ModelIR(mmd)

    def _prepare_worker(self, spec: ServeConfig) -> None:
        if spec.is_server:
            self._predict_fn = self.modelir.predict_fn
        else:
            self._initialize_worker(self.modelir, spec)

    async def run(self) -> None:
        """Run pipeline."""
        _ = asyncio.create_task(self._collect_metrics())
        _ = asyncio.create_task(self._handle_message())

        await self._initial_cfg_event.wait()

        try:
            if self._config_manager.is_server():
                await self._run_server()
            else:
                await self._run_worker()
        except Exception as e:
            kill_worker(e)
