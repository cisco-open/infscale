# Copyright 2025 Cisco Systems, Inc. and its affiliates
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

"""generator.py."""

import asyncio
import os
from abc import ABC, abstractmethod

import numpy as np
from torch import Tensor

from infscale import get_logger
from infscale.configs.controller import GenParams, RateScheduleItem, ReqGenEnum
from infscale.execution.metrics_collector import MetricsCollector
from infscale.module.dataset import HuggingFaceDataset


logger = None


class Generator(ABC):
    """Abstact Generator class."""

    def initialize(
        self,
        dataset: HuggingFaceDataset,
        params: GenParams,
        batch_size: int,
        mc: MetricsCollector,
    ) -> None:
        """Initialize a generator."""
        self._dataset = dataset
        self._params = params
        self._batch_size = batch_size
        self._mc = mc
        self._seqno = 0

        global logger
        logger = get_logger(f"{os.getpid()}")

    @abstractmethod
    async def get(self) -> list[Tensor | None]:
        """Return generated requests as batch."""
        pass


class DefaultGenerator(Generator):
    """DefaultGenerator class."""

    async def get(self) -> list[tuple[Tensor, bool]]:
        """Return one batch of requests as a list.

        initialize() method must be called once before calling this method.
        """
        self._mc.update(self._seqno)
        self._seqno += 1
        return [self._dataset.next_batch()]


class ExponentialGenerator(Generator):
    """ExponentialGenerator class."""

    def initialize(
        self,
        dataset: HuggingFaceDataset,
        params: GenParams,
        batch_size: int,
        mc: MetricsCollector,
    ) -> None:
        """Initialize the generator with exponential distribution."""
        # For exponential generator, params can't be None
        assert params is not None

        super().initialize(dataset, params, batch_size, mc)

        self._batch_rate = self._params.rate / self._batch_size

        self._queue = asyncio.Queue()
        self._gen_evt = asyncio.Event()
        _ = asyncio.create_task(self._generate())

    async def _generate(self) -> None:
        # wait for a generation event
        # we'd like to generate data once get() is called
        await self._gen_evt.wait()

        while True:
            batch, is_last = self._dataset.next_batch()
            await self._queue.put((batch, is_last))

            if is_last:
                break

            self._mc.update(self._seqno)
            self._seqno += 1

            iat = self._compute_iat()
            await asyncio.sleep(iat)

    def _compute_iat(self):
        return np.random.exponential(scale=1 / self._batch_rate)

    async def get(self) -> list[tuple[Tensor, bool]]:
        """Return one batch of requests.

        initialize() method must be called once before calling this method.
        """
        self._gen_evt.set()

        batches = []
        while True:
            # this guarantees at least one batch of requests is returned
            batch, is_last = await self._queue.get()
            batches.append((batch, is_last))

            if self._queue.empty():
                break

        return batches


class MultiRateExponentialGenerator(ExponentialGenerator):
    """Exponential generator with replay-dependent rate schedule."""

    def initialize(
        self,
        dataset,
        params,
        batch_size,
        mc,
    ) -> None:
        assert params is not None
        # intentionally bypassing super().initialize
        # for properly setting up queue and event and to avoid duplicating
        # asyncio task creation for _generate method
        Generator.initialize(self, dataset, params, batch_size, mc)

        self.range_list = self._prepare_schedule(
            self._params.rate, self._params.schedule, self._params.replay
        )

        self._range_index = 0
        rate = self.range_list[0][2]
        self._batch_rate = rate / self._batch_size

        self._queue = asyncio.Queue()
        self._gen_evt = asyncio.Event()
        _ = asyncio.create_task(self._generate())

        msg = f"generator initialized with rate={rate}"
        msg += f" replay rate update schedule={self._params.schedule}"
        logger.info(msg)

    def _prepare_schedule(
        self, base_rate: float, schedule: list[RateScheduleItem], max_replay: int
    ) -> list[tuple[int, int, float]]:
        """Convert replay-based schedule into continuous replay ranges."""
        schedule_sorted = sorted(schedule, key=lambda s: s.replay_index)

        rate_schedule_ranges = []
        prev_replay = 0
        prev_rate = base_rate

        for item in schedule_sorted:
            # range [prev_replay, item.replay_index - 1] uses prev_rate
            rate_schedule_ranges.append((prev_replay, item.replay_index - 1, prev_rate))
            prev_replay = item.replay_index
            prev_rate = item.rate

        # last range goes until max_replay
        rate_schedule_ranges.append((prev_replay, max_replay, prev_rate))
        return rate_schedule_ranges

    def _compute_iat(self):
        current_replay = self._params.replay - self._dataset._replay
        range_info = self.range_list[self._range_index]

        if not range_info[0] <= current_replay <= range_info[1]:
            self._range_index += 1

            range_info = self.range_list[self._range_index]
            rate = range_info[2]
            self._batch_rate = rate / self._batch_size

            logger.info(f"sending rate updated to {rate}")

        return np.random.exponential(scale=1 / self._batch_rate)


class GeneratorFactory:
    """Request generator factory class."""

    @staticmethod
    def get(sort: ReqGenEnum) -> Generator:
        """Return request generator instance of a chosen type."""
        generators = {
            ReqGenEnum.DEFAULT: DefaultGenerator(),
            ReqGenEnum.EXP: ExponentialGenerator(),
            ReqGenEnum.MULTIRATE_EXP: MultiRateExponentialGenerator(),
        }

        return generators[sort]
