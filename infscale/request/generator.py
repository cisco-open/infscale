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
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import Tensor

from infscale.module.dataset import HuggingFaceDataset
from infscale.request.config import GenParams, ReqGenEnum


class Generator(ABC):
    """Abstact Generator class."""

    def initialize(self, dataset: HuggingFaceDataset, params: GenParams) -> None:
        """Initialize a generator."""
        self._dataset = dataset
        self._params = params

    @abstractmethod
    async def get(self) -> list[Tensor | None]:
        """Return generated requests as batch."""
        pass


class DefaultGenerator(Generator):
    """DefaultGenerator class."""

    async def get(self) -> list[Tensor | None]:
        """Return one batch of requests as a list.

        initialize() method must be called once before calling this method.
        """
        return [self._dataset.next_batch()]


class ExponentialGenerator(Generator):
    """ExponentialGenerator class."""

    def initialize(self, dataset: HuggingFaceDataset, params: GenParams) -> None:
        """Initialize the generator with exponential distribution."""
        # For exponential generator, params can't be None
        assert params is not None

        super().initialize(dataset, params)

        self._queue = asyncio.Queue()
        self._gen_evt = asyncio.Event()
        _ = asyncio.create_task(self._generate())

    async def _generate(self) -> None:
        # wait for a generation event
        # we'd like to generate data once get() is called
        await self._gen_evt.wait()

        while True:
            batch = self._dataset.next_batch()
            await self._queue.put(batch)

            if batch is None:
                break

            iat = np.random.exponential(scale=1 / self._params.rate)
            await asyncio.sleep(iat)

    async def get(self) -> list[Tensor | None]:
        """Return one batch of requests.

        initialize() method must be called once before calling this method.
        """
        self._gen_evt.set()

        batches = []
        while True:
            # this guarantees at least one batch of requests is returned
            batch = await self._queue.get()
            batches.append(batch)

            if self._queue.empty():
                break

        return batches


class GeneratorFactory:
    """Request generator factory class."""

    @staticmethod
    def get(sort: ReqGenEnum) -> Generator:
        """Return request generator instance of a chosen type."""
        generators = {
            ReqGenEnum.DEFAULT: DefaultGenerator(),
            ReqGenEnum.EXP: ExponentialGenerator(),
        }

        return generators[sort]
