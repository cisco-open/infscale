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

from abc import ABC, abstractmethod
from enum import Enum

import torch
from torch import Tensor

from infscale.module.dataset import HuggingFaceDataset


class ReqGenEnum(Enum):
    """Request generation enum."""

    DEFAULT = "default"


class Generator(ABC):
    """Abstact Generator class."""

    def initialize(self, device: torch.device, dataset: HuggingFaceDataset) -> None:
        """Initialize a generator."""
        self._initialized = True

        self.dataset = dataset
        self.device = device

    @abstractmethod
    def get(self) -> Tensor | None:
        """Return generated requests as batch."""
        pass


class DefaultGenerator(Generator):
    """DefaultGenerator class."""

    def get(self) -> Tensor | None:
        """Return one batch of requests.

        initialize() method must be called once before calling this method.
        """
        return self.dataset.next_batch(self.device)


class GeneratorFactory:
    """Request generator factory class."""

    @staticmethod
    def get(sort: ReqGenEnum) -> Generator:
        """Return request generator instance of a chosen type."""
        generators = {
            ReqGenEnum.DEFAULT: DefaultGenerator(),
        }

        return generators[sort]
