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

"""config.py."""

from dataclasses import dataclass
from enum import Enum


class ReqGenEnum(str, Enum):
    """Request generation enum."""

    DEFAULT = "default"
    EXP = "exponential"


@dataclass
class DefaultParams:
    """Config class for default generator."""

    # variable to decide loading all dataset into memory
    # TODO: currently no-op; implement it
    in_memory: bool = False


@dataclass
class ExponentialParams(DefaultParams):
    """Exponential distribution."""

    rate: float = 1.0  # rate is per-second


GenParams = DefaultParams | ExponentialParams


@dataclass
class GenConfig:
    """Configuration class for request generation."""

    sort: str
    params: GenParams | None = None

    def __post_init__(self):
        """Conduct post-init task."""
        try:
            self.sort = ReqGenEnum(self.sort)
        except KeyError:
            raise ValueError(f"unknown request generator type: {self.sort}")

        match self.sort:
            case ReqGenEnum.DEFAULT:
                if self.params is None:
                    self.params = DefaultParams()
                else:
                    self.params = DefaultParams(**self.params)

            case ReqGenEnum.EXP:
                self.params = ExponentialParams(**self.params)
