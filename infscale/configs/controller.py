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

"""controller.py."""

import os
from dataclasses import dataclass
from enum import Enum

from infscale.common.constants import (
    APISERVER_PORT,
    CONTROLLER_PORT,
)


class DeploymentPolicyEnum(Enum):
    """Deployment policy enum.

    STATIC: use job config as is assuming that the config has all the info for
            deployment
    """

    EVEN = "even"
    PACKING = "packing"
    RANDOM = "random"
    STATIC = "static"


DEFAULT_DEPLOYMENT_POLICY = DeploymentPolicyEnum.RANDOM.value


class ReqGenEnum(str, Enum):
    """Request generation enum."""

    DEFAULT = "default"
    EXP = "exponential"


@dataclass
class DefaultParams:
    """Config class for default generator."""

    # variable to decide loading all dataset into memory
    in_memory: bool = False
    # variable to decide number of dataset replays
    # 0: no replay; -1: infinite
    replay: int = 0


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


@dataclass
class CtrlConfig:
    """CtrlConfig class."""

    api_port: int = APISERVER_PORT
    ctrl_port: int = CONTROLLER_PORT
    deploy_policy: DeploymentPolicyEnum = DeploymentPolicyEnum.RANDOM
    autoscale: bool = False
    # a directory path to job deployment templates/plans
    job_plans: str = ""
    reqgen: GenConfig = GenConfig(sort=ReqGenEnum.DEFAULT.value)

    def __post_init__(self):
        """Populate controller's config with correct data types."""
        if self.autoscale:
            # it can't be empty if autoscaling is enabled
            self.job_plans = os.path.expanduser(self.job_plans)
            assert os.path.isdir(self.job_plans), f"invalid  folder: {self.job_plans}"
            self.deploy_policy = DeploymentPolicyEnum.STATIC.value
            print(f"Using {self.deploy_policy} policy since autoscale is enabled")

        if isinstance(self.deploy_policy, str):
            try:
                self.deploy_policy = DeploymentPolicyEnum(self.deploy_policy)
            except ValueError:
                chosen = DeploymentPolicyEnum.RANDOM
                msg = f"WARNING: {self.deploy_policy} is an invalid policy; using {chosen.value}"
                print(msg)
                self.deploy_policy = chosen

        if not isinstance(self.reqgen, GenConfig):
            self.reqgen = GenConfig(**self.reqgen)
