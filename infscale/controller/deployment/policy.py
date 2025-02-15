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

import copy
from abc import ABC, abstractmethod
from enum import Enum

from infscale.config import JobConfig


class DeploymentPolicyEnum(Enum):
    """Deployment policy enum."""

    EVEN = "even"
    RANDOM = "random"


class DeploymentPolicy(ABC):
    """Abstract class for deployment policy."""

    @abstractmethod
    def split(self, job_config: JobConfig) -> dict[str, JobConfig]:
        """
        Split the job config using random deployment policy
        and return updated job config for each agent.
        """
        pass

    def _get_agent_updated_cfg(
        self, wrk_distr: dict[str, list[str]], job_config: JobConfig
    ) -> dict[str, JobConfig]:
        """Return updated job config for each agent."""
        agents_config = {}
        for agent_id, wrk_ids in wrk_distr.items():
            # create a job_config copy to update and pass it to the agent.
            cfg = copy.deepcopy(job_config)

            for w in cfg.workers:
                # set the deploy flag if the worker is in worker distribution for this agent
                w.deploy = w.id in wrk_ids

            agents_config[agent_id] = cfg

        return agents_config
