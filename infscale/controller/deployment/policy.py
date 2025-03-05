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

from infscale import get_logger
from infscale.config import JobConfig, WorkerData
from infscale.controller.agent_context import AgentResources, DeviceType
from infscale.controller.job_context import AgentMetaData


class DeploymentPolicyEnum(Enum):
    """Deployment policy enum.

    STATIC: use job config as is assuming that the config has all the info for
            deployment
    """

    EVEN = "even"
    RANDOM = "random"
    STATIC = "static"


class DeploymentPolicy(ABC):
    """Abstract class for deployment policy."""

    def __init__(self):
        global logger
        logger = get_logger()

    @abstractmethod
    def split(
        self,
        dev_type: DeviceType,
        agent_data: list[AgentMetaData],
        agent_resources: dict[str, AgentResources],
        job_config: JobConfig,
    ) -> tuple[dict[str, JobConfig], dict[str, set[tuple[str, str]]]]:
        """
        Split the job config using a deployment policy
        and return updated job config and worker distribution for each agent.
        """
        pass

    def get_workers(
        self, distribution: dict[str, set[str]], workers: list[WorkerData]
    ) -> list[WorkerData]:
        """Return a list of workers."""
        # flat worker ids from each agent
        curr_worker_ids = {
            wid for wrk_data in distribution.values() for wid, _ in wrk_data
        }

        # get new worker ids
        new_workers = [worker for worker in workers if worker.id not in curr_worker_ids]

        return new_workers

    def get_curr_distribution(
        self, agent_data: list[AgentMetaData]
    ) -> dict[str, set[tuple[str, str]]]:
        """Return current distribution for each agent."""
        results = {}

        for data in agent_data:
            if len(data.existing_workers):
                results[data.id] = data.existing_workers

        return results

    def update_agents_distr(
        self, distribution: dict[str, set[tuple[str, str]]], workers: list[WorkerData]
    ) -> None:
        """Check if worker distribution has changed and update if needed."""
        # new worker ids
        worker_ids = {worker.id for worker in workers}

        # flatten the current worker set
        current_workers = {wid for wrk in distribution.values() for wid, _ in wrk}

        # compute removed workers
        removed_workers = set(current_workers) - set(worker_ids)

        # remove workers from the distribution
        for agent_id, workers in distribution.items():
            distribution[agent_id] = {
                (wid, device) for wid, device in workers if wid not in removed_workers
            }

    def _get_agent_updated_cfg(
        self, wrk_distr: dict[str, set[tuple[str, str]]], job_config: JobConfig
    ) -> dict[str, JobConfig]:
        """Return updated job config for each agent."""
        logger.info(f"got new worker distribution for agents: {wrk_distr}")

        agents_config = {}
        for agent_id, wrk in wrk_distr.items():
            # create a job_config copy to update and pass it to the agent.
            cfg = copy.deepcopy(job_config)

            for w in cfg.workers:
                # set the deploy flag if the worker is in worker distribution for this agent
                w.deploy = w.id in {wrk_id for wrk_id, _ in wrk}

            agents_config[agent_id] = cfg

        return agents_config
