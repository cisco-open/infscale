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

"""policy.py"""

import copy
from abc import ABC, abstractmethod
from enum import Enum

from infscale import get_logger
from infscale.config import JobConfig, WorkerData, WorldInfo
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
    PACKING = "packing"


class AssignmentData:
    """AssignmentData class."""

    def __init__(self, wid: str, device: str, worlds_map: dict[str, WorldInfo]):
        self.wid = wid
        self.device = device
        self.worlds_map = worlds_map


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
    ) -> tuple[dict[str, JobConfig], dict[str, set[AssignmentData]]]:
        """
        Split the job config using a deployment policy
        and return updated job config and worker assignment map for each agent.
        """
        pass

    def get_workers(
        self, assignment_map: dict[str, set[AssignmentData]], workers: list[WorkerData]
    ) -> list[WorkerData]:
        """Return a list of new workers."""
        # flat worker ids from each agent
        curr_worker_ids = {
            data.wid
            for assignment_set in assignment_map.values()
            for data in assignment_set
        }

        # get new worker ids
        new_workers = [worker for worker in workers if worker.id not in curr_worker_ids]

        return new_workers

    def get_curr_assignment_map(
        self, agent_data: list[AgentMetaData]
    ) -> dict[str, set[AssignmentData]]:
        """Return current assignment map for each agent."""
        results = {}

        for data in agent_data:
            if len(data.assignment_set):
                results[data.id] = data.assignment_set

        return results

    def update_agents_assignment_map(
        self, assignment_map: dict[str, set[AssignmentData]], workers: list[WorkerData]
    ) -> None:
        """Check if worker assignment map has changed and update if needed."""
        # new worker ids
        worker_ids = {worker.id for worker in workers}

        # flatten the current worker set
        current_workers = {
            data.wid
            for assignment_set in assignment_map.values()
            for data in assignment_set
        }

        # compute removed workers
        removed_workers = set(current_workers) - set(worker_ids)

        # remove workers from the assignment map
        for agent_id, assignment_set in assignment_map.items():
            assignment_map[agent_id] = {
                AssignmentData(data.wid, data.device)
                for data in assignment_set
                if data.wid not in removed_workers
            }

    def _get_agent_updated_cfg(
        self, assignment_map: dict[str, set[AssignmentData]], job_config: JobConfig
    ) -> dict[str, JobConfig]:
        """Return updated job config for each agent."""
        logger.info(f"got new worker assignment map for agents: {assignment_map}")

        agents_config = {}
        for agent_id, assignment_set in assignment_map.items():
            # create a job_config copy to update and pass it to the agent.
            cfg = copy.deepcopy(job_config)

            for w in cfg.workers:
                # set the deploy flag if the worker is in worker assignment for this agent
                w.deploy = w.id in {data.wid for data in assignment_set}

            agents_config[agent_id] = cfg

        return agents_config

    def _get_worker_worlds_map(
        self, worker_id: str, config: JobConfig
    ) -> dict[str, WorldInfo]:
        """Return world info map for worker."""
        result = {
            world_info.name: world_info for world_info in config.flow_graph[worker_id]
        }

        return result

    def _update_backend(
        self, worlds_map: dict[str, WorldInfo], device: str, auto_config: bool
    ) -> dict[str, WorldInfo]:
        """Update backend value based on device if auto config is True."""
        if not auto_config:
            return

        for world in worlds_map.values():
            world.backend = "gloo" if device == "cpu" else "nccl"
