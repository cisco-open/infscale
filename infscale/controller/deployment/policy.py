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

"""policy.py."""

from abc import ABC, abstractmethod
from enum import Enum

from infscale import get_logger
from infscale.configs.job import JobConfig, WorkerData, WorldInfo
from infscale.controller.agent_context import AgentResources, DeviceType
from infscale.controller.deployment.assignment import (
    AssignmentCollection,
    AssignmentData,
)
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


DEFAULT_DEPLOYMENT_POLICY = DeploymentPolicyEnum.RANDOM.value


class DeploymentPolicy(ABC):
    """Abstract class for deployment policy."""

    def __init__(self):
        """Initialize an DeploymentPolicy policy."""
        global logger
        logger = get_logger()

    @abstractmethod
    def split(
        self,
        dev_type: DeviceType,
        agent_data_list: list[AgentMetaData],
        agent_resources: dict[str, AgentResources],
        job_config: JobConfig,
    ) -> dict[str, AssignmentCollection]:
        """Assign workers to agents based on config and deployment policy."""
        pass

    def get_new_workers(
        self, assignment_map: dict[str, AssignmentCollection], workers: list[WorkerData]
    ) -> list[WorkerData]:
        """Return a list of new workers."""
        curr_worker_ids = set()
        for coll in assignment_map.values():
            curr_worker_ids |= coll.worker_ids()

        # get new worker ids
        new_workers = [worker for worker in workers if worker.id not in curr_worker_ids]

        return new_workers

    def get_curr_assignment_map(
        self, agent_data_list: list[AgentMetaData]
    ) -> dict[str, AssignmentCollection]:
        """Return current assignment map for each agent."""
        results = {}

        for agent_data in agent_data_list:
            if len(agent_data.assignment_coll):
                results[agent_data.id] = agent_data.assignment_coll

        return results

    def update_agents_assignment_map(
        self, assignment_map: dict[str, AssignmentCollection], config: JobConfig
    ) -> None:
        """Check if worker assignment map has changed and update if needed."""
        # new worker ids
        new_worker_ids = {worker.id for worker in config.workers}

        curr_worker_ids = set()
        for coll in assignment_map.values():
            curr_worker_ids |= coll.worker_ids()

        # compute removed workers
        removed_workers = curr_worker_ids - new_worker_ids

        # update assignment map by creating new assignment collections
        for agent_id, assignment_coll in assignment_map.items():
            coll = AssignmentCollection()

            assignment_list = assignment_coll.get_assignment_list_by_excluding(
                removed_workers
            )
            for data in assignment_list:
                # update worlds map due to possible flow graph change
                worker_worlds_map = self._get_worker_worlds_map(data.wid, config)
                new_data = AssignmentData(data.wid, data.device, worker_worlds_map)
                coll.add(new_data)

            assignment_map[agent_id] = coll

    def _get_worker_worlds_map(
        self, worker_id: str, config: JobConfig
    ) -> dict[str, WorldInfo]:
        """Return world info map for worker."""
        result = {
            world_info.name: world_info for world_info in config.flow_graph[worker_id]
        }

        return result

    def _update_backend(
        self, worlds_map: dict[str, WorldInfo], device: str
    ) -> dict[str, WorldInfo]:
        """Update backend value based on device."""
        for world in worlds_map.values():
            world.backend = "gloo" if device == "cpu" else "nccl"

    def _set_rollback_data(
        self,
        resources: AgentResources,
        device: str,
        temp_res: dict[AgentResources, set[str]],
    ) -> None:
        """Set resources and devices for rollback."""
        if resources in temp_res:
            temp_res[resources].add(device)
        else:
            temp_res[resources] = {device}

    def _rollback_device_state(self, temp_res: dict[AgentResources, set[str]]) -> None:
        """Rollback device state in agent resources."""
        for res, devices in temp_res.items():
            for gpu_stat in res.gpu_stats:
                if f"cuda:{gpu_stat.id}" not in devices:
                    continue
                gpu_stat.used = False
