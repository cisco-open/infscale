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

from itertools import cycle
from typing import Iterator

from infscale.configs.job import JobConfig
from infscale.controller.agent_context import AgentResources, DeviceType
from infscale.controller.deployment.assignment import (
    AssignmentCollection,
    AssignmentData,
)
from infscale.controller.deployment.policy import DeploymentPolicy
from infscale.controller.job_context import AgentMetaData


class EvenDeploymentPolicy(DeploymentPolicy):
    """Even deployment policy class."""

    def __init__(self):
        """Initialize an EvenDeploymentPolicy instance."""
        super().__init__()

    def split(
        self,
        dev_type: DeviceType,
        agent_info: dict[str, AgentMetaData],
        agent_resources: dict[str, AgentResources],
        job_config: JobConfig,
    ) -> dict[str, AssignmentCollection]:
        """
        Assign workers to agents based on config and even deployment policy.

        Workers are distributed as evenly as possible across the available agents.
        If the number of workers isn't perfectly divisible by the number of agents,
        the "extra" workers are assigned to the first agents in the list.
        """
        # dictionary to hold the workers for each agent_id
        assignment_map = self.get_curr_assignment_map(agent_info)
        agent_data_list = list(agent_info.values())

        workers = self.get_new_workers(assignment_map, job_config.workers)

        # check if the assignment map has changed
        self.update_agents_assignment_map(assignment_map, job_config)

        # sort agent data ascending by number of assignments
        agent_data_list.sort(key=lambda agent: len(agent.assignment_coll))
        agent_cycle = cycle(agent_data_list)

        while workers:
            device, agent_data = self._get_device_n_agent_data(
                dev_type, agent_resources, agent_cycle, job_config.job_id
            )

            if agent_data.id not in assignment_map:
                assignment_map[agent_data.id] = AssignmentCollection()

            worker = workers.pop()
            worlds_map = self._get_worker_worlds_map(worker.id, job_config)
            self._update_backend(worlds_map, device)
            assignment_data = AssignmentData(worker.id, device, worlds_map)

            assignment_map[agent_data.id].add(assignment_data)

        return assignment_map

    def _get_device_n_agent_data(
        self,
        dev_type: DeviceType,
        agent_resources: dict[str, AgentResources],
        agent_cycle: Iterator[AgentMetaData],
        job_id: str,
    ) -> tuple[str, AgentMetaData]:
        """Get device and agent data."""
        while True:
            agent_data = next(agent_cycle)  # Get the next agent
            resources = agent_resources[agent_data.id]

            device = resources.get_n_set_device(dev_type)
            if device:
                return device, agent_data
