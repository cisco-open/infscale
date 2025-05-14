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

import random

from infscale.configs.job import JobConfig
from infscale.controller.agent_context import AgentResources, DeviceType
from infscale.controller.deployment.assignment import (
    AssignmentCollection,
    AssignmentData,
)
from infscale.controller.deployment.policy import DeploymentPolicy
from infscale.controller.job_context import AgentMetaData


class RandomDeploymentPolicy(DeploymentPolicy):
    """Random deployment policy class."""

    def __init__(self):
        """Initialize a random deployment policy."""
        super().__init__()

    def split(
        self,
        dev_type: DeviceType,
        agent_info: dict[str, AgentMetaData],
        agent_resources: dict[str, AgentResources],
        job_config: JobConfig,
    ) -> dict[str, AssignmentCollection]:
        """
        Assign workers to agents based on config and random deployment policy.

        Each agent gets at least one worker from the shuffled list.
        The remaining workers are distributed randomly.
        The random.shuffle(workers) ensures that the initial assignment
        of workers to agents is random.
        The random.choice(agent_ids) assigns the remaining workers in a random way,
        ensuring no agent is left out.
        """
        # dictionary to hold the workers for each agent_id
        assignment_map = self.get_curr_assignment_map(agent_info)

        workers = self.get_new_workers(assignment_map, job_config.workers)

        # check if the assignment map has changed
        self.update_agents_assignment_map(assignment_map, job_config)

        # distribute the remaining workers randomly
        while workers:
            agent_data = random.choice(
                list(agent_info.values())
            )  # choose an agent randomly
            resources = agent_resources[agent_data.id]

            device = resources.get_n_set_device(dev_type, job_config.job_id)

            # this means that current agent don't have enough resources,
            # so we have to move to the next agent before popping the worker
            if device is None:
                continue

            if agent_data.id not in assignment_map:
                assignment_map[agent_data.id] = AssignmentCollection()

            worker = workers.pop()
            worlds_map = self._get_worker_worlds_map(worker.id, job_config)
            self._update_backend(worlds_map, device)
            assignment_data = AssignmentData(worker.id, device, worlds_map)

            assignment_map[agent_data.id].add(assignment_data)

        return assignment_map
