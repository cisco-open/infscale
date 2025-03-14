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

from infscale.config import JobConfig
from infscale.controller.agent_context import AgentResources, DeviceType
from infscale.controller.deployment.policy import AssignmentData, DeploymentPolicy
from infscale.controller.job_context import AgentMetaData


class RandomDeploymentPolicy(DeploymentPolicy):
    """Random deployment policy class."""

    def __init__(self):
        super().__init__()

    def split(
        self,
        dev_type: DeviceType,
        agent_data: list[AgentMetaData],
        agent_resources: dict[str, AgentResources],
        job_config: JobConfig,
    ) -> tuple[dict[str, JobConfig], dict[str, set[AssignmentData]]]:
        """
        Split the job config using random deployment policy
        and update config and worker assignment map for each agent.

        Each agent gets at least one worker from the shuffled list.
        The remaining workers are distributed randomly.
        The random.shuffle(workers) ensures that the initial assignment
        of workers to agents is random.
        The random.choice(agent_ids) assigns the remaining workers in a random way,
        ensuring no agent is left out.

        Return updated config and worker assignment map for each agent
        """

        # dictionary to hold the workers for each agent_id
        assignment_map = self.get_curr_assignment_map(agent_data)

        workers = self.get_workers(assignment_map, job_config.workers)

        # check if the assignment map has changed
        self.update_agents_assignment_map(assignment_map, job_config.workers)

        # distribute the remaining workers randomly
        while workers:
            data = random.choice(agent_data)  # choose an agent randomly
            resources = agent_resources[data.id]

            decided_device = None

            if job_config.auto_config:
                decided_device = resources.get_n_set_device(dev_type)

            # this means that auto config is True but current agent
            # don't have enough resources, so we have to move to the
            # next agent before popping the worker
            if decided_device is None and job_config.auto_config:
                continue

            worker = workers.pop()
            device = decided_device or worker.device

            if data.id in assignment_map:
                assignment_map[data.id].add(AssignmentData(worker.id, device))
            else:
                assignment_map[data.id] = {AssignmentData(worker.id, device)}

        return self._get_agent_updated_cfg(assignment_map, job_config), assignment_map
