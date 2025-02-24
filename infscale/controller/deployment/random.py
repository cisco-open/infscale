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
from infscale.controller.deployment.policy import DeploymentPolicy
from infscale.controller.job_context import AgentDeviceMap, AgentMetaData


class RandomDeploymentPolicy(DeploymentPolicy):
    """Random deployment policy class."""

    def __init__(self):
        super().__init__()

    def split(
        self,
        agent_data: list[AgentMetaData],
        job_config: JobConfig,
        agent_device_map: dict[str, AgentDeviceMap],
    ) -> tuple[dict[str, JobConfig], dict[str, set[tuple[str, str]]]]:
        """
        Split the job config using random deployment policy
        and update config and worker distribution for each agent.

        Each agent gets at least one worker from the shuffled list.
        The remaining workers are distributed randomly.
        The random.shuffle(workers) ensures that the initial distribution
        of workers to agents is random.
        The random.choice(agent_ids) assigns the remaining workers in a random way,
        ensuring no agent is left out.

        Return updated config and worker distribution for each agent
        """

        # TODO: distribute based on agent_device_map

        # dictionary to hold the workers for each agent_id
        used_agent_data = [
            agent for agent in agent_data if agent.id in agent_device_map
        ]

        dev_type = "gpu"  # TODO: replace with config.dev_type

        distribution = self.get_curr_distribution(used_agent_data)

        workers = self.get_workers(distribution, job_config.workers)

        # check if the distribution has changed
        self.check_agents_distr(distribution, job_config.workers)

        # assign at least one worker / agent
        for data in agent_data:
            # we might not have workers if update job is made with less workers
            if len(workers):
                random.shuffle(workers)
                device = self._get_device(dev_type, data.id, agent_device_map)
                distribution[data.id].add((workers.pop().id, device))

        # distribute the remaining workers randomly
        while workers:
            data = random.choice(agent_data)  # choose an agent randomly
            device = self._get_device(dev_type, data.id, agent_device_map)
            distribution[data.id].add((workers.pop().id, device))

        return self._get_agent_updated_cfg(distribution, job_config), distribution
