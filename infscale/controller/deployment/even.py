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

from infscale.config import JobConfig
from infscale.controller.deployment.policy import DeploymentPolicy
from infscale.controller.job_context import AgentMetaData


class EvenDeploymentPolicy(DeploymentPolicy):
    """Even deployment policy class."""

    def __init__(self):
        super().__init__()

    def split(
        self, agent_data: list[AgentMetaData], job_config: JobConfig
    ) -> tuple[dict[str, JobConfig], dict[str, set[str]]]:
        """
        Split the job config using even deployment policy
        and update config and worker distribution for each agent.

        Workers are distributed as evenly as possible across the available agents.
        If the number of workers isn't perfectly divisible by the number of agents,
        the "extra" workers are assigned to the first agents in the list.

        Return updated config and worker distribution for each agent
        """
        # dictionary to hold the workers for each agent_id
        distribution = self.get_curr_distribution(agent_data)

        workers = self.get_workers(distribution, job_config.workers)

        # check if the distribution has changed
        self.check_agents_distr(distribution, job_config.workers)

        num_agents = len(agent_data)

        # assign workers to agents evenly by splitting the list of workers
        workers_per_agent = len(workers) // num_agents
        remaining_workers = len(workers) % num_agents

        start_index = 0
        for i, data in enumerate(agent_data):
            # for the first 'remaining_workers' agents, assign one extra worker
            num_workers_for_agent = workers_per_agent + (
                1 if i < remaining_workers else 0
            )

            # assign only worker id to the current agent
            distribution[data.id].update(
                worker.id
                for worker in workers[start_index : start_index + num_workers_for_agent]
            )

            # move the start index to the next batch of workers
            start_index += num_workers_for_agent

        return self._get_agent_updated_cfg(distribution, job_config), distribution
