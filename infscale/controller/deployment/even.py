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

from infscale.config import JobConfig, WorkerData
from infscale.controller.deployment.policy import DeploymentPolicy
from infscale.controller.job_context import AgentDeviceMap, AgentMetaData


class EvenDeploymentPolicy(DeploymentPolicy):
    """Even deployment policy class."""

    def __init__(self):
        super().__init__()

    def split(
        self,
        agent_data: list[AgentMetaData],
        job_config: JobConfig,
        agent_device_map: dict[str, AgentDeviceMap],
    ) -> tuple[dict[str, JobConfig], dict[str, set[tuple[str, str]]]]:
        """
        Split the job config using even deployment policy
        and update config and worker distribution for each agent.

        Workers are distributed as evenly as possible across the available agents.
        If the number of workers isn't perfectly divisible by the number of agents,
        the "extra" workers are assigned to the first agents in the list.

        Return updated config and worker distribution for each agent
        """
        used_agent_data = [
            agent for agent in agent_data if agent.id in agent_device_map
        ]
        dev_type = "gpu"  # TODO: replace with config.dev_type

        # dictionary to hold the workers for each agent_id
        distribution = self.get_curr_distribution(used_agent_data)

        workers = self.get_workers(distribution, job_config.workers)

        # check if the distribution has changed
        self.check_agents_distr(distribution, job_config.workers)

        # find the minimum number of workers for each agent,
        # based on the agent with the lowest dev_type resources
        workers_per_agent = min(
            len(agent.gpu) if dev_type == "gpu" else agent.cpu
            for agent in agent_device_map.values()
        )

        self._update_distribution(
            distribution,
            workers,
            used_agent_data,
            agent_device_map,
            workers_per_agent,
            len(workers),
            dev_type,
        )

        return self._get_agent_updated_cfg(distribution, job_config), distribution

    def _update_distribution(
        self,
        distribution: dict[str, set[tuple[str, str]]],
        workers: list[WorkerData],
        agent_data: list[AgentMetaData],
        agent_device_map: dict[str, AgentDeviceMap],
        calculated_wrk_per_agent: int,
        remaining_workers: int,
        dev_type: str,
        curr_idx: int = 0,
    ):
        # if there aren't any workers to split, exit recursion
        if remaining_workers <= 0:
            return

        # first iteration will set a batch of workers to each agent
        # after that, each agent will get workers one by one until all workers are set
        workers_per_agent = calculated_wrk_per_agent if curr_idx == 0 else 1

        # calculate next workers batch index
        idx = curr_idx

        for data in agent_data:
            # if the current agent doesn't have any resources, move to the next one
            if not self._has_resources(data.id, distribution, agent_device_map):
                continue

            # assign worker id and device to the current agent
            for worker in workers[idx : idx + workers_per_agent]:
                device = self._get_device(dev_type, data.id, agent_device_map)

                distribution[data.id].add(
                    (
                        worker.id,
                        device,
                    )
                )
            # calculate remaining workers
            remaining_workers -= workers_per_agent

            # move the start index to the next batch of workers
            idx += workers_per_agent

        # recurring call of this method to assign remaining workers
        self._update_distribution(
            distribution,
            workers,
            agent_data,
            agent_device_map,
            workers_per_agent,
            remaining_workers,
            dev_type,
            idx,
        )

    def _has_resources(
        self,
        agent_id: str,
        distribution: dict[str, set[tuple[str, str]]],
        agent_device_map: dict[str, AgentDeviceMap],
    ):
        assigned_workers = len(distribution[agent_id])
        resources = agent_device_map[agent_id]
        available_resources = resources.cpu + len(resources.gpu)

        return assigned_workers < available_resources
