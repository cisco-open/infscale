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
        agent_info: list[AgentMetaData],
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

        # get a list of agent info
        used_agent_data = [
            agent for agent in agent_info if agent.id in agent_device_map
        ]

        # dictionary to hold the workers for each agent_id
        distribution = self.get_curr_distribution(used_agent_data)

        workers = self.get_workers(distribution, job_config.workers)

        # check if the distribution has changed
        self.update_agents_distr(distribution, job_config.workers)

        self._update_distribution(
            distribution,
            workers,
            used_agent_data,
            agent_device_map,
            len(workers),
        )

        return self._get_agent_updated_cfg(distribution, job_config), distribution

    def _update_distribution(
        self,
        distribution: dict[str, set[tuple[str, str]]],
        workers: list[WorkerData],
        agent_data: list[AgentMetaData],
        agent_device_map: dict[str, AgentDeviceMap],
        remaining_workers: int,
        curr_idx: int = 0,
        dev_type: str = "gpu",
    ):
        # if there aren't any workers to split, exit recursion
        if remaining_workers <= 0:
            return

        # calculate next workers batch index
        idx = curr_idx

        for data in agent_data:
            # if the current agent doesn't have any resources, move to the next one
            if distribution.keys() and not self._has_resources(
                data.id, distribution, agent_device_map
            ):
                continue

            # assign worker id and device to the current agent
            for worker in workers[idx : idx + 1]:
                device = self._get_device(dev_type, data.id, agent_device_map)

                if data.id in distribution:
                    distribution[data.id].add(
                        (
                            worker.id,
                            device,
                        )
                    )
                else:
                    distribution[data.id] = {(worker.id, device)}

            # calculate remaining workers
            remaining_workers -= 1

            # move the start index to the next batch of workers
            idx += 1

        # recurring call of this method to assign remaining workers
        self._update_distribution(
            distribution,
            workers,
            agent_data,
            agent_device_map,
            remaining_workers,
            idx,
        )
