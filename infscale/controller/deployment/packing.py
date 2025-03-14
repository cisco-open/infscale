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
from infscale.controller.agent_context import AgentResources, DeviceType
from infscale.controller.deployment.policy import AssignmentData, DeploymentPolicy
from infscale.controller.job_context import AgentMetaData


class PackingPolicy(DeploymentPolicy):
    """Packing deployment policy class."""

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
        Split the job config using packing policy.

        Agent with most resources given dev_type is selected.
        Deploy as many workers as the resources allow.

        Return updated config and worker assignment map for each agent.
        """
        # dictionary to hold the workers for each agent_id
        assignment_map = self.get_curr_assignment_map(agent_data)

        workers = self.get_workers(assignment_map, job_config.workers)

        # check if the assignment map has changed
        self.update_agents_assignment_map(assignment_map, job_config.workers)

        while workers:
            agent_id, resources = self._select_agent_with_most_resources(
                dev_type, agent_resources
            )
            decided_device = None

            if job_config.auto_config:
                decided_device = resources.get_n_set_device(dev_type)

            worker = workers.pop()
            device = decided_device or worker.device

            if agent_id in assignment_map:
                assignment_map[agent_id].add(AssignmentData(worker.id, device))
            else:
                assignment_map[agent_id] = {AssignmentData(worker.id, device)}

        return self._get_agent_updated_cfg(assignment_map, job_config), assignment_map

    def _select_agent_with_most_resources(
        self, dev_type: DeviceType, agent_resources: dict[str, AgentResources]
    ) -> tuple[str, AgentResources]:
        """Return the agent_id and AgentResources instance with the most available resources based on dev_type."""

        if dev_type == DeviceType.GPU:
            # return resources with largest number of unused GPU
            return max(
                agent_resources.items(),
                key=lambda item: sum(not gpu.used for gpu in (item[1].gpu_stats or [])),
            )

        # return resources with biggest CPU efficiency score
        return max(
            agent_resources.items(),
            key=lambda item: (
                (100 - item[1].cpu_stats.load) * item[1].cpu_stats.total_cpus
            ),
        )
