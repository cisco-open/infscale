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

from infscale.configs.job import JobConfig, WorkerData
from infscale.controller.agent_context import AgentResources, DeviceType
from infscale.controller.deployment.assignment import (
    AssignmentCollection,
    AssignmentData,
)
from infscale.controller.deployment.policy import DeploymentPolicy
from infscale.controller.job_context import AgentMetaData


class PackingPolicy(DeploymentPolicy):
    """Packing deployment policy class."""

    def __init__(self):
        """Initialize a packing policy instance."""
        super().__init__()

    def split(
        self,
        dev_type: DeviceType,
        agent_data_list: list[AgentMetaData],
        agent_resources: dict[str, AgentResources],
        job_config: JobConfig,
    ) -> dict[str, AssignmentCollection]:
        """
        Assign workers to agents based on config and packing deployment policy.

        Agent with most resources given dev_type is selected.
        Deploy as many workers as the resources allow.
        """
        # dictionary to hold the workers for each agent_id
        assignment_map = self.get_curr_assignment_map(agent_data_list)

        workers = self.get_new_workers(assignment_map, job_config.workers)

        # check if the assignment map has changed
        self.update_agents_assignment_map(assignment_map, job_config)

        while workers:
            agent_id, resources = self._select_agent_with_most_resources(
                dev_type, agent_resources
            )

            self._assign_workers(
                dev_type, agent_id, job_config, workers, assignment_map, resources
            )

        return assignment_map

    def _assign_workers(
        self,
        dev_type: DeviceType,
        agent_id: str,
        job_config: JobConfig,
        workers: list[WorkerData],
        assignment_map: dict[str, AssignmentCollection],
        resources: AgentResources,
    ) -> None:
        workers_num = len(workers)
        workers_to_deploy = workers_num

        if dev_type == DeviceType.GPU:
            available_gpu_num = sum(
                1 for gpu_stat in resources.gpu_stats if not gpu_stat.used
            )
            workers_to_deploy = (
                workers_num if available_gpu_num >= workers_num else available_gpu_num
            )

        for _ in range(workers_to_deploy):
            device = resources.get_n_set_device(dev_type)
            self._assign_worker(
                workers,
                agent_id,
                job_config,
                device,
                assignment_map,
            )

    def _assign_worker(
        self,
        workers: list[WorkerData],
        agent_id: str,
        job_config: JobConfig,
        device: str,
        assignment_map: dict[str, AssignmentCollection],
    ) -> None:
        """Assign worker and update backend."""
        if agent_id not in assignment_map:
            assignment_map[agent_id] = AssignmentCollection()

        worker = workers.pop()
        worlds_map = self._get_worker_worlds_map(worker.id, job_config)
        self._update_backend(worlds_map, device)
        assignment_data = AssignmentData(worker.id, device, worlds_map)

        assignment_map[agent_id].add(assignment_data)

    def _select_agent_with_most_resources(
        self,
        dev_type: DeviceType,
        agent_resources: dict[str, AgentResources],
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
