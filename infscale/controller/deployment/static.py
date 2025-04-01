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

"""Static deployment policy."""

from infscale.common.exceptions import InvalidConfig
from infscale.config import JobConfig, WorkerData
from infscale.controller.agent_context import AgentResources, DeviceType
from infscale.controller.deployment.policy import AssignmentData, DeploymentPolicy
from infscale.controller.job_context import AgentMetaData


class StaticDeploymentPolicy(DeploymentPolicy):
    """Static deployment policy class."""

    def __init__(self):
        """Initialize static deployment policy instance."""
        super().__init__()

    def split(
        self,
        dev_type: DeviceType,
        agent_data: list[AgentMetaData],
        agent_resources: dict[str, AgentResources],
        job_config: JobConfig,
    ) -> tuple[dict[str, JobConfig], dict[str, set[AssignmentData]]]:
        """Split the job config statically based on its details."""
        assignment_map = self.get_curr_assignment_map(agent_data)

        workers = self.get_workers(assignment_map, job_config.workers)

        self.update_agents_assignment_map(assignment_map, job_config)

        agent_ip_to_id = {}
        for data in agent_data:
            agent_ip_to_id[data.ip] = data.id

        handled_worker_ids = set()
        # check if the config is complete and deployable
        # and build assignment map
        for worker_id, world_infos in job_config.flow_graph.items():
            # create a set to remove duplicate
            ips = set(world_info.addr for world_info in world_infos)
            # convert the set to a list
            ips = list(ips)

            if len(ips) != 1:
                msg1 = f"worlds of worker {worker_id} can't have more than one IP;"
                msg2 = f" {len(ips)} IPs exist in the config"
                raise InvalidConfig(msg1 + msg2)

            ip = ips[0]
            if ip not in agent_ip_to_id:
                raise InvalidConfig(f"{ip} not a valid agent IP")

            worker_in_use = next(
                (data for data in agent_data if worker_id in data.wids_to_deploy), None
            )

            if worker_in_use:
                # worker already processed
                continue

            agent_id = agent_ip_to_id[ip]
            resources = agent_resources[agent_id]
            device = self._get_n_update_worker_device(
                worker_id, job_config.workers, resources
            )
            worlds_map = self._get_worker_worlds_map(worker_id, job_config)

            assignment_data = AssignmentData(worker_id, device, worlds_map)
            if agent_id in assignment_map:
                assignment_map[agent_id].add(assignment_data)
            else:
                assignment_map[agent_id] = {assignment_data}

            handled_worker_ids.add(worker_id)

        for worker in workers:
            if worker.id in handled_worker_ids:
                continue

            # we will not run into this exception as long as the flow graph has
            # an entry for new workers
            raise InvalidConfig(f"failed to assign {worker.id} to an agent")

        return self._get_agent_updated_cfg(assignment_map, job_config), assignment_map

    def _get_n_update_worker_device(
        self, worker_id: str, workers: list[WorkerData], resources: AgentResources
    ) -> str:
        """Get and update worker device."""
        worker = next(w for w in workers if w.id == worker_id)

        device = worker.device

        if device == "cpu":
            return device

        gpu_id = int(device.split(":")[1])

        gpu_stat = next(
            gpu_stat for gpu_stat in resources.gpu_stats if gpu_stat.id == gpu_id
        )

        if gpu_stat.used:
            raise InvalidConfig(
                f"GPU {gpu_stat.id} is used, please consider using another device."
            )

        gpu_stat.used = True

        return device
