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

from infscale.config import JobConfig
from infscale.controller.deployment.policy import DeploymentPolicy
from infscale.controller.exceptions import InvalidConfig
from infscale.controller.job_context import AgentMetaData


class StaticDeploymentPolicy(DeploymentPolicy):
    """Static deployment policy class."""

    def __init__(self):
        """Initialize static deployment policy instance."""
        super().__init__()

    def split(
        self, agent_data: list[AgentMetaData], job_config: JobConfig
    ) -> tuple[dict[str, JobConfig], dict[str, set[str]]]:
        """Split the job config statically based on its details."""
        distribution = self.get_curr_distribution(agent_data)

        workers = self.get_workers(distribution, job_config.workers)

        self.update_agents_distr(distribution, job_config.workers)

        agent_ip_to_id = {}
        for data in agent_data:
            agent_ip_to_id[data.ip] = data.id

        handled_worker_ids = set()
        # check if the config is complete and deployable
        # and build distribution
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

            agent_id = agent_ip_to_id[ip]
            if agent_id in distribution:
                distribution[agent_id].add(worker_id)
            else:
                distribution[agent_id] = {worker_id}

            handled_worker_ids.add(worker_id)

        for worker in workers:
            if worker.id in handled_worker_ids:
                continue

            # we will not run into this exception as long as the flow graph has
            # an entry for new workers
            raise InvalidConfig(f"failed to assign {worker.id} to an agent")

        return self._get_agent_updated_cfg(distribution, job_config), distribution
