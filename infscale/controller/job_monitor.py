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

"""job_monitor.py."""

from collections import defaultdict
from infscale import get_logger
from infscale.common.job_msg import WorkerStatus
from infscale.configs.job import JobConfig


logger = None


class JobMonitor:
    """JobMonitor class."""

    def __init__(
        self,
        worker_status: dict[str, WorkerStatus],
    ):
        """Initialize an instance.

        Attributes:
            worker_status (dict[str, WorkerStatus]): worker status dict
        """
        global logger
        logger = get_logger()
        self.worker_status = worker_status
        self.job_config: JobConfig
        self.server_ids: set[str] = set()

    def init_config(self, job_config: JobConfig) -> None:
        """Initialize data based on job config."""
        self.job_config = job_config

        for worker in self.job_config.workers:
            if worker.is_server:
                self.server_ids.add(worker.id)

    def is_job_failed(self) -> bool:
        """Decide wether the job is failed or not."""
        # if all server workers are FAILED
        if all(
            self.worker_status[sid] == WorkerStatus.FAILED for sid in self.server_ids
        ):
            logger.error("job failed due to all workers failed")
            return True

        # build directed graph from flow_graph
        graph = defaultdict(list)
        for src, nodes in self.job_config.flow_graph.items():
            for node in nodes:
                graph[src].extend(node.peers)

        # check if any server workers has a valid loop
        for sid in self.server_ids:
            if self.worker_status[sid] != WorkerStatus.FAILED and self._dfs(
                graph, sid, sid, set(), []
            ):
                return False

        logger.error("job failed due worker failure and incomplete data loop")
        return True

    def _dfs(
        self,
        graph: defaultdict[list],
        start: str,
        node: str,
        visited: set[str],
        path: list,
    ) -> bool:
        """Depth-First search for a cycle.

        Returns to start and avoids FAILED nodes.
        """
        if node == start and path:
            return True  # found a valid cycle
        if node in visited or self.worker_status.get(node) == WorkerStatus.FAILED:
            return False

        visited.add(node)

        for neighbor in graph.get(node, []):
            if self._dfs(graph, start, neighbor, visited, path + [neighbor]):
                return True  # found a valid cycle

        visited.remove(node)

        return False
