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

"""job_checker.py."""

from collections import defaultdict

from infscale import get_logger
from infscale.common.job_msg import WorkerStatus
from infscale.configs.job import JobConfig


logger = None


class JobChecker:
    """JobChecker class."""

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
        self.server_ids: set[str] = set()
        self.graph: defaultdict[str, list[str]] = None

    def setup(self, job_config: JobConfig) -> None:
        """Setup checker data based on job config."""
        for worker in job_config.workers:
            if worker.is_server:
                self.server_ids.add(worker.id)

        # build directed graph from flow_graph
        self.graph = defaultdict(list)
        for src, nodes in job_config.flow_graph.items():
            for node in nodes:
                self.graph[src].extend(node.peers)

    def is_job_failed(self) -> bool:
        """Decide wether the job is failed or not."""
        # check if any server workers has a valid loop
        for sid in self.server_ids:
            if self.worker_status[sid] != WorkerStatus.FAILED and self._is_cycle(
                sid, sid, set(), True
            ):
                return False

        logger.error("job failed due to worker failure or incomplete data loop")
        return True

    def _is_cycle(
        self,
        start: str,
        node: str,
        visited: set[str],
        is_start: bool,
    ) -> bool:
        """Return true if there is a cycle in the flow graph."""
        if node == start and not is_start:
            return True  # found a valid cycle
        if node in visited or self.worker_status.get(node) == WorkerStatus.FAILED:
            return False

        visited.add(node)

        for neighbor in self.graph.get(node, []):
            if self._is_cycle(start, neighbor, visited, False):
                return True  # found a valid cycle

        visited.remove(node)

        return False
