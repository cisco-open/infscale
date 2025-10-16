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

"""pipeline_inspector.py."""

from collections import defaultdict

from infscale.configs.job import ServeConfig


class PipelineInspector:
    """PipelineInspector class."""

    def configure(self, spec: ServeConfig) -> None:
        """Configure inspector data."""
        self.spec = spec
        self.fwd_graph = self._build_forward_graph(self.spec)

    def _build_forward_graph(self, spec: ServeConfig) -> dict[str, list[str]]:
        """Build directional graph where key is sender and value is a list of receivers."""
        fwd_graph = defaultdict(list)
        for wid, world_list in spec.flow_graph.items():
            for world_info in world_list:
                for peer in world_info.peers:
                    fwd_graph[peer].append(wid)

        return fwd_graph

    def _is_loop(
        self,
        graph: dict[str, list[str]],
        start: str,
        my_id: str,
        failed_wids: set[str],
        visited: set[str],
    ) -> bool:
        """Perform graph search from `start` to check loop back to `my_id`, avoiding `failed_wid`."""
        if start == my_id:
            return True
        if start in visited or start in failed_wids:
            return False

        visited.add(start)
        return any(
            self._is_loop(graph, neighbor, my_id, failed_wids, visited)
            for neighbor in graph.get(start, [])
        )

    def get_suspended_worlds(self, failed_wids: set[str] = None) -> set[str]:
        """Return the set of suspended world names based on failed worker id."""
        suspended_worlds = set()
        my_id = self.spec.stage.id

        for wid, world_list in self.spec.flow_graph.items():
            # skip direct worlds owned by the failed worker
            if wid in failed_wids:
                continue

            for world_info in world_list:
                if my_id not in world_info.peers:
                    continue  # worker doesn't send to this receiver

                # check if there's any path back to my id.
                visited = set()
                if not self._is_loop(self.fwd_graph, wid, my_id, failed_wids, visited):
                    suspended_worlds.add(world_info.name)

        return suspended_worlds
