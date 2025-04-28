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

"""assignment.py."""

from __future__ import annotations

from infscale.configs.job import WorldInfo


class AssignmentData:
    """AssignmentData class."""

    def __init__(self, wid: str, device: str, worlds_map: dict[str, WorldInfo]) -> None:
        """Initialize an instance."""
        self.wid = wid
        self.device = device
        self.worlds_map = worlds_map

    def __str__(self) -> str:
        """Return string represenation of the object."""
        rep = f"worker id: {self.wid}, device: {self.device}, worlds_map: {self.worlds_map}"
        return rep


class AssignmentCollection:
    """AssignmentCollection class."""

    def __init__(self) -> None:
        """Initialize an instance."""
        self._coll_map: dict[str, AssignmentData] = {}

        self._is_worker_cache_valid = True
        self._worker_ids: set[str] = set()

        self._is_device_cache_valid = True
        self._devices: set[str] = set()

    def __len__(self) -> int:
        """Return the length of assignment collection."""
        return len(self._coll_map)

    def __or__(self, other: AssignmentCollection) -> AssignmentCollection:
        """Return a new instsance after combining two collections."""
        if not isinstance(other, AssignmentCollection):
            raise TypeError(f"wrong type: {type(other).__name__}")

        coll = AssignmentCollection()
        coll._coll_map = self._coll_map | other._coll_map

        return coll

    def add(self, data: AssignmentData) -> None:
        """Add assignment data to the collection."""
        self._coll_map[data.wid] = data

        self._is_worker_cache_valid = False
        self._is_device_cache_valid = False

    def remove(self, wid: str) -> None:
        """Remove assignment data of a given worker id from the collection."""
        data = self._coll_map.pop(wid, None)
        if data is None:
            return

        self._is_worker_cache_valid = False
        self._is_device_cache_valid = False

    def worker_ids(self) -> set[str]:
        """Return worker ids in the collection."""
        if not self._is_worker_cache_valid:
            self._worker_ids = {k for k in self._coll_map.keys()}
            self._is_worker_cache_valid = True

        return self._worker_ids

    def devices(self) -> set[str]:
        """Return devices in the collection."""
        if not self._is_device_cache_valid:
            self._devices = {data.device for data in self._coll_map.values()}
            self._is_device_cache_valid = True

        return self._devices

    def get_assignment_data(self, wid: str) -> AssignmentData | None:
        """Return assignment data for a given worker id."""
        return self._coll_map.get(wid)

    def get_assignment_list_by_excluding(
        self, workers: set[str]
    ) -> list[AssignmentData]:
        """Return a list of assignment data by excluding given workers."""
        return [data for data in self._coll_map.values() if data.wid not in workers]
