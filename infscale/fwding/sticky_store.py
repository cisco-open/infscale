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

"""sticky_store.py."""

import asyncio

from infscale.execution.world import WorldInfo


class StickyStore:
    """StickyStore class."""

    def __init__(self):
        """Initialize an instance."""
        self._store: dict[int, [WorldInfo, asyncio.Queue]] = dict()

    def select(self, seqno: int) -> tuple[WorldInfo, asyncio.Queue] | None:
        """Return tx queue if there is match in sticky store."""
        if seqno not in self._store:
            return None

        return self._store[seqno]

    def update(self, seqno: int, world_info: WorldInfo, tx_q: asyncio.Queue) -> None:
        """Update world info and tx queue if sticky is true."""
        if seqno in self._store:
            return

        # TODO: Here we keep adding a new element for the requst of a seqno.
        # However, we have to remove it once is served. Implement the cleanup
        # functionality.
        self._store[seqno] = (world_info, tx_q)
