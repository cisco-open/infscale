# Copyright 2024 Cisco Systems, Inc. and its affiliates
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

"""shortest.py."""

import asyncio
import sys

from infscale.execution.world import WorldInfo
from infscale.fwding.base import BaseForwarder


class ShortestForwarder(BaseForwarder):
    """Shortest queue length forwarder class."""

    def __init__(self):
        """Initialize an instance."""
        super().__init__()

    def select(
        self, tx_qs: list[tuple[WorldInfo, asyncio.Queue]]
    ) -> tuple[WorldInfo, asyncio.Queue]:
        """Select tx queue that has the shortest queue length."""
        world_info, tx_q = None, None
        qlen = sys.maxsize
        for wi, q in tx_qs:
            qsize = q.qsize()
            if qsize < qlen:
                world_info = wi
                tx_q = q

        return world_info, tx_q
