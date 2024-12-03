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
import asyncio

from infscale.execution.world import WorldInfo

_curr_q_idx = 0


def select(tx_qs: list[tuple[WorldInfo, asyncio.Queue]]) -> (WorldInfo, asyncio.Queue):
    """Select tx queue in a round-robin fashion."""
    global _curr_q_idx

    try:
        world_info, tx_q = tx_qs[_curr_q_idx]
    except IndexError:
        # IndexError can happen due to online reconfiguration (e.g., worker leave)
        # In such a case, reset the current index and do selection again
        _curr_q_idx = 0
        world_info, tx_q = tx_qs[_curr_q_idx]

    _curr_q_idx = (_curr_q_idx + 1) % len(tx_qs)

    return world_info, tx_q
