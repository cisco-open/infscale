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

"""Class to keep world information."""
from dataclasses import dataclass

from infscale.execution.control import Channel


@dataclass(frozen=True)
class WorldInfo:
    """Information about World.

    Currently we only consider a world with one leader and one worker.
    Since there are two processes in each world, rank is either 0 or 1.
    """

    name: str  # world's name
    size: int  # size of the world
    addr: str  # IP address or hostname
    port: int  # port number for CCL communication
    backend: str  # backend
    channel: Channel  # control channel

    my_id: str  # my id
    me: int  # my rank

    other_id: str  # other id
    other: int  # other peer's rank

    recover: bool
    conflict_count: int
    multiworld_name: str
