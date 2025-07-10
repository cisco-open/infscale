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

"""base.py."""

import asyncio
from abc import abstractmethod

from infscale.execution.world import WorldInfo


class BaseForwarder:
    """Base forwarder class."""

    def __init__(self):
        """Initialize an instance."""
        pass

    @abstractmethod
    def select(
        self, tx_qs: list[tuple[WorldInfo, asyncio.Queue]]
    ) -> tuple[WorldInfo, asyncio.Queue]:
        """Select a tx queue from a given tx queue list."""
        pass
