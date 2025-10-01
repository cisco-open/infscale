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

"""config_runner.py."""

import asyncio
from typing import Awaitable, Callable

from infscale.configs.job import ServeConfig


class ConfigRunner:
    """ConfigRunner class."""
    def __init__(self):
        """Initialize config runner instance."""
        self._loop = asyncio.get_event_loop()
        self._task: asyncio.Task | None = None
        self._event = asyncio.Event()
        self._spec: ServeConfig = None
        self._event.set() # initially no configure running
        self._curr_worlds_to_configure: set[str] = set()
        self._cancel_cur_cfg = False
        
    def handle_new_spec(self, spec: ServeConfig) -> None:
        """Handle new spec."""
        self._cancel_cur_cfg = self._should_cancel_current(spec)
        self._spec = spec
        
    def _should_cancel_current(self, spec: ServeConfig) -> bool:
        """Decide if current configuration should be cancelled."""
        if self._spec is None:
            return False

        new_worlds_to_configure = ServeConfig.get_worlds_to_configure(
            self._spec, spec
        )

        # cancel if the new config affects worlds currently being configured
        # TODO: if there's a overlap between new worlds and curr worlds we cancel
        # current configuration. This needs to be fixed, to cancel only the worlds that
        # are affected (eg new_worlds & curr_worlds)
        return not new_worlds_to_configure.isdisjoint(self._curr_worlds_to_configure)
        
    def set_worlds_to_configure(self, world_names: set[str]) -> None:
        """Set the world names currently being configured."""
        self._curr_worlds_to_configure = world_names

    async def schedule(self, coro_factory: Callable[[], Awaitable[None]]):
        """Cancel any in-progress configure and schedule a new one."""
        # wait for current to finish if we do not want to cancel
        if not self._cancel_cur_cfg:
            await self._event.wait()

        # cancel current if running
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # block again for new run
        self._event.clear()
        self._task = self._loop.create_task(self._run(coro_factory))

    async def _run(self, coro_factory: Callable[[], Awaitable[None]]):
        """Run coroutine factory."""
        try:
            await coro_factory()
        except asyncio.CancelledError:
            pass
        finally:
            # reset class attributes and events
            self._event.set()
            self._curr_worlds_to_configure = set()
