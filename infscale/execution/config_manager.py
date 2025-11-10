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

"""config_manager.py."""

import asyncio
from typing import Awaitable, Callable

from infscale.configs.job import ServeConfig
from infscale.execution.control import Channel as CtrlCh
from infscale.execution.world import WorldInfo


class ConfigManager:
    """ConfigManager class."""

    def __init__(self):
        """Initialize config manager instance."""
        self._loop = asyncio.get_event_loop()
        self._task: asyncio.Task | None = None
        self._event = asyncio.Event()
        self._spec: ServeConfig = None
        self._event.set()
        self._curr_worlds_to_configure: set[str] = set()
        self._cancel_cur_cfg = False
        self._world_infos: dict[str, WorldInfo] = {}

    def handle_new_spec(self, spec: ServeConfig) -> None:
        """Handle new spec."""
        self._cancel_cur_cfg = self._should_cancel_current(spec)
        self._spec = spec

    def _should_cancel_current(self, spec: ServeConfig) -> bool:
        """Decide if current configuration should be cancelled."""
        if self._spec is None:
            return False

        new_worlds_to_configure = ServeConfig.get_worlds_to_configure(self._spec, spec)

        # cancel if the new config affects worlds currently being configured
        # TODO: if there's a overlap between new worlds and curr worlds we cancel
        # current configuration. This needs to be fixed, to cancel only the worlds that
        # are affected (eg new_worlds & curr_worlds)
        return not new_worlds_to_configure.isdisjoint(self._curr_worlds_to_configure)

    def set_worlds_to_configure(self, world_names: set[str]) -> None:
        """Set the world names currently being configured."""
        self._curr_worlds_to_configure = world_names

    def set_world_infos(self, worlds: list[WorldInfo]) -> None:
        """Set new world infos."""
        for world_info in worlds:
            self._world_infos[world_info.name] = world_info

    def get_world_infos(self) -> dict[str, WorldInfo]:
        "Get world infos."
        return self._world_infos

    def is_first_run(self) -> bool:
        "Return boolean if is first run or not."
        return not self._world_infos

    def remove_world_info(self, world_name: str) -> None:
        """Remove world info by name."""
        del self._world_infos[world_name]

    def get_worlds_to_add_and_remove(self) -> tuple[list[WorldInfo], list[WorldInfo]]:
        """Return a list of world infos to add and to remove."""
        new_world_infos = self._build_world_infos()

        new = new_world_infos.keys()
        cur = self._world_infos.keys()

        worlds_to_add = [new_world_infos[name] for name in new - cur]
        worlds_to_remove = [new_world_infos[name] for name in cur - new]

        return worlds_to_add, worlds_to_remove

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

    def _build_world_infos(self) -> dict[str, WorldInfo]:
        world_infos: dict[str, WorldInfo] = {}

        my_id = self._spec.stage.id
        for k, v in self._spec.flow_graph.items():
            for cfg_world_info in v:
                # NOTE: no. of peers is always 1 for now
                assert len(cfg_world_info.peers) == 1

                if my_id == k:
                    my_rank = 0
                    other_rank = 1
                    other_id = cfg_world_info.peers[0]
                elif my_id in cfg_world_info.peers:
                    # NOTE: this is always 1 for now
                    my_rank = cfg_world_info.peers.index(my_id) + 1
                    other_rank = 0
                    other_id = k
                else:
                    continue

                name, backend, addr, data_port, ctrl_port, recover, conflict_count = (
                    cfg_world_info.name,
                    cfg_world_info.backend,
                    cfg_world_info.addr,
                    cfg_world_info.data_port,
                    cfg_world_info.ctrl_port,
                    cfg_world_info.recover,
                    cfg_world_info.conflict_count,
                )

                world_size = len(cfg_world_info.peers) + 1
                ctrl_ch = CtrlCh(my_rank, world_size, addr, ctrl_port)

                data = {
                    "name": name,
                    "size": world_size,
                    "addr": addr,
                    "port": data_port,
                    "backend": backend,
                    "channel": ctrl_ch,
                    "my_id": my_id,
                    "me": my_rank,
                    "other_id": other_id,
                    "other": other_rank,
                    "recover": recover,
                    "conflict_count": conflict_count,
                    "multiworld_name": f"{name}-{conflict_count}",
                }
                world_info = WorldInfo(**data)
                world_infos[name] = world_info

        return world_infos

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
