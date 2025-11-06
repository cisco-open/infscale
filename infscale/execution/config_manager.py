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
        # semaphore event for back-to-back configs
        self._config_event = asyncio.Event()
        self._config_event.set()
        self._world_tasks: dict[str, asyncio.Task] = {}
        self._curr_spec: ServeConfig = None
        self._curr_world_infos: dict[str, WorldInfo] = {}
        self._new_world_infos: dict[str, WorldInfo] = {}
        self.worlds_to_cancel = set()

    async def handle_new_spec(self, spec: ServeConfig) -> None:
        """Handle new spec."""
        new_worlds_to_configure = ServeConfig.get_worlds_to_configure(
            self._curr_spec, spec
        )

        # on the first run, both new and cur will be empty sets
        new = self._new_world_infos.keys()
        cur = self._curr_world_infos.keys()
        curr_worlds_to_configure = new - cur

        self.worlds_to_cancel = new_worlds_to_configure & curr_worlds_to_configure

        if len(self.worlds_to_cancel):
            await self._cancel_world_configuration(self.worlds_to_cancel)

        # wait for current configuration to finish
        await self._config_event.wait()

        # executed after each configuration
        self._new_world_infos = self._build_world_infos(spec)
        self._curr_spec = spec
        self.worlds_to_cancel = set()

        # block handling new spec after doing cleanup for the current one
        self._config_event.clear()

    def unblock_next_config(self) -> None:
        """Set task event and unblock next config process."""
        self._config_event.set()

    def update_world_infos(self, worlds_names: set[str]) -> None:
        """Update world infos."""
        for world_name in worlds_names:
            world_info = self._new_world_infos[world_name]
            self._curr_world_infos[world_info.name] = world_info

    def get_curr_world_infos(self) -> dict[str, WorldInfo]:
        "Get current world infos."
        return self._curr_world_infos

    def is_first_run(self) -> bool:
        "Return boolean if is first run or not."
        return not self._curr_world_infos

    def remove_world_info(self, world_name: str) -> None:
        """Remove world info by name."""
        del self._curr_world_infos[world_name]

    def get_worlds_to_add_and_remove(self) -> tuple[set[str], set[str]]:
        """Return a list of world infos to add and to remove."""
        new = self._new_world_infos.keys()
        cur = self._curr_world_infos.keys()

        worlds_to_add = new - cur
        worlds_to_remove = cur - new

        return worlds_to_add, worlds_to_remove

    def get_new_world_info(self, world_name: str) -> dict[str, WorldInfo]:
        """Return new world info based on world name."""
        return self._new_world_infos[world_name]

    def get_worlds_to_add(self, world_names: set[str]) -> list[WorldInfo]:
        """Return a list of world infos to add."""
        return [self._new_world_infos[world_name] for world_name in world_names]

    def get_worlds_to_remove(self, world_names: set[str]) -> list[WorldInfo]:
        """Return a list of world infos to remove."""
        return [self._curr_world_infos[world_name] for world_name in world_names]

    async def _cancel_world_configuration(self, world_names: set[str]):
        """Cancel only worlds that are impacted by new spec."""
        coroutines = [self._cancel_world(w) for w in world_names]
        await asyncio.gather(*coroutines, return_exceptions=True)

    def schedule_world_cfg(
        self, world_info: WorldInfo, coro_factory: Callable[[], Awaitable[None]]
    ):
        """Schedule configuration for a single world."""
        task = self._loop.create_task(self._run_world(world_info, coro_factory))
        self._world_tasks[world_info.name] = task
        return task

    async def _cancel_world(self, world_name: str):
        """Cancel an in-progress world config task."""
        task = self._world_tasks.pop(world_name, None)
        if task and not task.done():
            task.cancel()
            raise asyncio.CancelledError

    def _build_world_infos(self, spec: ServeConfig) -> dict[str, WorldInfo]:
        world_infos: dict[str, WorldInfo] = {}

        my_id = spec.stage.id
        for k, v in spec.flow_graph.items():
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

    async def _run_world(
        self, world_info: WorldInfo, coro_factory: Callable[[], Awaitable[None]]
    ):
        """Run and cleanup world configuration."""
        try:
            await coro_factory(world_info)
        except asyncio.CancelledError:
            raise
        finally:
            self._world_tasks.pop(world_info.name, None)
