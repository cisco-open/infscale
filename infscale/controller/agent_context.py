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

"""AgentContext class."""
from __future__ import annotations

import asyncio
from enum import Enum
from typing import TYPE_CHECKING, Union

from infscale import get_logger
from infscale.common.constants import HEART_BEAT_PERIOD
from infscale.monitor.cpu import CPUStats, DRAMStats
from infscale.monitor.gpu import GpuStat, VramStat
from infscale.utils.timer import Timer


if TYPE_CHECKING:
    from grpc import ServicerContext

    from infscale.controller.controller import Controller

DEFAULT_TIMEOUT = 2 * HEART_BEAT_PERIOD
WMA_WEIGHT = 0.9
CPU_LOAD_THRESHOLD = 30

logger = None


class DeviceType(str, Enum):
    """Enum class for device type."""

    GPU = "gpu"
    CPU = "cpu"


class AgentResources:
    """Class for keeping agent resources."""

    def __init__(
        self,
        gpu_stats: list[GpuStat] = None,
        vram_stats: list[VramStat] = None,
        cpu_stats: CPUStats = None,
        dram_stats: DRAMStats = None,
    ):
        """Initialize AgentResources instance."""
        self.gpu_stats: list[GpuStat] = gpu_stats
        self.vram_stats: list[VramStat] = vram_stats
        self.cpu_stats: CPUStats = cpu_stats
        self.dram_stats: DRAMStats = dram_stats

    def get_n_set_device(self, dev_type: DeviceType) -> str | None:
        """
        Return device string based on device type.

        In case of GPU, select the first unused GPU and update its used property.
        """
        if dev_type == DeviceType.CPU:
            return "cpu"

        stat = next((gpu for gpu in self.gpu_stats if not gpu.used), None)

        if stat is not None:
            stat.used = True
            return f"cuda:{stat.id}"
        return None

    def update_gpu_wma(self, gpu_stats: list[GpuStat]) -> None:
        """Update wma for gpu stats."""
        if self.gpu_stats is None:
            self.gpu_stats = gpu_stats
            return

        for i, gpu_stat in enumerate(gpu_stats):
            # used is a boolean variable; so assign it directly
            self.gpu_stats[i].used = gpu_stat.used

            util = self._wma(self.gpu_stats[i].util, gpu_stat.util)
            self.gpu_stats[i].util = util

    def update_vram_wma(self, vram_stats: list[VramStat]) -> None:
        """Update wma for vram stats."""
        if self.vram_stats is None:
            self.vram_stats = vram_stats
            return

        for i, vram_stat in enumerate(vram_stats):
            used = self._wma(self.vram_stats[i].used, vram_stat.used)
            total = self._wma(self.vram_stats[i].total, vram_stat.total)
            self.vram_stats[i].used = used
            self.vram_stats[i].total = total

    def update_cpu_wma(self, cpu_stats: CPUStats) -> None:
        """Update wma for cpu stats."""
        if self.cpu_stats is None:
            self.cpu_stats = cpu_stats
            return

        load = self._wma(self.cpu_stats.load, cpu_stats.load)
        freq = self._wma(self.cpu_stats.current_frequency, cpu_stats.current_frequency)
        self.cpu_stats.load = load
        self.cpu_stats.current_frequency = freq

    def update_dram_wma(self, dram_stats: DRAMStats) -> None:
        """Update wma for dram stats."""
        if self.dram_stats is None:
            self.dram_stats = dram_stats
            return

        used = self._wma(self.dram_stats.used, dram_stats.used)
        total = self._wma(self.dram_stats.total, dram_stats.total)
        self.dram_stats.used = used
        self.dram_stats.total = total

    def _wma(self, curr_val: float, new_val: float) -> float:
        """Compuate weighted moving average."""
        wma = (1 - WMA_WEIGHT) * curr_val + WMA_WEIGHT * new_val
        return wma


class AgentContext:
    """Agent Context class."""

    def __init__(self, ctrl: Controller, id: str, ip: str):
        """Initialize instance."""
        global logger
        logger = get_logger()

        self.ctrl = ctrl
        self.id: str = id
        self.ip: str = ip

        self.grpc_ctx = None
        self.grpc_ctx_event = asyncio.Event()

        self.alive: bool = False
        self.timer: Timer = None

        self.resources = AgentResources()

    def get_grpc_ctx(self) -> Union[ServicerContext, None]:
        """Return grpc context (i.e., servicer context)."""
        return self.grpc_ctx

    def set_grpc_ctx(self, ctx: ServicerContext):
        """Set grpc servicer context."""
        self.grpc_ctx = ctx

    def get_grpc_ctx_event(self) -> asyncio.Event:
        """Return grpc context event."""
        return self.grpc_ctx_event

    def set_grpc_ctx_event(self):
        """Set event for grpc context.

        This will release the event.
        """
        self.grpc_ctx_event.set()

    def avail_gpu_count(self) -> int:
        """Return unused GPU count."""
        if self.resources.gpu_stats is None:
            return 0

        return sum(not gpu.used for gpu in self.resources.gpu_stats)

    def avail_gpus(self) -> set[int]:
        """Return a set of IDs of available GPUs."""
        if self.resources.gpu_stats is None:
            return set()

        return {gpu.id for gpu in self.resources.gpu_stats if not gpu.used}

    def update_resource_statistics(
        self,
        gpu_stats: list[GpuStat],
        vram_stats: list[VramStat],
        cpu_stats: CPUStats,
        dram_stats: DRAMStats,
    ) -> None:
        """Update statistis on resources."""
        self.resources.update_gpu_wma(gpu_stats)
        self.resources.update_vram_wma(vram_stats)
        self.resources.update_cpu_wma(cpu_stats)
        self.resources.update_dram_wma(dram_stats)

    def keep_alive(self):
        """Set agent's status to alive."""
        logger.debug(f"keeping agent context alive for {self.id}")
        self.alive = True

        if not self.timer:
            self.timer = Timer(DEFAULT_TIMEOUT, self.ctrl.reset_agent_context, self.id)
        self.timer.renew()

    def reset(self):
        """Reset the agent context state."""
        logger.debug(f"agent {self.id} context reset")
        self.alive = False
        self.timer = None

        self.grpc_ctx = None
        self.set_grpc_ctx_event()
