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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from infscale import get_logger
from infscale.constants import HEART_BEAT_PERIOD
from infscale.monitor.cpu import CPUStats, DRAMStats
from infscale.monitor.gpu import GpuStat, VramStat
from infscale.utils.timer import Timer

if TYPE_CHECKING:
    from grpc import ServicerContext
    from infscale.controller.controller import Controller

DEFAULT_TIMEOUT = 2 * HEART_BEAT_PERIOD
WMA_WEIGHT = 0.9


logger = None


@dataclass
class AgentResources:
    """Class for keeping agent resources."""

    gpu_stats: list[GpuStat]
    vram_stats: list[VramStat]
    cpu_stats: CPUStats
    dram_stats: DRAMStats


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

        self.resources: AgentResources = None

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

    def set_resources(
        self,
        gpu_stats: list[GpuStat],
        vram_stats: list[VramStat],
        cpu_stats: CPUStats,
        dram_stats: DRAMStats,
    ) -> None:
        curr_res = self.resources

        wma_gpu, wma_vram, wma_cpu, wma_dram = self._compute_wma_resources(
            curr_res, gpu_stats, vram_stats, cpu_stats, dram_stats
        )

        resources = AgentResources(wma_gpu, wma_vram, wma_cpu, wma_dram)

        self.resources = resources

    def _compute_wma_resources(
        self,
        curr_res: AgentResources,
        gpu_stats: list[GpuStat],
        vram_stats: list[VramStat],
        cpu_stats: CPUStats,
        dram_stats: DRAMStats,
    ) -> tuple[list[GpuStat], list[DRAMStats], CPUStats, DRAMStats]:
        """Compute resources using weighted moving average."""

        if curr_res is None:
            # first set of resources, return those
            return gpu_stats, vram_stats, cpu_stats, dram_stats

        gpu_wma = [
            self._compute_wma(gpu_stat, curr_res.gpu_stats[i])
            for i, gpu_stat in enumerate(gpu_stats)
        ]
        vram_wma = [
            self._compute_wma(vram_stat, curr_res.vram_stats[i])
            for i, vram_stat in enumerate(vram_stats)
        ]
        cpu_wma = self._compute_wma(cpu_stats, curr_res.cpu_stats)
        dram_wma = self._compute_wma(dram_stats, curr_res.dram_stats)

        return gpu_wma, vram_wma, cpu_wma, dram_wma

    def _compute_wma(
        self,
        new_stat: Union[GpuStat, VramStat, CPUStats, DRAMStats],
        old_stat: Union[GpuStat, VramStat, CPUStats, DRAMStats, None],
    ) -> Union[GpuStat, VramStat]:
        """Compute WMA for numeric values."""
        if old_stat is None:
            return new_stat

        match new_stat:
            case VramStat():
                new_stat.used = self._wma(new_stat.used, old_stat.used)

            case GpuStat():
                new_stat.util = self._wma(new_stat.util, old_stat.util)

            case CPUStats():
                new_stat.load = self._wma(new_stat.load, old_stat.load)
                new_stat.current_frequency = self._wma(
                    new_stat.current_frequency, old_stat.current_frequency
                )

            case DRAMStats():
                new_stat.used = self._wma(new_stat.used, old_stat.used)

        return new_stat

    def _wma(self, curr_val: float, new_val: float) -> float:
        """Return WMA between old and current value."""
        wma = (1 - WMA_WEIGHT) * float(curr_val) + (WMA_WEIGHT * float(new_val))

        return wma

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
