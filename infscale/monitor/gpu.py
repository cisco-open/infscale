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

"""GPU monitoring class."""
import asyncio
import dataclasses
import json
from dataclasses import dataclass
from enum import Enum
from typing import Union

from google.protobuf.json_format import MessageToJson, Parse
from infscale import get_logger
from infscale.proto import management_pb2 as pb2
from pynvml import (nvmlDeviceGetComputeRunningProcesses, nvmlDeviceGetCount,
                    nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlDeviceGetName, nvmlDeviceGetUtilizationRates, nvmlInit)

DEFAULT_INTERVAL = 10  # 10 seconds

logger = None


class GpuType(str, Enum):
    """Gpu Type enum class."""

    def __str__(self):
        """__str__ dunder method."""
        return str(self.value)

    UNKNOWN = "unknown"
    T4 = "T4"
    V100 = "V100"
    A100 = "A100"


@dataclass
class GpuStat:
    """GPU statistics."""

    id: int
    type: GpuType
    used: bool
    util: int


@dataclass
class VramStat:
    """VRAM statistics."""

    id: int
    used: int
    total: int

    def __post_init__(self):
        """Do post init manipulation."""
        if type(self.used) is str:
            self.used = int(self.used)

        if type(self.total) is str:
            self.total = int(self.total)


class GpuMonitor:
    """GpuMonitor class."""

    def __init__(self, interval: int = DEFAULT_INTERVAL):
        """Initialize GpuMonitor instance."""
        global logger
        logger = get_logger()

        self.gpu_available = False
        try:
            nvmlInit()
            self.gpu_available = True
            logger.debug("GPU successfully initialized.")
        except Exception as e:
            logger.warning("Failed to initialize NVML. No GPU available.")
            logger.debug(f"Exception: {e}")

        self.interval = interval

        self.mon_event = asyncio.Event()
        self.computes = list()
        self.mems = list()

    def get_metrics(self) -> tuple[list[GpuStat], list[VramStat]]:
        """Return gpu and vram statistics."""
        if not self.gpu_available:
            logger.info("no GPU available, skipping metrics collection.")
            return [], []

        computes, mems = self._get_gpu_stats()

        return computes, mems

    async def metrics(self) -> tuple[list[GpuStat], list[VramStat]]:
        """Return statistics on GPU resources."""
        # Wait until data refreshes
        logger.debug("wait for monitor event")
        await self.mon_event.wait()
        logger.debug("monitor event is set")

        return self.computes, self.mems

    async def start(self):
        """Start to monitor GPU statistics - utilization and vram usage.

        utilization reports device's utilization.
        it's not a per-application metric.
        vram usage is also an aggregated metric, not a per-application metric.
        """
        if not self.gpu_available:
            logger.info("no GPU available, skipping metrics collection.")
            return

        while True:
            computes, mems = self._get_gpu_stats()

            self.mems = mems
            self.computes = computes
            # unlbock metrics() call
            self.mon_event.set()
            # block metrics() call again
            self.mon_event.clear()

            await asyncio.sleep(self.interval)

    def _get_gpu_stats(self) -> tuple[list[GpuStat], list[VramStat]]:
        """Return GPU and VRam resources."""
        count = nvmlDeviceGetCount()
        mems = [None] * count
        computes = [None] * count
        for i in range(count):
            try:
                handle = nvmlDeviceGetHandleByIndex(i)

                # gpu memory information
                mem_info = nvmlDeviceGetMemoryInfo(handle)
                # device name
                name = nvmlDeviceGetName(handle)
                # GPU utilization
                util = nvmlDeviceGetUtilizationRates(handle).gpu
                # processes running on gpu
                processes = nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception as e:
                # TODO: need to catch more specific exception
                #       Exception is too generic
                logger.debug(f"failed to retrieve info for GPU {i}")
                logger.debug(f"exception: {e}")
                continue

            mems[i] = VramStat(i, mem_info.total, mem_info.used)

            dev_type = GpuType.UNKNOWN
            for t in GpuType:
                if str(t) in name:
                    dev_type = t
                    break

            used = len(processes) > 0

            computes[i] = GpuStat(i, dev_type, used, util)

        return computes, mems

    @staticmethod
    def stats_to_proto(
        stats: Union[list[GpuStat], list[VramStat]],
    ) -> Union[None, list[pb2.GpuStat], list[pb2.VramStat]]:
        """Convert GpuStats or VramStats to a list of protobuf messages."""
        if not isinstance(stats, list) or len(stats) == 0:
            return None

        if isinstance(stats[0], GpuStat):
            pb_msg_obj = pb2.GpuStat
        elif isinstance(stats[0], VramStat):
            pb_msg_obj = pb2.VramStat
        else:
            return None

        proto = list()
        for stat in stats:
            json_str = json.dumps(dataclasses.asdict(stat))
            message = Parse(json_str, pb_msg_obj())
            proto.append(message)

        return proto

    @staticmethod
    def proto_to_stats(
        proto: Union[list[pb2.GpuStat], list[pb2.VramStat]],
    ) -> Union[None, list[GpuStat], list[VramStat]]:
        """Convert a list of protobuf messages to GpuStats or VramStats."""
        global logger
        logger = get_logger()

        if len(proto) == 0:
            logger.debug("no protobuf message")
            return None

        if isinstance(proto[0], pb2.GpuStat):
            target = GpuStat
        elif isinstance(proto[0], pb2.VramStat):
            target = VramStat
        else:
            logger.debug(f"unknown message: {type(proto[0])}")
            return None

        stats = list()
        for msg in proto:
            json_str = MessageToJson(msg, always_print_fields_with_no_presence=True)
            json_obj = json.loads(json_str)
            inst = target(**json_obj)
            stats.append(inst)

        return stats
