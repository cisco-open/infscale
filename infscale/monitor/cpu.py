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

"""CPU monitoring class."""
import json
from dataclasses import asdict, dataclass
from typing import Union

import psutil
from google.protobuf.json_format import Parse
from infscale.proto import management_pb2 as pb2


@dataclass
class CPUStats:
    """CPU statistics."""

    total_cpus: int
    max_frequency: float
    current_frequency: float
    min_frequency: float
    load: float  # overall CPU load as a percentage


@dataclass
class DRAMStats:
    """DRAM statistics."""

    used: int
    total: int


class CpuMonitor:
    """CpuMonitor class."""

    def get_metrics(self) -> tuple[CPUStats, DRAMStats]:
        """Start to monitor CPU statistics"""
        # total number of CPUs (logical)
        total_cpus = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=1)
        dram_stats = psutil.virtual_memory()

        # current CPU clock speed
        cpu_freq = psutil.cpu_freq()

        cpu_stats = CPUStats(
            total_cpus, cpu_freq.max, cpu_freq.current, cpu_freq.min, cpu_percent
        )
        dram_stats = DRAMStats(
            dram_stats.used,
            dram_stats.total,
        )

        return cpu_stats, dram_stats

    @staticmethod
    def stats_to_proto(
        stats: Union[CPUStats, DRAMStats]
    ) -> Union[pb2.CpuStats, pb2.DramStats]:
        """Convert CPUStats or DRAMStats to a protobuf messages."""
        if isinstance(stats, CPUStats):
            pb_msg_obj = pb2.CpuStats
        elif isinstance(stats, DRAMStats):
            pb_msg_obj = pb2.DramStats

        stats_str = json.dumps(asdict(stats))

        msg = Parse(stats_str, pb_msg_obj())

        return msg
