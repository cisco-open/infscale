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

"""metrics_collector.py."""

import time

from infscale.common.metrics import PerfMetrics


class MetricsCollector:
    """MetricsCollector class."""

    def __init__(self, coeff: float = 0.9):
        """Initialize an instance."""
        self._batch_size = 1  # to correctly calculate throughput (no. of reqs / sec)
        self._coeff = coeff

        self._qlevel = 0
        self._delay = 0
        self._input_rate = 0
        self._output_rate = 0

        # key: sequence number
        # value: relative time that request entered the system
        self._metrics_map: dict[int, float] = {}
        self._arrived = 0
        self._served = 0
        self._last_time = time.perf_counter()

        # a flag to enable/disable metrics collection in router
        self._enable = True

    def set_batch_size(self, value: int) -> None:
        """Set batch size."""
        self._batch_size = value

    def enable_in_router(self, val: bool) -> None:
        """Enable or disable metrics collection."""
        self._enable = val

    def can_collect_in_router(self) -> bool:
        """Return true if metric collection is allowed in router."""
        return self._enable

    def update(self, seqno: int) -> None:
        """Update metrics."""
        if seqno not in self._metrics_map:
            self._metrics_map[seqno] = time.perf_counter()
            self._arrived += 1
        else:
            start = self._metrics_map[seqno]
            del self._metrics_map[seqno]

            qlevel = len(self._metrics_map) * self._batch_size
            self._qlevel = (1 - self._coeff) * self._qlevel + self._coeff * qlevel

            end = time.perf_counter()
            delay = end - start
            self._delay = (1 - self._coeff) * self._delay + self._coeff * delay

            self._served += 1

    def _compute_input_output_rates(self) -> None:
        now = time.perf_counter()

        rate = self._arrived * self._batch_size / (now - self._last_time)
        self._input_rate = (1 - self._coeff) * self._input_rate + self._coeff * rate

        rate = self._served * self._batch_size / (now - self._last_time)
        self._output_rate = (1 - self._coeff) * self._output_rate + self._coeff * rate

        # reset arrived/served counts
        self._arrived = 0
        self._served = 0
        # update the last time with the current time
        self._last_time = now

    def retrieve(self) -> PerfMetrics:
        """Return metrics.

        Note: This function should be called periodically (e.g., every second).
        """
        # we can only compute input / output rate every time we call this function
        # since we need an interval to compute throughput
        self._compute_input_output_rates()

        m = PerfMetrics(self._qlevel, self._delay, self._input_rate, self._output_rate)
        return m
