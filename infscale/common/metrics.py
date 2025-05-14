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

"""metrics.py."""

from dataclasses import dataclass


@dataclass
class PerfMetrics:
    """PerfMetrics class."""

    # the number of requests in the system (worker)
    qlevel: float = 0.0
    # second to serve one request
    delay: float = 0.0
    # the number of requests arrived per second
    input_rate: float = 0.0
    # the number of requests served per second
    output_rate: float = 0.0

    # a factor used to evaluate congestion and to compute a desired rate
    # for relieving the congestion
    _weight_factor: float = 1.5

    def update(
        self, qlevel: float, delay: float, input_rate: float, output_rate: float
    ) -> None:
        """Update metric's values."""
        self.qlevel = qlevel
        self.delay = delay
        self.input_rate = input_rate
        self.output_rate = output_rate

    def is_congested(self) -> bool:
        """Return true if queue continues to build up while throughput is saturated."""
        # measure qlevel is larger than output rate * weight factor
        cond = self.qlevel > self.output_rate * self._weight_factor

        return cond

    def rate_to_decongest(self) -> float:
        """Return a required rate to relieve congestion."""
        return self.input_rate * self._weight_factor

    def __str__(self) -> str:
        """Return string representation for the object."""
        return f"qlevel: {self.qlevel}, delay: {self.delay}, input_rate: {self.input_rate}, output_rate: {self.output_rate}"
