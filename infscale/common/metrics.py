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

import math
from collections import deque
from dataclasses import dataclass


class RollingStats:
    """RollingStatus class."""

    def __init__(self, window_size: int = 10) -> None:
        """Initialize an instance of RollingStats."""
        self._window = deque(maxlen=window_size)
        self._window_size = window_size
        self._sum = 0.0
        self._sum_sq = 0.0
        self._std = 0.0

    def update(self, val: float) -> None:
        """Update statistics."""
        if self.is_filled():
            old = self._window.popleft()
            self._sum -= old
            self._sum_sq -= old**2
        self._window.append(val)

        self._sum += val
        self._sum_sq += val**2

    def is_filled(self) -> bool:
        """Return true if window is filled."""
        return len(self._window) == self._window_size

    def mean(self) -> float:
        """Return an average."""
        if not self.is_filled():
            return 0.0

        return self._sum / self._window_size

    def std(self) -> float:
        """Return a standard deviation."""
        if not self.is_filled():
            return 0.0

        mean_sq = self._sum_sq / self._window_size
        mean = self._sum / self._window_size

        try:
            self._std = math.sqrt(mean_sq - mean**2)
        except ValueError:
            # in some edge cases ValueError: math domain error is raised
            # if that happens, ignore the error and return previous value.
            pass

        return self._std


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

    # a factor used to set qlevel threshold
    _sensitivity_factor: float = 2
    _qthresh: float = 10**9

    _qlevel_rs: RollingStats = None
    _in_rate_rs: RollingStats = None
    _out_rate_rs: RollingStats = None

    def update(
        self, qlevel: float, delay: float, input_rate: float, output_rate: float
    ) -> None:
        """Update metric's values."""
        self.qlevel = qlevel
        self.delay = delay
        self.input_rate = input_rate
        self.output_rate = output_rate

    def update_stats(self) -> None:
        """Update stats on qlevel, input and output rates with a window of samples.

        This method doesn't need to be called for every worker in a job. Instead,
        call this method for workers that need to monitor congestion.
        This method should be only called in conjunction with update() for accurate
        calculation.
        """
        self._in_rate_rs.update(self.input_rate)
        self._out_rate_rs.update(self.output_rate)
        self._qlevel_rs.update(self.qlevel)

        if not self._qlevel_rs.is_filled():
            return

        mean = self._qlevel_rs.mean()
        std = self._qlevel_rs.std()

        self._qthresh = mean + self._sensitivity_factor * std

    def is_congested(self) -> bool:
        """Return true if queue continues to build up."""
        return self.qlevel > self._qthresh

    def rate_to_decongest(self) -> float:
        """Return a required rate to relieve congestion.

        The required rate means the additional rate by subtracting the average
        output rate from the average input rate.
        """
        return max(self._in_rate_rs.mean() - self._out_rate_rs.mean(), 0.0)

    def __post_init__(self) -> None:
        """Do post init work."""
        self._qlevel_rs = RollingStats()
        self._in_rate_rs = RollingStats()
        self._out_rate_rs = RollingStats()

    def __str__(self) -> str:
        """Return string representation for the object."""
        str_rep = (
            f"qlevel: {self.qlevel:.6f}, delay: {self.delay:.6f}, "
            + f"input_rate: {self.input_rate:.6f}, "
            + f"output_rate: {self.output_rate:.6f}"
        )

        return str_rep
