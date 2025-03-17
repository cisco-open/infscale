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

"""exceptions.py."""


class InfScaleException(Exception):
    """An unbrella class for infscale-specific exceptions."""

    pass


class InvalidJobStateAction(InfScaleException):
    """Exception for invalid actions in a job state."""

    def __init__(self, job_id, action, state):
        """Initialize InvalidJobStateAction exception instance."""
        self.job_id = job_id
        self.action = action
        self.state = state

        super().__init__(f"{job_id}: {action} disallowed in {state}.")


class InvalidConfig(InfScaleException):
    """Exception for invalid job configuration."""

    def __init__(self, err_msg: str):
        """Initialize InvalidConfig exception instance."""
        super().__init__(err_msg)


class InsufficientResources(InfScaleException):
    """Exception for insufficient agent resources."""

    def __init__(self, err_msg: str):
        """Initialize InsufficientResources exception instance."""
        super().__init__(err_msg)
