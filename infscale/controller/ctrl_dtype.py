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

"""ctrl_dtype.py."""

from enum import Enum
from typing import Optional

from infscale.config import Dataset, JobConfig, StageConfig, WorkerInfo
from pydantic import BaseModel, model_validator


class ReqType(str, Enum):
    """Enum class for request type."""

    UNKNOWN = "unknown"
    JOB_ACTION = "job_action"


class CommandAction(str, Enum):
    """Enum class for request type."""

    START = "start"   # CLI - Controller start command
    STOP = "stop"     # CLI - Controller stop command
    UPDATE = "update" # CLI - Controller update command
    SETUP = "setup"   # Controller - Agent setup job, assign port numbers to workers
    RESOURCE_STAT = "resource_stat" # Controller - Agent action to gather GPU and CPU resource statistics


class CommandActionModel(BaseModel):
    """Command action model."""

    action: CommandAction
    job_id: Optional[str] = None
    config: Optional[JobConfig] = None

    @model_validator(mode="after")
    def check_config_for_update(self):
        """Validate command action model."""
        if self.action in [CommandAction.UPDATE, CommandAction.START] and self.config is None:
            raise ValueError("config is required when updating a job")

        if self.action == CommandAction.STOP and self.job_id is None:
            raise ValueError("job id is required stopping or updating a job")
        return self


class ServeSpec(BaseModel):
    """ServiceSpec model."""

    name: str
    model: str
    stage: StageConfig
    dataset: Dataset
    flow_graph: dict[str, list[WorkerInfo]]
    device: str = "cpu"
    nfaults: int = 0  # no of faults to tolerate, default: 0 (no fault tolerance)
    micro_batch_size: int = 8
    fwd_policy: str = "random"


class Response(BaseModel):
    """Response model."""

    message: str
