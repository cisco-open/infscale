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

"""apiserver class."""
from __future__ import annotations

import asyncio
from enum import Enum
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from infscale.config import Dataset, JobConfig, StageConfig, WorkerInfo
from infscale.constants import APISERVER_PORT
from pydantic import BaseModel, model_validator
from uvicorn import Config, Server

if TYPE_CHECKING:
    from infscale.controller.controller import Controller

_ctrl = None
app = FastAPI()


class ApiServer:
    """ApiServer class."""

    def __init__(self, ctrl: Controller, port: int = APISERVER_PORT):
        """Initialize an instance."""
        global _ctrl
        _ctrl = ctrl

        self.port = port

    async def run(self):
        """Run apiserver."""
        config = Config(
            app=app,
            host="0.0.0.0",
            port=self.port,
            loop=asyncio.get_event_loop(),
        )

        server = Server(config)
        await server.serve()


class ReqType(str, Enum):
    """Enum class for request type."""

    UNKNOWN = "unknown"
    JOB_ACTION = "job_action"


class JobAction(str, Enum):
    """Enum class for request type."""

    START = "start"
    STOP = "stop"
    UPDATE = "update"


class JobActionModel(BaseModel):
    action: JobAction
    job_id: Optional[str] = None
    config: Optional[JobConfig] = None

    @model_validator(mode="after")
    def check_config_for_update(self):
        if self.action in [JobAction.UPDATE, JobAction.START] and self.config is None:
            raise ValueError("config is required when updating a job")

        if self.action == JobAction.STOP and self.job_id is None:
            raise ValueError("job id is required stopping or updating a job")
        return self


class ServeSpec(BaseModel):
    """ServiceSpec model."""

    name: str
    model: str
    stage: StageConfig
    dataset: Dataset
    flow_graph: dict[str, list[WorkerInfo]]
    rank_map: dict[str, int]
    device: str = "cpu"
    nfaults: int = 0  # no of faults to tolerate, default: 0 (no fault tolerance)
    micro_batch_size: int = 8
    fwd_policy: str = "random"


class Response(BaseModel):
    """Response model."""

    message: str


@app.post("/job", response_model=Response)
async def manage_job(job_action: JobActionModel):
    """Start or Stop a job."""
    try:
        await _ctrl.handle_fastapi_request(
            ReqType.JOB_ACTION,
            job_action,
        )
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content=e.detail)

    res = "job started" if job_action.action == JobAction.START else "job stopped"

    return JSONResponse(status_code=status.HTTP_200_OK, content=res)


@app.put("/job", response_model=Response)
async def update_job(job_action: JobActionModel):
    """Update job with new config."""
    try:
        await _ctrl.handle_fastapi_request(ReqType.JOB_ACTION, job_action)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content=e.detail)

    res = "job updated"

    return JSONResponse(status_code=status.HTTP_200_OK, content=res)
