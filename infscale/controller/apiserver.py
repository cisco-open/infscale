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

from fastapi import FastAPI
from infscale.config import Dataset, JobConfig, Stage, WorkerInfo
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
    SERVE = "serve"
    JOB_ACTION = "job_action"


class JobAction(str, Enum):
    """Enum class for request type."""

    START = "start"
    STOP = "stop"
    UPDATE = "update"


class JobActionModel(BaseModel):
    job_id: str
    action: JobAction
    config: Optional[JobConfig] = None

    @model_validator(mode="after")
    def check_config_for_update(self):
        if self.action == JobAction.UPDATE and self.config is None:
            raise ValueError("config is required when updating a job")
        return self


class ServeSpec(BaseModel):
    """ServiceSpec model."""

    name: str
    model: str
    stage: Stage
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


@app.post("/models", response_model=Response)
async def serve(spec: ServeSpec):
    """Serve a model."""
    await _ctrl.handle_fastapi_request(ReqType.SERVE, spec)

    res = {"message": "started serving"}
    return res


@app.post("/job", response_model=Response)
async def manage_job(job_action: JobActionModel):
    """Start or Stop a job."""
    await _ctrl.handle_fastapi_request(
        ReqType.JOB_ACTION, {"job_id": job_action.job_id, "action": job_action.action}
    )

    res = {
        "message": "job started"
        if job_action.action == JobAction.START
        else "job stopped"
    }
    return res


@app.put("/job", response_model=Response)
async def update_job(job_action: JobActionModel):
    """Update job with new config."""
    await _ctrl.handle_fastapi_request(
        ReqType.JOB_ACTION,
        JobActionModel(
            job_id=job_action.job_id, action=job_action.action, config=job_action.config
        ),
    )

    res = {"message": "job updated"}
    return res
