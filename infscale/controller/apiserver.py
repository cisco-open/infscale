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
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from infscale.constants import APISERVER_PORT
from infscale.controller.ctrl_dtype import (JobAction, JobActionModel, ReqType,
                                            Response)
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
