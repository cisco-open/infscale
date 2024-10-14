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

import asyncio
import click

from infscale.actor.agent import Agent
from infscale.constants import APISERVER_PORT, CONTROLLER_PORT, LOCALHOST
from infscale.controller import controller as ctrl


@click.group()
def start():
    """Start command."""
    pass


@start.command()
@click.option("--port", default=CONTROLLER_PORT, help="port number")
@click.option("--apiport", default=APISERVER_PORT, help="port number for api server")
def controller(port: int, apiport: int):
    """Run controller."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(ctrl.Controller(port=port, apiport=apiport).run())


@start.command()
@click.option("--host", default=LOCALHOST, help="Controller's IP or hostname")
@click.option("--port", default=CONTROLLER_PORT, help="Controller's port number")
@click.option("--controller/--no-controller", default=False, help="Use controller")
@click.argument("id")
@click.argument("jobconfig", nargs=-1)
def agent(host: str, port: int, controller: bool, id: str, jobconfig: str):
    """Run agent."""
    endpoint = f"{host}:{port}"

    # Don't use the following code asyncio.run()
    # see https://github.com/grpc/grpc/issues/32480 for more details

    ctrl_instance = ctrl.Controller(list(jobconfig))

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        Agent(
            id=id,
            endpoint=endpoint,
            use_controller=controller,
            controller=ctrl_instance,
        ).run()
    )


@start.command()
@click.argument("job_id", required=True)
def job(job_id):
    """Start a job with JOB_ID."""
    click.echo(f"Starting job {job_id}...")
