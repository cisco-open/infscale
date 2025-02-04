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
import atexit
import requests
import yaml
from infscale.actor.agent import Agent
from infscale.constants import (APISERVER_ENDPOINT, APISERVER_PORT,
                                CONTROLLER_PORT, DEFAULT_DEPLOYMENT_POLICY, LOCALHOST)
from infscale.controller import controller as ctrl
from infscale.controller.ctrl_dtype import JobAction, JobActionModel
from infscale.monitor.gpu_profiler import GPUProfiler


@click.group()
def start():
    """Start command."""
    pass


@start.command()
@click.option("--port", default=CONTROLLER_PORT, help="port number")
@click.option("--apiport", default=APISERVER_PORT, help="port number for api server")
@click.option("--policy", default=DEFAULT_DEPLOYMENT_POLICY, help="deployment policy")
def controller(port: int, apiport: int, policy: str):
    """Run controller."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(ctrl.Controller(port=port, apiport=apiport, policy=policy).run())


@start.command()
@click.option("--host", default=LOCALHOST, help="Controller's IP or hostname")
@click.option("--port", default=CONTROLLER_PORT, help="Controller's port number")
@click.argument("id")
def agent(host: str, port: int, id: str):
    """Run agent."""
    endpoint = f"{host}:{port}"

    # Start the GPU profiler when the agent starts.
    profiler = GPUProfiler()
    profiler.start()

    # Register profiler.stop() to be called automatically when the process exits.
    atexit.register(profiler.stop)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        Agent(
            id=id,
            endpoint=endpoint,
        ).run()
    )


@start.command()
@click.option("--endpoint", default=APISERVER_ENDPOINT, help="Controller's endpoint")
@click.argument("config", required=True)
def job(endpoint: str, config: str) -> None:
    """Start a job with config."""
    with open(config) as f:
        job_config = yaml.safe_load(f)

    payload = JobActionModel(
        action=JobAction.START,
        config=job_config,
    ).model_dump_json()

    try:
        response = requests.post(
            f"{endpoint}/job",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            click.echo(f"{response.status_code}: Job started successfully")
        else:
            click.echo(f"{response.status_code}: {response.content.decode('utf-8')}")
    except requests.exceptions.RequestException as e:
        click.echo(f"Error making request: {e}")
