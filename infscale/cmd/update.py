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

import click
import requests
import yaml
from infscale.constants import APISERVER_ENDPOINT
from infscale.controller.apiserver import JobAction, JobActionModel


@click.group()
def update():
    """Update command."""
    pass


@update.command()
@click.option("--endpoint", default=APISERVER_ENDPOINT, help="Controller's endpoint")
@click.argument("job_id", required=True)
@click.argument("config", required=True)
def job(endpoint: str, job_id: str, config: str):
    """Update a job with JOB_ID using a new config."""

    with open(config) as f:
        job_config = yaml.safe_load(f)

    payload = JobActionModel(
        job_id=job_id, action=JobAction.UPDATE, config=job_config
    ).model_dump_json()

    try:
        response = requests.put(
            endpoint, data=payload, headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            click.echo("Job updated successfully.")
        else:
            click.echo(f"Failed to update job. Status code: {response.status_code}")
            click.echo(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        click.echo(f"Error making request: {e}")
