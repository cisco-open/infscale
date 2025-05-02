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

"""status.py."""


import json

import click
import requests

from infscale.common.constants import APISERVER_ENDPOINT


@click.group()
def status():
    """Status command."""
    pass


@status.command()
@click.option("--endpoint", default=APISERVER_ENDPOINT, help="Controller's endpoint")
@click.argument("job_id", required=True)
def job(endpoint: str, job_id: str) -> None:
    """Get status of a job."""
    try:
        response = requests.get(f"{endpoint}/job/{job_id}")

        result = {"status_code": response.status_code, "content": response.json()}

        if response.status_code == 200:
            click.echo(response.content.decode("utf-8"))
        else:
            click.echo(json.dumps(result, indent=2))
    except requests.exceptions.RequestException as e:
        click.echo(f"Error making request: {e}")
