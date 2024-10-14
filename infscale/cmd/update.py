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


@click.group()
def update():
    """Update command."""
    pass


@update.command()
@click.argument("job_id", required=True)
@click.argument("config", required=True)
def job(job_id, config):
    """Update a job with JOB_ID using a new config."""
    click.echo(f"Updating job {job_id} with config {config}...")
