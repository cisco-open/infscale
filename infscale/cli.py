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

"""command line tool."""
import click

from infscale.cmd.start import start
from infscale.cmd.status import status
from infscale.cmd.stop import stop
from infscale.cmd.update import update
from infscale.version import VERSION


@click.group()
@click.version_option(version=VERSION)
def cli():  # noqa: D103
    pass


cli.add_command(start)
cli.add_command(stop)
cli.add_command(update)
cli.add_command(status)

if __name__ == "__main__":
    cli()
