"""command line tool."""
import asyncio

import click

from infscale.actor.agent import Agent
from infscale.constants import CONTROLLER_PORT, LOCALHOST
from infscale.controller import controller as ctrl
from infscale.version import VERSION


@click.group()
@click.version_option(version=VERSION)
def cli():  # noqa: D103
    pass


@cli.command()
@click.option("--port", default=CONTROLLER_PORT, help="port number")
def controller(port: int):
    """Run controller."""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(ctrl.Controller(port=port).run())


@cli.command()
@click.option("--host", default=LOCALHOST, help="Controller's IP or hostname")
@click.option("--port", default=CONTROLLER_PORT, help="Controller's port number")
@click.argument("id")
def agent(host: str, port: int, id: str):
    """Run agent."""
    endpoint = f"{host}:{port}"
    print(f"Controller endpoint: {endpoint}")

    # Don't use the following code asyncio.run()
    # see https://github.com/grpc/grpc/issues/32480 for more details

    loop = asyncio.get_event_loop()
    loop.run_until_complete(Agent(id=id, endpoint=endpoint).run())


if __name__ == "__main__":
    cli()
