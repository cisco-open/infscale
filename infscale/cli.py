"""command line tool."""
import click

from infscale.version import VERSION


@click.group()
@click.version_option(version=VERSION)
def cli():  # noqa
    pass


@cli.command()
def controller():
    """Run controller."""
    print("Run controller")


@cli.command()
def agent():
    """Run agent."""
    print("Run agent")


if __name__ == "__main__":
    cli()
