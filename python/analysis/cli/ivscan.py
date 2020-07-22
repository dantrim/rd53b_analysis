import click


@click.group(name="ivscan")
def cli():
    """The ivscan CLI group."""


@cli.command()
@click.argument("config")
def ivscan(config):
    print(f"ivscan input: config = {config}")
