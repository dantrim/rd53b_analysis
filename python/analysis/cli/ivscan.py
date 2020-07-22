import click

@click.group(name = "ivscan")
def cli():
    pass

@cli.command()
@click.argument("config")
def dummy(config):
    pint(f"ivscan input: config = {config}")
