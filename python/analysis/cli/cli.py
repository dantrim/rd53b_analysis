""" RD53B Analysis command line interface """

import click

from . import ivscan


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def rd53b():
    """Top-level entrypoint into RD53B analysis infrastructure"""


rd53b.add_command(ivscan.ivscan)
