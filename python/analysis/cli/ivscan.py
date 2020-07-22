import click

from pathlib import Path
import sys
import os

from analysis.ivscan import ivscan as iv


def file_does_not_exist(input_file):
    """
    Checks if the provided path points to an existing file.

    Args:
        input_file (string): path to a file.

    Returns:
        bool: True if the file could not be found, False otherwise.
    """

    p = Path(input_file)
    file_is_ok = p.exists() and p.is_file()
    return not file_is_ok


@click.group(name="ivscan")
def cli():
    """The ivscan CLI group."""


@cli.command()
@click.argument("input-file")
@click.option(
    "--current",
    type=click.Choice(["digital", "analog"], case_sensitive=False),
    default="digital",
)
@click.option("--summary", is_flag=True)
def ivscan(input_file, current, summary):

    if file_does_not_exist(input_file):
        print(f"ERROR Provided input file (={input_file}) could not be found")
        sys.exit(1)

    if summary:
        ok, err = iv.plot_summary(input_file, current)
        if not ok:
            print(f"ERROR Failed to plot IV-scan summary: {err}")
            sys.exit(1)
    else:
        ok, err = iv.plot(input_file, current)
        if not ok:
            print(f"ERROR Failed to plot IV-scan data: {err}")
            sys.exit(1)
