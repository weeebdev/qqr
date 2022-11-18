import logging
import click

from .version import __version__
from .decoder import *

logging.basicConfig()
log = logging.getLogger(__name__)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(version=__version__)
@click.option("-f", "--file", type=click.Path(exists=True), help="Path to the file to be processed")
def qqr(file):
    """qqr: a command line tool to query qrcode from terminal."""
    decode(file)


if __name__ == "__main__":
    qqr()
