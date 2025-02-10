"""Helper utility functions to download and otherwise manage
containers required for flint"""

from __future__ import annotations

from argparse import ArgumentParser

from flint.logging import logger
from flint.options import BaseOptions


class FlintContainer(BaseOptions):
    """Item representing a Flint container"""

    name: str
    """Name of the container"""
    url: str
    """URL of the container that can be used to pull with apptainer, e.g. docker://alecthomson/aoflagger:latest"""
    filename: str
    """The expected filename of the container. This will be appended to the container directory path."""
    description: str | None = None
    """Short description on the purpose of the container"""


calibrate_container = FlintContainer(
    name="calibrate",
    filename="flint-containers_calibrate.sif",
    url="alecthomson/flint-containers:calibrate",
    description="Contains AO calibrate and addmodel",
)
wsclean_container = FlintContainer(
    name="wsclean",
    filename="flint-containers_wsclean.sif",
    url="alecthomson/flint-containers:wsclean",
    description="Container with the wsclean deconvolution software",
)
askapsoft_contaer = FlintContainer(
    name="askapsoft",
    filename="flint-containers_askapsoft.sif",
    url="alecthomson/flint-containers:askapsoft",
    description="Container with askapsoft (also known as yandasoft)",
)
aoflagger_contaer = FlintContainer(
    name="aoflagger",
    filename="flint-containers_aoflagger.sif",
    url="alecthomson/flint-containers:aoflagger",
    description="Container with aoflagger, used to autonomously flag measurement sets",
)
aegean_contaer = FlintContainer(
    name="aegean",
    filename="flint-containers_aegean.sif",
    url="alecthomson/flint-containers:aegean",
    description="Container with aegean, used to source find",
)

LIST_OF_KNOWN_CONTAINERS = (
    calibrate_container,
    wsclean_container,
    askapsoft_contaer,
    aoflagger_contaer,
    aegean_contaer,
)
KNOWN_CONTAINER_LOOKUP: dict[str, FlintContainer] = {
    v.name: v for v in LIST_OF_KNOWN_CONTAINERS
}


def log_known_containers() -> None:
    """Log the known containers"""

    for idx, (known_name, known_container) in enumerate(KNOWN_CONTAINER_LOOKUP.items()):
        logger.info(f"Container {idx + 1} of {len(LIST_OF_KNOWN_CONTAINERS)}")
        logger.info(f"\tName: {known_container.name}")
        logger.info(f"\tFilename: {known_container.filename}")
        logger.info(f"\tURL: {known_container.url}")
        logger.info(f"\tDescription: {known_container.description}")


def get_parser() -> ArgumentParser:
    """Create the CLI argument parser

    Returns:
        ArgumentParser: Constructed argument parser
    """
    parser = ArgumentParser(description="Download and manage Flint containers")
    subparsers = parser.add_subparsers(dest="mode")

    _ = subparsers.add_parser(
        name="list", help="List the containers that are known to Flint"
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "list":
        log_known_containers()


if __name__ == "__main__":
    cli()
