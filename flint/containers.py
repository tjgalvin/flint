"""Helper utility functions to download and otherwise manage
containers required for flint"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from flint.logging import logger
from flint.options import BaseOptions
from flint.sclient import pull_container


class FlintContainer(BaseOptions):
    """Item representing a Flint container"""

    name: str
    """Name of the container"""
    uri: str
    """URL of the container that can be used to pull with apptainer, e.g. docker://alecthomson/aoflagger:latest"""
    filename: str
    """The expected filename of the container. This will be appended to the container directory path."""
    description: str | None = None
    """Short description on the purpose of the container"""


calibrate_container = FlintContainer(
    name="calibrate",
    filename="flint-containers_calibrate.sif",
    uri="docker://alecthomson/flint-containers:calibrate",
    description="Contains AO calibrate and addmodel",
)
wsclean_container = FlintContainer(
    name="wsclean",
    filename="flint-containers_wsclean.sif",
    uri="docker://alecthomson/flint-containers:wsclean",
    description="Container with the wsclean deconvolution software",
)
askapsoft_contaer = FlintContainer(
    name="askapsoft",
    filename="flint-containers_askapsoft.sif",
    uri="docker://alecthomson/flint-containers:askapsoft",
    description="Container with askapsoft (also known as yandasoft)",
)
aoflagger_contaer = FlintContainer(
    name="aoflagger",
    filename="flint-containers_aoflagger.sif",
    uri="docker://alecthomson/flint-containers:aoflagger",
    description="Container with aoflagger, used to autonomously flag measurement sets",
)
aegean_contaer = FlintContainer(
    name="aegean",
    filename="flint-containers_aegean.sif",
    uri="docker://ßßalecthomson/flint-containers:aegean",
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

    for idx, known_container in enumerate(LIST_OF_KNOWN_CONTAINERS):
        logger.info(f"Container {idx + 1} of {len(LIST_OF_KNOWN_CONTAINERS)}")
        logger.info(f"\tName: {known_container.name}")
        logger.info(f"\tFilename: {known_container.filename}")
        logger.info(f"\tURL: {known_container.url}")
        logger.info(f"\tDescription: {known_container.description}")


def download_known_containers(container_directory: Path | str) -> tuple[Path, ...]:
    container_directory = Path(container_directory)

    containers_downloaded = []
    for idx, known_catalogue in enumerate(LIST_OF_KNOWN_CONTAINERS):
        logger.info(f"{idx + 1} of {len(LIST_OF_KNOWN_CONTAINERS)}")

        expected_output_path = container_directory / known_catalogue.filename

        if expected_output_path.exists():
            logger.info(f"{expected_output_path=} already exists. Skipping.")
            continue

        _container_path = pull_container(
            container_directory=container_directory,
            uri=known_catalogue.uri,
            filename=known_catalogue.filename,
        )
        if not expected_output_path.exists():
            logger.error(
                f"{expected_output_path=} but was not. Instead received {_container_path=}"
            )

        containers_downloaded.append(expected_output_path)

    logger.info(f"Downloaded {len(containers_downloaded)}")
    return tuple(containers_downloaded)


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

    download_parser = subparsers.add_parser(
        name="download", help="Pull each of the known containers"
    )

    download_parser.add_argument(
        "container_directory", type=Path, help="Location to download containers to"
    )

    return parser


def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()

    if args.mode == "list":
        log_known_containers()
    elif args.mode == "download":
        download_known_containers(container_directory=args.containers_directory)
    else:
        logger.info(f"Unknown directive: {args.mode}")


if __name__ == "__main__":
    cli()
