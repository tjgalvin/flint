"""Utilities related to running commands in a singularity container"""

from pathlib import Path
from socket import gethostname
from subprocess import CalledProcessError
from typing import Collection, Optional, Union

from spython.main import Client as sclient

from flint.logging import logger


def run_singularity_command(
    image: Path, command: str, bind_dirs: Optional[Union[Path, Collection[Path]]] = None
) -> None:
    """Executes a command within the context of a nominated singularity
    container

    Args:
        image (Path): The singularity container image to use
        command (str): The command to execute
        bind_dirs (Optional[Union[Path,Collection[Path]]], optional): Specifies a Path, or list of Paths, to bind to in the container. Defaults to None.

    Raises:
        FileNotFoundError: Thrown when container image not found
        CalledProcessError: Thrown when the command into the container was not successful
    """

    if not image.exists():
        raise FileNotFoundError(f"The singularity container {image} was not found. ")

    logger.info(f"Running {command} in {image}")
    logger.info(f"Attempting to run singularity command on {gethostname()}")

    bind_str = None
    if bind_dirs:
        if isinstance(bind_dirs, Path):
            bind_dirs = [bind_dirs]

        # Get only unique paths to avoid duplication in bindstring.
        # bind_str = ",".join(set([str(p.absolute().resolve()) for p in bind_dirs]))
        bind = (
            list(set([str(p.absolute().resolve()) for p in bind_dirs]))
            if len(bind_dirs) > 0
            else None
        )

        logger.debug(f"Constructed singularity bindings: {bind_str}")

    try:
        output = sclient.execute(
            image=image.resolve(strict=True).as_posix(),
            command=command.split(),
            bind=bind,
            return_result=True,
            quiet=False,
            stream=True,
            stream_type="both",
        )

        for line in output:
            logger.info(line.rstrip())
    except CalledProcessError as e:
        logger.error(f"Failed to run command: {command}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        logger.error(f"{e=}")

        raise e
