"""Utilities related to running commands in a singularity container"""

from pathlib import Path
from subprocess import CalledProcessError
from time import sleep
from typing import Callable, Collection, Optional, Union

from spython.main import Client as sclient

from flint.logging import logger
from flint.utils import log_job_environment


def run_singularity_command(
    image: Path,
    command: str,
    bind_dirs: Optional[Union[Path, Collection[Path]]] = None,
    stream_callback_func: Optional[Callable] = None,
) -> None:
    """Executes a command within the context of a nominated singularity
    container

    Args:
        image (Path): The singularity container image to use
        command (str): The command to execute
        bind_dirs (Optional[Union[Path,Collection[Path]]], optional): Specifies a Path, or list of Paths, to bind to in the container. Defaults to None.
        stream_callback_func (Optional[Callable], optional): Provide a function that is applied to each line of output text when singularity is running and `stream=True`. IF provide it should accept a single (string) parameter. If None, nothing happens. Defaultds to None.

    Raises:
        FileNotFoundError: Thrown when container image not found
        CalledProcessError: Thrown when the command into the container was not successful
    """

    if not image.exists():
        raise FileNotFoundError(f"The singularity container {image} was not found. ")

    logger.info(f"Running {command} in {image}")

    job_info = log_job_environment()

    if bind_dirs:
        if isinstance(bind_dirs, Path):
            bind_dirs = [bind_dirs]

        # Get only unique paths to avoid duplication in bindstring.
        # bind_str = ",".join(set([str(p.absolute().resolve()) for p in bind_dirs]))
        bind = (
            list(set([str(Path(p).absolute().resolve()) for p in bind_dirs]))
            if len(bind_dirs) > 0
            else None
        )

        logger.debug(f"Constructed singularity bindings: {bind}")

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
            if stream_callback_func:
                stream_callback_func(line)

        # Sleep for a few moments. If the command created files (often they do), give the lustre a moment
        # to properly register them. You dirty sea dog.
        sleep(2.0)
    except CalledProcessError as e:
        logger.error(f"Failed to run command: {command}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        logger.error(f"{e=}")
        logger.error(f"{job_info}")

        raise e
