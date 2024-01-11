"""Common prefect related utilities that can be used between flows.
"""
from uuid import UUID
from typing import Any, List, TypeVar
from pathlib import Path
import base64

from prefect import task
from prefect.artifacts import create_markdown_artifact

from flint.logging import logger

T = TypeVar("T")

SUPPORTED_IMAGE_TYPES = ("png",)


def upload_image_as_artifact(image_path: Path) -> UUID:
    """Create and submit a markdown artifact tracked by prefect for an
    input image. Currently supporting png formatted images.

    The input image is converted to a base64 encoding, and embedded directly
    within the markdown string. Therefore, be mindful of the image size as this
    is tracked in the postgres database.

    Args:
        image_path (Path): Path to the image to upload

    Returns:
        UUID: Generated UUID of the registered artifact
    """
    image_type = image_path.suffix.replace(".", "")
    assert image_path.exists(), f"{image_path} does not exist"
    assert (
        image_type in SUPPORTED_IMAGE_TYPES
    ), f"{image_path} has type {image_type}, and is not supported. Supported types are {SUPPORTED_IMAGE_TYPES}"

    with open(image_path, "rb") as open_image:
        logger.info(f"Encoding {image_path} in base64")
        image_base64 = base64.b64encode(open_image.read()).decode()

    logger.info("Creating markdown tag")
    markdown = f"![{image_path.stem}](data:image/{image_type};base64,{image_base64})"

    logger.info("Registering artifact")
    image_uuid = create_markdown_artifact(markdown=markdown)

    return image_uuid


@task
def task_get_attributes(item: Any, attribute: str) -> Any:
    """Retrieve an attribute from an input instance of a class or structure.

    This is intended to be used when dealing with a prefect future object that
    has yet to be evaluated or is otherwise not immediatedly accessible.

    Args:
        item (Any): The item that has the input class or structure
        attribute (str): The attribute to extract

    Returns:
        Any: Vlue of the requested attribute
    """
    logger.debug(f"Pulling {attribute=}")
    return item.__dict__[attribute]


@task
def task_flatten(to_flatten: List[List[T]]) -> List[T]:
    """Will flatten a list of lists into a single list. This
    is useful for when a task-descorated function returns a list.


    Args:
        to_flatten (List[List[T]]): Input list of lists to flatten

    Returns:
        List[T]: Flattened form of input
    """
    logger.debug(f"Received {len(to_flatten)} to flatten.")

    flatten_list = [item for sublist in to_flatten for item in sublist]

    logger.debug(f"Flattened list {len(flatten_list)}")

    return flatten_list
