"""Common prefect tasks around interacting with measurement sets"""

from pathlib import Path
from typing import Optional

from prefect import task

from flint.calibrate.aocalibrate import AddModelOptions, add_model
from flint.logging import logger
from flint.imager.wsclean import WSCleanCommand


# TODO: This can be a dispatcher type function should
# other modes be added
def add_model_source_list_to_ms(
    wsclean_command: WSCleanCommand, calibrate_container: Optional[Path] = None
) -> WSCleanCommand:
    logger.info("Updating MODEL_DATA with source list")
    ms = wsclean_command.ms

    assert (
        wsclean_command.imageset is not None
    ), f"{wsclean_command.imageset=}, which is not allowed"

    source_list_path = wsclean_command.imageset.source_list
    if source_list_path is None:
        logger.info(f"{source_list_path=}, so not updating")
        return wsclean_command
    assert source_list_path.exists(), f"{source_list_path=} does not exist"

    if calibrate_container is None:
        logger.info(f"{calibrate_container=}, so not updating")
        return wsclean_command
    assert calibrate_container.exists(), f"{calibrate_container=} does not exist"

    add_model_options = AddModelOptions(
        model_path=source_list_path,
        ms_path=ms.path,
        mode="c",
        datacolumn="MODEL_DATA",
    )
    add_model(
        add_model_options=add_model_options,
        container=calibrate_container,
        remove_datacolumn=True,
    )
    return wsclean_command


task_add_model_source_list_to_ms = task(add_model_source_list_to_ms)
