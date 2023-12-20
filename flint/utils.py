"""Collection of functions and tooling intended
for general usage.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from flint.logging import logger


def create_cd_into_directory(in_folder: Path) -> Path:
    """Create a directory if it does not exist, and then
    change the current workign directory into it. If the
    input directory does not exist, then it and its parents
    will be created.

    Internally there is no handling of potential race conditions
    should this function be called by multiple workers.

    Args:
        in_folder (Path): The directory to create and change into

    Returns:
        Path: The input directory after it has been created and changed into
    """
    in_folder = Path(in_folder)

    if in_folder.exists():
        logger.info(f"{in_folder} already exists. ")
    else:
        logger.info(f"Creating {in_folder}")
        in_folder.mkdir(parents=True)

    logger.info(f"Chaging working directory to {in_folder}")
    os.chdir(path=in_folder)

    return in_folder


def zip_folder(in_path: Path, out_zip: Optional[Path] = None) -> Path:
    """Zip a directory and remove the original.

    Args:
        in_path (Path): The path that will be zipped up.
        out_zip (Path, optional): Name of the output file. A zip extension will be added. Defaults to None.

    Returns:
        Path: the path of the compressed zipped folder
    """

    out_zip = in_path if out_zip is None else out_zip

    logger.info(f"Zipping {in_path}.")
    shutil.make_archive(str(out_zip), "zip", base_dir=str(in_path))
    remove_files_folders(in_path)

    return out_zip


def rsync_copy_directory(target_path: Path, out_path: Path) -> Path:
    """A small attempt to rsync a directtory from one location to another.
    This is an attempt to verify a copy was completed successfully.

    Args:
        target_path (Path): The target directory to copy
        out_path (Path): The location to copy the directory to

    Returns:
        Path: The output path of the new directory.
    """

    rsync_cmd = (
        f"rsync -avh --progress --stats " f"{str(target_path)}/ " f"{str(out_path)}/ "
    )
    logger.info(f"Rsync copying {target_path} to {out_path}.")
    logger.debug(f"Will run {rsync_cmd}")
    rsync_run = subprocess.Popen(rsync_cmd.split(), stdout=subprocess.PIPE)

    if rsync_run.stdout is not None:
        for line in rsync_run.stdout:
            logger.debug(line.decode().rstrip())

    return out_path


def remove_files_folders(*paths_to_remove: Path) -> List[Path]:
    """Will remove a set of paths from the file system. If a Path points
    to a folder, it will be recursively removed. Otherwise it is simply
    unlinked.

    Args:
        paths_to_remove (Path): Set of Paths that will be removed

    Returns:
        List[Path]: Set of Paths that were removed
    """

    files_removed = []

    file: Path
    for file in paths_to_remove:
        file = Path(file)
        if not file.exists():
            logger.debug(f"{file} does not exist. Skipping, ")
            continue

        if file.is_dir():
            logger.info(f"Removing folder {str(file)}")
            shutil.rmtree(file)
        else:
            logger.info(f"Removing file {file}.")
            file.unlink()

        files_removed.append(file)

    return files_removed
