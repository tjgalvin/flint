"""Collection of functions and tooling intended 
for general usage. 
"""

import shutil
import subprocess
from typing import List
from pathlib import Path

from flint.logging import logger


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