"""Collection of functions and tooling intended 
for general usage. 
"""

import subprocess
from pathlib import Path

from flint.logging import logger 

def rsync_copy_directory(
    target_path: Path, out_path: Path 
) -> Path:
    """A small attempt to rsync a directtory from one location to another.
    This is an attempt to verify a copy was completed successfully. 

    Args:
        target_path (Path): The target directory to copy
        out_path (Path): The location to copy the directory to

    Returns:
        Path: The output path of the new directory. 
    """

    rsync_cmd = (
        f"rsync -avh --progress --stats "
        f"{str(target_path)}/ "
        f"{str(out_path)}/ "
    )
    
    logger.info(f"Will run {rsync_cmd}")
    rsync_run = subprocess.Popen(
        rsync_cmd.split(), stdout=subprocess.PIPE
    )
    
    if rsync_run.stdout is not None:
        for line in rsync_run.stdout:
            logger.info(line.decode().rstrip())
    
    return out_path