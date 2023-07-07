"""Code to use AO calibrate s
"""
from pathlib import Path
from typing import NamedTuple, Optional
from argparse import ArgumentParser

from spython.main import Client as sclient

from flint.logging import logger
from flint.ms import MS
from flint.sclient import run_singularity_command

class CalibrateCommand(NamedTuple):
    """The AO Calibrate command and output path of the corresponding solutions file
    """
    cmd: str
    """The calibrate command that will be executed
    """
    solution_path: Path
    """The output path of the solutions file
    """
    ms: Optional[MS] = None
    """Optionally may attach a measurement set to the solutions. 
    """

class ApplySolutionsCommand(NamedTuple):
    """The applysolutions command to execute"""
    cmd: str 
    """The command that will be executed"""
    solution_path: Path 
    """Location of the solutions file to apply"""
    ms: MS
    """The measurement set that will have the solutions applied to"""

CALIBRATE_SUFFIX="calibrate.bin"

def create_calibrate_cmd(
    ms: MS, 
    calibrate_model: Path,
    solution_path: Optional[Path] = None,
    **kwargs
) -> CalibrateCommand:
    """Generate a typical ao calibrate command. Any extra keyword arguments
    are passed through as additional options to the `calibrate` program. 

    Args:
        ms (MS): The measurement set to calibrate. There needs to be a nominated data_column. 
        calibrate_model (Path): Path to a generated calibrate sky-model
        solution_path (Path, optional): The output path of the calibrate solutions file. 
        If None, a default suffix of "calibrate.bin" is used. Defaults to None. 

    Raises:
        FileNotFoundError: Raised when calibrate_model can not be found. 

    Returns:
        CalibrateCommand: The calibrate command to execute and output solution file
    """
    logger.info(f"Creating calibrate command for {ms.path}")

    # This is a typical calibrate command. 
    # calibrate -minuv 100 -i 50 -datacolumn DATA
    #        -m 2022-04-14_100122_0.calibrate.txt  
    #        2022-04-14_100122_0.ms 2022-04-14_100122_0.aocalibrate.bin

    assert ms.column is not None, f"{ms} does not have a nominated data_column"
    
    if not calibrate_model.exists():
        raise FileNotFoundError(f"Calibrate model {calibrate_model} not found. ")

    solution_path = ms.path.with_suffix(CALIBRATE_SUFFIX) if solution_path is None else solution_path

    calibrate_kwargs: str = " ".join([f"-{key} {item}" for key, item in kwargs.items()])

    cmd = (
        f"calibrate "
        f"-datacolumn {ms.column} "
        f"-m {str(calibrate_model)} "
        f"{calibrate_kwargs} "
        f"{str(ms.path)} "
        f"{str(solution_path)} "
    ) 

    logger.debug(f"Constructed calibrate command is {cmd=}")
    
    return CalibrateCommand(
        cmd=cmd, solution_path=solution_path, ms=ms
    )

def create_apply_solutions_cmd(
    ms: MS, solutions_file: Path, output_column: Optional[str]=None
) -> ApplySolutionsCommand:
    """Construct the command to apply calibration solutions to a MS
    using an AO calibrate style solutions file. 
    
    The `applysolutions` program does not appear to have the ability to set
    a desured output column name. If the `output_column` specified matches
    the nominated column in `ms`, then `applysolutions` will simply overwrite
    the column with updated data. Otherwise, a `CORRECTED_DATA` column is produced. 
    
    NOTE: Care to be taken when the nominated column is `CORRECTED_DATA`. 

    Args:
        ms (MS): Measurement set to have solutions applied to
        solutions_file (Path): Path to the solutions file to apply
        output_column (Optional[str], optional): The desired output column name. See notes above. Defaults to None.

    Returns:
        ApplySolutionsCommand: _description_
    """
    
    assert ms.path.exists(), f"The measurement set {ms} was not found. "
    assert ms.column is not None, f"{ms} does not have a nominated data_column. "
    assert solutions_file.exists(), f"The solutions file {solutions_file} does not exists. "
    
    input_column = ms.column
    copy_mode = '-nocopy' if input_column == output_column else '-copy'
    
    logger.info(f"Setting {copy_mode=}.")
    
    if copy_mode == '-copy':
        output_column = 'CORRECT_DATA'
        
    cmd = (
        f"applysolutions "
        f"-datacolumn {input_column} "
        f"{copy_mode} "
        f"{str(ms.path)} "
        f"{str(solutions_file)} "
    )

    logger.info(f"Constructed {cmd=}")

    return ApplySolutionsCommand(
        cmd=cmd, solution_path=solutions_file, ms=ms
    )

def run_calibrate(
    calibrate_cmd: CalibrateCommand, container: Path
) -> None:
    """Execute a calibrate command within a singularity container

    Args:
        calibrate_cmd (CalibrateCommand): The constructed calibrate command
        container (Path): Location of the container
    """
    
    assert container.exists(), f"The calibrate container {calibrate_container} does not exist. "
    assert calibrate_cmd.ms is not None, f"When calibrating the 'ms' field attribute must be defined. "
    
    sols_dir_str = str(calibrate_cmd.solution_path.parent)
    ms_dir_str = str(calibrate_cmd.ms.path.parent)
    bind_str = ",".join((sols_dir_str, ms_dir_str))
    logger.debug(f"The bind string is {bind_str=}")
    
    run_singularity_command(
        image=container, command=calibrate_cmd.cmd, bind_str=bind_str
    )
    
def run_apply_solutions(
    apply_solutions_cmd: ApplySolutionsCommand, container: Path    
) -> None:
    """Will execute the applysolutions command inside the specified singularity 
    container. 

    Args:
        apply_solutions_cmd (ApplySolutionsCommand): The constructed applysolutions command
        container (Path): Location of the existing solutions file
    """
    
    assert container.exists(), f"The applysolutions container {apply_solutions_container} does not exist. "
    assert apply_solutions_cmd.ms.path.exists(), f"The measurement set {apply_solutions_cmd.ms} was not found. "
    
    sols_dir_str = str(apply_solutions_cmd.solution_path.parent)
    ms_dir_str = str(apply_solutions_cmd.ms.path.parent)
    bind_str = ",".join((sols_dir_str, ms_dir_str))
    logger.debug(f"The bind string is {bind_str=}")
     
    run_singularity_command(
         image=container, command=apply_solutions_cmd.cmd, bind_str=bind_str
    )


def calibrate_apply_ms(
    ms_path: Path, model_path: Path, container: Path, data_column: str='DATA'    
) -> MS:
    
    ms = MS(path=ms_path, column=data_column)
    
    logger.info(f"Will be attempting to calibrate {ms}")
    
    calibrate_cmd = create_calibrate_cmd(
        ms=ms, calibrate_model=model_path
    )

    run_calibrate(
        calibrate_cmd=calibrate_cmd, container=container
    )
    
def get_parser() -> ArgumentParser:
    
    parser = ArgumentParser(
        description="Run calibrate and apply the solutions given a measurement set and sky-model."
    )
    
    parser.add_argument('ms', type=Path, help="The measurement set to calibrate and apply solutions to. ")
    parser.add_argument('aoskymodel', type=Path, help="The AO-style sky-model file to use when calibrating. ")
    parser.add_argument('--calibrate-container', type=Path, default="./calibrate.sif", help="The container containing calibrate and applysolutions. ")
    parser.add_argument('--data-column', type=str, default='DATA', help='The column to calibrate')
    
    return parser 
    
def cli() -> None:
    
    parser = get_parser()
    
    args = parser.parse_args()
    
    calibrate_apply_ms(
        ms_path = args.ms,
        model_path=args.aoskymodel,
        container=args.calibrate_container,
        data_column=args.data_column
    )
    
if __name__ == '__main__':
    cli()
     
    