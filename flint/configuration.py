"""Basic utilities to load operational parameters from a yaml-based 
configuration file. The idea being that a configuration file would
be used to specify the options for imaging and self-calibration
throughout the pipeline. 
"""
import yaml
from typing import Optional, Any
from argparse import ArgumentParser
from pathlib import Path 

from flint.imager.wsclean import WSCleanOptions
from flint.selfcal.casa import GainCalOptions
from flint.logging import logger 

def load_yaml(input_yaml: Path) -> Any:
    """Load in a flint based configuration file, which
    will be used to form the strategy for imaging of
    a field. 
    
    The format of the return is likely to change. This
    is not to be relied on for the moment, and should 
    be considered a toy. There will be a mutiny. 

    Args:
        input_yaml (Path): The imaging strategy to use

    Returns:
        Any: The parameters of the imaging and self-calibration to use. 
    """

    logger.info(f"Loading {input_yaml} file. ")
    
    with open(input_yaml, 'r') as in_file:
        input_strategy = yaml.load(in_file, Loader=yaml.Loader)
        
    logger.info(f"Loaded strategy is: ")
    
    init_wsclean = WSCleanOptions(**input_strategy['initial'])
    logger.info(f"The initial wsclean options:\n {init_wsclean}")
    
    if 'selfcal' in input_strategy.keys():
        for selfcal_round, selfcal in enumerate(input_strategy['selfcal']):
            wsclean = WSCleanOptions(**selfcal['imager'])
            casa = GainCalOptions(**selfcal['gaincal'])
    
            logger.info(f"Self-calibration round {selfcal_round}: ")
            logger.info(f"wsclean options: {wsclean}")
            logger.info(f"casa gaincal options: {casa}")
    
    return input_strategy

def create_default_yaml(output_yaml: Path, selfcal_rounds: Optional[int] = None) -> Path:
    """Create an example stategy yaml file that outlines the options to use at varies stages
    of some assumed processing pipeline. 
    
    This is is completely experimental, and expected fields might change. 

    Args:
        output_yaml (Path): Location to write the yaml file to. 
        selfcal_rounds (Optional[int], optional): Will specify the number of self-calibration loops to include the file. If None, there will be none written. Defaults to None.

    Returns:
        Path: Path to the written yaml output file. 
    """
    logger.info(f"Generating a default stategy. ")
    strategy = {}

    initial_wsclean = WSCleanOptions()
    strategy['initial'] = initial_wsclean._asdict()
    
    if selfcal_rounds:
        logger.info(f"Creating {selfcal_rounds} self-calibration rounds. ")
        selfcal = []
        for round in range(1, selfcal_rounds+1):
            selfcal.append(
                {
                    'imager': WSCleanOptions()._asdict(),
                    'gaincal': GainCalOptions()._asdict()
                }
            )
        strategy['selfcal'] = selfcal
    
    with open(output_yaml, "w") as out_file:
        logger.info(f"Writing {output_yaml}.")
        yaml.dump(data=strategy, stream=out_file)
        
    return output_yaml

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Tooling to interact with flint yaml configuration files. ")

    subparser = parser.add_subparsers(dest='mode')
    
    create_parser = subparser.add_parser('create', help='Create an initail yaml file for editing. ')
    create_parser.add_argument('output_yaml', type=Path, default='flint_strategy.yaml', help='The output YAML file to write with default options for various stages. ')
    create_parser.add_argument('--selfcal-rounds', type=int, default=None, help="Number of self-calibration rounds to use. ")

    load_parser = subparser.add_parser("load")
    load_parser.add_argument("input_yaml", type=Path, help="Path to a strategy yaml file to load and inspect. ")

    return parser

def cli() -> None:
    parser = get_parser()

    args = parser.parse_args()
    
    if args.mode == 'create':
        create_default_yaml(
            output_yaml=args.output_yaml,
            selfcal_rounds=args.selfcal_rounds
        )
    elif args.mode == "load":
        load_yaml(input_yaml=args.input_yaml)
    else:
        logger.error(f"{args.mode=} is not set or not known. Check --help. ")

if __name__ == '__main__':
    cli()
