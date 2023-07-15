"""Simple interface into wsclean
"""
from __future__ import annotations
from pathlib import Path
from typing import NamedTuple, Collection, Union, List, Tuple, Any
from numbers import Number
from argparse import ArgumentParser

from flint.logging import logger 
from flint.ms import MS 
from flint.sclient import run_singularity_command

class WSCleanOptions(NamedTuple):
    """A basic container to handle WSClean options. These attributes should
    conform to the same option name in the calling signature of wsclean 
    """
    absmem: int = 100
    """Memory wsclean should try to limit itself to"""
    psfwindow: int = 65
    """Size of the window used to estimate rms noise"""
    size: int = 7000
    """Image size"""
    forcemask: float = 10
    """Round of force masked derivation"""
    maskthresh: float = 5
    """How deep the construct clean mask is during each cycle"""
    autothresh: float = 0.5
    """How deep to clean once initial clean threshold reached"""
    channels_out: int = 8
    """Number of output channels"""
    mgain: float = 0.7
    """Major cycle gain"""
    nmiter: int = 15
    """Maximum number of major cycles to perform"""
    niter: int = 50000
    """Maximum numer of minor cycles"""
    multiscale: bool = False
    """Enable multiscale deconvolution"""
    multiscale_scale_bias: float = 0.9
    """Multiscale bias term"""
    fit_spectral_pol: int = 4
    """Number of spectral terms to include during sub-band subtractin"""
    robust: float = 0.5
    """Robustness of the weighting used"""
    data_column: str = 'CORRECTED_DATA'
    """Which column in the MS to image"""
    def with_options(self, **kwargs) -> WSCleanOptions:
        _dict = self._asdict()
        _dict.update(**kwargs)
        
        return WSCleanOptions(**_dict)
    
class WSCleanCMD(NamedTuple):
    cmd: str 
    ms: Union[MS,Collection[MS]]

def create_wsclean_cmd(ms: MS, wsclean_options: WSCleanOptions) -> WSCleanCMD:
    
    cmd = 'wsclean '
    unknowns: List[Tuple[Any,Any]] = []
    logger.info("Creating wsclean command.")
    for key, value in wsclean_options._asdict().items():
        key = key.replace('_', '-')
        logger.debug(f"{key=} {value=} {type(value)=}")
        if isinstance(value, bool):
            cmd += f'-{key} '
        if isinstance(value, (str, Number)):
            cmd += f'-{key} {value} '
        else:
            unknowns.append((key, value))
    
    if len(unknowns) > 0:
        msg = ', '.join([f"{t[0]} {t[1]}" for t in unknowns])
        raise ValueError(f"Unknown wsclean option types: {msg}")
    
    cmd += f"{str(ms.path)} "
    
    logger.info(f"Constructed wsclean command: {cmd=}")
    
    return WSCleanCMD(cmd=cmd, ms=ms)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Routines related to wsclean")

    subparser = parser.add_subparsers(dest='mode')
    
    wsclean_parser = subparser.add_parser('image', help='Attempt to run a wsclean commmand. ')
    wsclean_parser.add_argument('ms', type=Path, help='Path to a measurement set to image')
    wsclean_parser.add_argument('-v','--verbose', action='store_true', help='Extra output logging.')
    
    return parser 
    

def cli() -> None:
    parser = get_parser()
    
    args = parser.parse_args()
    
    if args.mode == 'image':
        if args.verbose:
            import logging 
            logger.setLevel(logging.DEBUG)
            
        ms = MS(path=args.ms)
        wsclean_options = WSCleanOptions()
        
        create_wsclean_cmd(ms=ms, wsclean_options=wsclean_options)

if __name__ == '__main__':
    cli()
    