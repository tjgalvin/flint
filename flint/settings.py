"""This is a container that will attempt to centralise the 
set of flint processing related options. 
"""

from pathlib import Path
from typing import NamedTuple, Optional


class Settings(NamedTuple):
    """Container that represents the flint related options that
    might be used throughout components.
    """

    flagger_container: Path
    """Path to the singularity aoflagger container"""
    calibrate_container: Path
    """Path to the singularity calibrate container"""
    expected_ms: int = 36
    """The expected number of measurement set files to find"""
    wsclean_container: Optional[Path] = None
    """Path to the singularity wsclean container"""
    yandasoft_container: Optional[Path] = None
    """Path to the singularity yandasoft container"""
    holofile: Optional[Path] = None
    """Path to the holography FITS cube that will be used when co-adding beams"""
    rounds: int = 2
    """Number of required rouds of self-calibration to perform"""
    zip_ms: bool = False
    """Whether to zip measurement sets once they are no longer required"""
    run_aegean: bool = False
    """Whether to run the aegean source finding tool"""
    aegean_container: Optional[Path] = None
    """Path to the singularity aegean container"""
    no_imaging: bool = False
    """Whether to skip the imaging process (including self-calibration)"""
    reference_catalogue_directory: Optional[Path] = None
    """Path to the directory container the refernce catalogues, used to generate valiation plots"""
    butterworth_filter: bool = False
    """Whether a Butterworth filter should be used when constructing the clean mask"""
