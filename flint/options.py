"""This is a container that will attempt to centralise the
set of flint processing related options.
"""

from pathlib import Path
from typing import NamedTuple, Optional


class FieldOptions(NamedTuple):
    """Container that represents the flint related options that
    might be used throughout components related to the actual
    pipeline.

    In its present form this ``FieldOptions`` class is not intended
    to container properties on the data that is being processed,
    rather how those data will be processed.

    These settins are not meant to be adjustable throughout
    rounds of self-calibration.
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
    linmos_residuals: bool = False
    """Linmos the cleaning residuals together into a field image"""
    beam_cutoff: float = 150
    """Cutoff in arcseconds to use when calculating the common beam to convol to"""
    pb_cutoff: float = 0.1
    """Primary beam attentuation cutoff to use during linmos"""
    use_preflagger: bool = True
    """Whether to apply (or search for solutions with) bandpass solutions that have gone through the preflagging operations"""
    use_smoothed: bool = True
    """Whether to apply (or search for solutions with) a bandpass smoothing operation applied"""
