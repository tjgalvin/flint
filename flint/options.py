"""This is a container that will attempt to centralise the
set of flint processing related options.
"""

from __future__ import (  # Used for mypy/pylance to like the return type of MS.with_options
    annotations,
)

from pathlib import Path
from typing import Collection, List, NamedTuple, Optional, Union


class BandpassOptions(NamedTuple):
    """Container that reoresents the flint related options that
    might be used throughout the processing of bandpass calibration
    data.

    In its present form this `BandpassOptions` class is not intended
    to contain properties of the data that arebeing processed, rather
    how these data will be processed.

    These settings are not meant to be adjustabled throughout
    a single bandpass pipeline run
    """

    flagger_container: Path
    """Path to the singularity aoflagger container"""
    calibrate_container: Path
    """Path to the singularity calibrate container"""
    expected_ms: int = 36
    """The expected number of measurement set files to find"""
    smooth_solutions: bool = False
    """Will activate the smoothing of the bandpass solutions"""
    smooth_window_size: int = 16
    """The width of the smoothing window used to smooth the bandpass solutions"""
    smooth_polynomial_order: int = 4
    """The polynomial order used by the Savgol filter when smoothing the bandpass solutions"""
    flag_calibrate_rounds: int = 3
    """The number of times the bandpass will be calibrated, flagged, then recalibrated"""
    minuv: Optional[float] = None
    """The minimum baseline length, in meters, for data to be included in bandpass calibration stage"""
    preflagger_ant_mean_tolerance: float = 0.2
    """Tolerance that the mean x/y antenna gain ratio test before the antenna is flagged"""
    preflagger_mesh_ant_flags: bool = False
    """Share channel flags from bandpass solutions between all antenna"""
    preflagger_jones_max_amplitude: Optional[float] = None
    """Flag Jones matrix if any amplitudes with a Jones are above this value"""


class FieldOptions(NamedTuple):
    """Container that represents the flint related options that
    might be used throughout components related to the actual
    pipeline.

    In its present form this `FieldOptions` class is not intended
    to contain properties of the data that are being processed,
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
    potato_container: Optional[Path] = None
    """Path to the singularity potato peel container"""
    holofile: Optional[Path] = None
    """Path to the holography FITS cube that will be used when co-adding beams"""
    rounds: int = 2
    """Number of required rouds of self-calibration and imaging to perform"""
    skip_selfcal_on_rounds: Optional[List[int]] = None
    """Do not perform the derive and apply self-calibration solutions on these rounds"""
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
    linmos_residuals: bool = False
    """Linmos the cleaning residuals together into a field image"""
    beam_cutoff: float = 150
    """Cutoff in arcseconds to use when calculating the common beam to convol to"""
    fixed_beam_shape: Optional[List[float]] = None
    """Specify the final beamsize of linmos field images in (arcsec, arcsec, deg)"""
    pb_cutoff: float = 0.1
    """Primary beam attentuation cutoff to use during linmos"""
    use_preflagger: bool = True
    """Whether to apply (or search for solutions with) bandpass solutions that have gone through the preflagging operations"""
    use_smoothed: bool = False
    """Whether to apply (or search for solutions with) a bandpass smoothing operation applied"""
    use_beam_masks: bool = True
    """Construct beam masks from MFS images to use for the next round of imaging. """
    use_beam_mask_rounds: Union[str, List[int], int] = 1
    """If `use_beam_masks` is True, this sets which rounds should have a mask applied"""
    imaging_strategy: Optional[Path] = None
    """Path to a FLINT imaging yaml file that contains settings to use throughout imaging"""
    sbid_archive_path: Optional[Path] = None
    """Path that SBID archive tarballs will be created under. If None no archive tarballs are created. See ArchiveOptions. """
    sbid_copy_path: Optional[Path] = None
    """Path that final processed products will be copied into. If None no copying of file products is performed. See ArchiveOptions. """
    rename_ms: bool = False
    """Rename MSs throught rounds of imaging and self-cal instead of creating copies. This will delete data-columns throughout. """


# TODO: Perhaps move these to flint.naming, and can be built up
# based on rules, e.g. imager used, source finder etc.
DEFAULT_TAR_RE_PATTERNS = (
    r".*MFS.*image\.fits",
    r".*linmos.*",
    r".*weight\.fits",
    r".*yaml",
    r".*\.txt",
    r".*png",
    r".*beam[0-9]+\.ms\.zip",
    r".*beam[0-9]+\.ms",
    r".*\.caltable",
    r".*\.tar",
    r".*\.csv",
)
DEFAULT_COPY_RE_PATTERNS = (r".*linmos.*fits", r".*weight\.fits", r".*png", r".*csv")


class ArchiveOptions(NamedTuple):
    """Container for options related to archiving products from flint workflows"""

    tar_file_re_patterns: Collection[str] = DEFAULT_TAR_RE_PATTERNS
    """Regular-expressions to use to collect files that should be tarballed"""
    copy_file_re_patterns: Collection[str] = DEFAULT_COPY_RE_PATTERNS
    """Regular-expressions used to identify files to copy into a final location (not tarred)"""

    def with_options(self, **kwargs) -> ArchiveOptions:
        opts = self._asdict()
        opts.update(**kwargs)

        return ArchiveOptions(**opts)
