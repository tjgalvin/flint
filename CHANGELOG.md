# Change log

# dev

- added `wrapper_options_from_strategy` decorator helper function

# 0.2.7

- added in convolving of cubes to common resolution across channels
- cubes are supported when computing the yandasoft linmos weights and trimming
- `--coadd-cubes` option added to co-add cubes on the final imaging round
  together to form a single field spectral cube
- Cleaning up the `flint_masking` CLI:
  - added more options
  - removed references to butterworth filter
  - marked `minimum_boxcar)artefact_mask` as deprecated and to be removed
- Added initial `beam_shape_erode` to masking operations

# 0.2.6

- if `-temp-dir` used in wsclean then imaging products are produced here and
  then copied over to the same directory as the MS. This is intended to make use
  of compute nodes and fast local storage, like memory tmpfs or local disks.
- added stokes-v imaging. This includes a couple of minor / largish changes
  - the `name` attribute of `WSCleanOptions` is not recognised
  - the `pol` attribute of `WSCleanOptions` now needs to be a `str`
  - the `-name` CLI argument of `wsclean` is auto-generated and always provided,
    and will now always contain the `pol` values (i.e. has a `polVALUE` field in
    output filenames)
  - the strategy format now has a `operations` set of keywords, including
    `stokesv` to drawn options from
  - naming format of output linmos files could contain the pol field
  - `stokesv` imaging will not linmos the cleaning residuals together, even if
    the `--linmos-residuals` CLI is provided
- Capture `CleanDivergenceError` from `wsclean` and rerun with larger image size
  and lower gain
- Added `flint.catalogue`, which aims to collect all the catalogue related
  operations
  - a `flint_catalogue` CLI program to:
    - download reference catalogues that are known and expected from vizier
    - verify reference catalogues conform to expectations
    - list the reference catalogues that are expected
- Added `flint.leakage` CLI program to attempt to characterise leakage over
  polarisations, e.g. V/I
- removing the `pol` string in the polarisation field of the
  `ProcessedNameComponents`
  - `wclean` output `-` separation character chhanged to `.`
- The median RMS of the field image is calculated on an inner region when
  cnnstructing the validation plot
- Limit the number of sources to overlay on the RMS plot in the quick look
  validation figure
- Replace the `cassatasks` with a casa in a container
  - Added `flint.sclient.singularity_wrapper` decorator to help make running
    commands in a container easier
  - Removed `casatasks` and `casadata` from dependencies
- Moving to a minimum python=3.12 version
  - Partly enabled by removing `casatasks` and `casadata`, partly required by
    other dependencies that changed
  - Notably the `numpy.distutils` started to complain

# 0.2.5

- added in skip rounds for masking and selfcal
- Basic handling of CASDA measurement sets (preprocessing)
- Basic handling of environment variables in Options, only supported in
  WSCleanOptions (no need for others yet)
- basic renaming of MS and shuffling column names in place of straight up
  copying the MS
- added CLI argument `--fixed-beam-shape` to specify a fixed final resolution,
  overwritng the optimal beam shape that otherwise would be computed
- Added a SlurmInfo class to help with debugging crashed jobs. Primitive and
  likely to change.
- Made the `calibrated_bandpass_path` and optional CLI argument so that CASDA
  MSs can be better handled
- copying folders in `copy_files_into` when archiving
- added reading of ArchiveOptions from strategy file for continuum pipeline and
  `flint_archive`
- Adaptive colour bar scaling in the rms validation plot
- Create multiple linmos images if `--fixed-beam-shape` specified, one at an
  optimal common resolution and another at the specified resolution
- Dump the `FieldOptions` to the output science directory
- Weights produced by `linmos` are also trimmed in the same way as the
  corresponding image

## 0.2.4

- Added new masking modes
- More Option types available in the template configuration
- Tarball of files and linmos fields
- Added context managers to help with MEMDIR / LOCALDIR like variables

## Pre 0.2.3

Everything chsnges all the time
