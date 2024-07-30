# Change log

# dev
- if `-temp-dir` used in wsclean then imaging products are produced here and then copied over to the same directory as the MS. This is intended to make use of compute nodes and fast local storage, like memory tmpfs or local disks.
- added stokes-v imaging. This includes a couple of minor / largish changes
    - the `name` attribute of `WSCleanOptions` is not recognised
    - the `pol` attribute of `WSCleanOptions` now needs to be a `str`
    - the `-name` CLI argument of `wsclean` is auto-generated and always provided, and will now always contain the `pol` values (i.e. has a `polVALUE` field in output filenames)
    - the strategy format now has a `operations` set of keywords, including `stokesv` to drawn options from
    - naming format of output linmos files could contain the pol field
    - `stokesv` imaging will not linmos the cleaning residuals together, even if the `--linmos-residuals` CLI is provided
- Capture `CleanDivergenceError` from `wsclean` and rerun with larger image size and lower gain

# 0.2.5
- added in skip rounds for masking and selfcal
- Basic handling of CASDA measurement sets (preprocessing)
- Basic handling of environment variables in Options, only supported in WSCleanOptions (no need for others yet)
- basic renaming of MS and shuffling column names in place of straight up copying the MS
- added CLI argument `--fixed-beam-shape` to specify a fixed final resolution, overwritng the optimal beam shape that otherwise would be computed
- Added a SlurmInfo class to help with debugging crashed jobs. Primative and likely to change.
- Made the `calibrated_bandpass_path` and optional CLI argument so that CASDA MSs can be better handled
- copying folders in `copy_files_into` when archiving
- added reading of ArchiveOptions from strategy file for continuum pipeline and `flint_archive`
- Adaptive colour bar scaling in the rms validation plot
- Create multiple linmos images if `--fixed-beam-shape` specified, one at an optimal common resolution and another at the specified resolution
- Dump the `FieldOptions` to the output science directory
- Weights produced by `linmos` are also trimmed in the same way as the corresponding image

## 0.2.4

- Added new masking modes
- More Option types available in the template configuration
- Tarball of files and linmos fields
- Added context managers to help with MEMDIR / LOCALDIR like variables

## Pre 0.2.3

Everything chsnges all the time
