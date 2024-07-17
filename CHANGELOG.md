# Change log

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
