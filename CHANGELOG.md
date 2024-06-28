# Change log

# Dev
- added in skip rounds for masking and selfcal
- Basic handling of CASDA measurement sets (preprocessing)
- Basic handling of environment variables in Options, only supported in WSCleanOptions (no need for others yet)
- basic renaming of MS and shuffling column names in place of straight up copying the MS
- added CLI argument `--fixed-beam-shape` to specify a fixed final resolution, overwritng the optimal beam shape that otherwise would be computed
- Added a SlurmInfo class to help with debugging crashed jobs. Primative and likely to change.
- Made the `calibrated_bandpass_path` and optional CLI argument so that CASDA MSs can be better handled

## 0.2.4

- Added new masking modes
- More Option types available in the template configuration
- Tarball of files and linmos fields
- Added context managers to help with MEMDIR / LOCALDIR like variables

## Pre 0.2.3

Everything chsnges all the time
