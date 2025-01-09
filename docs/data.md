# Data

## Sky-model catalogues

The `flint_skymodel` command will attempt to create an in-field sky-model for a
particular measurement set using existing source catalogues and an idealised
primary beam response. Supported catalogues are those available through
`flint_catalogue download`. Note this mode has not be thoroughly tested and may
not be out-of-date relative to how the `flint_flow_continuum_pipeline` operates.
In the near future this may be expanded.

If calibrating a bandpass (i.e. `1934-638`) `flint` will use the packaged source
model. At the moment this is only provided for `calibrate`.

## About ASKAP Measurement Sets

Some of the innovative components of ASKAP and the `yandasoft` package have
resulted in measurement sets that are not immediately inline with external
tools. Measurement sets should first be processed with
[fixms](https://github.com/AlecThomson/FixMS). Be careful -- most (all) `flint`
tasks don't currently do this automatically. Be aware, me hearty.

(containers)=

## Containers

At the moment this pipeline uses `singularity` containers to use compiled
software that are outside the `python` ecosystem.

:::{attention} For the moment there are no 'supported' container packaged within
this repository -- sorry! :::

In a nutshell, the containers used throughout are passed in as command line
arguments, whose context should be enough to explain what it is expecting. At
the time of writing there are six containers for:

- calibration: this should contain `calibrate` and `applysolutions`. These are
  tools written by Andre Offringa.
- flagging: this should contain `aoflagger`, which is installable via a
  `apt install aoflagger` within ubuntu.
- imaging: this should contain `wsclean`. This should be at least version 3. At
  the moment a modified version is being used (which implements a
  `-force-mask-round` option).
- source finding: `aegeam` is used for basic component catalogue creation. It is
  not intedended to be used to produce final source catalogues, but to help
  construct quick-look data products. A minimal set of `BANE` and `aegean`
  options are used.
- source peeling: `potatopeel` is a package that uses `wsclean`, `casa` and a
  customisable rule set to peel out troublesome annoying objects. Although it is
  a python installable and importable package, there are potential conflicts
  with the `python-casacore` modules that `flint` uses. See
  [potatopeel's github repository for more information](https://gitlab.com/Sunmish/potato/-/tree/main)
- linear mosaicing: The `linmos` task from `yandasoft` is used to perform linear
  mosaicing. Importanting this `linmos` is capable of using the ASKAP primary
  beam responses characterised through holography. `yandasoft` docker images
  [are available from the CSIRO dockerhub page.](https://hub.docker.com/r/csirocass/askapsoft).
- self-calibration: `casa` is used to perform antenna-based self-calibration.
  Specifically the tasks `gaincal`, `applysolutions`, `cvel` and `mstransform`
  are used throughout this process. Careful selection of an appropriate CASA
  version should be made to keep the `casacore` library in compatible state with
  other components. Try the `docker://alecthomson/casa:ks9-5.8.0` image.
