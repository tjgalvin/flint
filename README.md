[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/tjgalvin/flint/main.svg)](https://results.pre-commit.ci/latest/github/tjgalvin/flint/main)
[![codecov](https://codecov.io/github/tjgalvin/flint/graph/badge.svg?token=7ZEKJ78TBZ)](https://codecov.io/github/tjgalvin/flint)

# flint

<!-- SPHINX-START -->

A pirate themed ASKAP pipeline.

Yarrrr-Harrrr fiddly-dee!

<img src="_static/logo.jpeg" alt="Capn' Flint - Credit: DALLE 3" style="width:400px;"/>

## Installation

Provided an appropriate environment installation should be as simple as a
`pip install`.

However, on some systems there are interactions with `casacore` and building
`python-casacore` appropriately. Issues have been noted when interacting with
large measurement sets across components with different `casacore` versions.
This seems to happen even across container boundaries (i.e. different versions
in containers might play a role). The exact cause is not at all understood, but
it appears to be related to the version of `python-casacore`, `numpy` and
whether pre-built wheels are used.

In practise it might be easier to leverage `conda` to install the appropriate
`boost` and `casacore` libraries.

A helpful script below may be of use.

```bash
BRANCH="main" # replace this with appropriate branch or tag
DIR="flint_${BRANCH}"
PYVERSION="3.12"

mkdir "${DIR}" || exit
cd "${DIR}" || exit


git clone git@github.com:tjgalvin/flint.git && \
        cd flint && \
        git checkout "${BRANCH}"

conda create -y  -n "${DIR}" python="${PYVERSION}" &&  \
        source /home/$(whoami)/.bashrc && \
        conda activate "${DIR}" && \
        conda install -y -c conda-forge boost casacore && \
        PIP_NO_BINARY="python-casacore" pip install -e .
```

This may set up an appropriate environment that is compatible with the
containers currently being used.

## About ASKAP Measurement Sets

Some of the innovative components of ASKAP and the `yandasoft` package have
resulted in measurement sets that are not immediately inline with external
tools. Measurement sets should first be processed with
[fixms](https://github.com/AlecThomson/FixMS). Be careful -- most (all) `flint`
tasks don't currently do this automatically. Be aware, me hearty.

## Containers

At the moment this toy pipeline uses `singularity` containers to use compiled
software that are outside the `python` ecosystem. For the moment there are no
'supported' container packaged with this repository -- sorry!

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

## Configuration based settings

Most settings within `flint` are stored in immutable option classes, e.g.
`WSCleanOptions`, `GainCalOptions`. Once they such an option class has been
created, any new option values may only be set by creating a new instance. In
such cases there is an appropriate `.with_options` method that might be of use.
This 'nothing changes unless explicitly done so' was adopted early as a way to
avoid confusing when moving to a distributed multi-node execution environment.

The added benefit is that it has defined very clear interfaces into key stages
throughout `flint`s calibration and imaging stages. The `flint_config` program
can be used to create template `yaml` file that lists default values of these
option classes that are expected to be user-tweakable, and provides the ability
to change values of options throughout initial imaging and subsequent rounds of
self-calibration.

In a nutshell, the _currently_ supported option classes that may be tweaked
through this template method are:

- `WSCleanOptions` (shorthand `wsclean`)
- `GainCalOptions` (shorthand `gaincal`)
- `MaskingOptions` (shorthand `masking`)
- `ArchiveOptions` (shorthand `archive`)
- `BANEOptions` (shorthand `bane`)
- `AegeanOptions` (shorthand `aegean`)

All attributed supported by these options may be set in this template format.
Not that these options would have to be retrieved within a particular flow and
passed to the appropriate functions - they are not (currently) automatically
accessed.

The `defaults` scope sets all of the default values of these classes. The
`initial` scope overrides the default imaging `wsclean` options to be used with
the first round of imaging _before self-calibration_.

The `selfcal` scope contains a key-value mapping, where an `integer` key relates
the options to that specific round of masking, imaging and calibration options
for that round of self-calibration. Again, options set here override the
corresponding options defined in the `defaults` scope.

`flint_config` can be used to generate a template file, which can then be
tweaked. The template file uses YAML to define scope and settings. So, use the
YAML standard when modifying this file. There are primitive verification
functions to ensure the modified template file is correctly form.

## CLI Configuration file

To help manage (and avoid) long CLI calls to configure `flint`, most command
line options may be dumped into a new-line delimited text file which can then be
set as the `--cli-config` option of some workflows. See the `configargparse`
python utility to read up on more on how options may be overridden if specified
in both the text file and CLI call.

## Validation Plots

The validation plots that are created are simple and aim to provide a quality
assessment at a quick glance. An RMS image and corresponding source component
catalogue are the base data products derived from the ASKAP data that are
supplied to the routine.

`flint` requires a set of reference catalogues to be present for some stages of
operation, the obvious being the validation plots described above. In some
computing environments (e.g. HPC) network access to external services are
blocked. To avoid these issues `flint` has a built in utility to download the
reference catalogues it expected from vizier and write them to a specified user
directory. See:

> `flint_catalogue download --help`

The parent directory that contains these cataloguues should be provided to the
appropriate tasks when appropriate.

In the current `flint` package these catalogues (and their expected columns)
are:

- ICRF

```python
Catalogue(
    survey="ICRF",
    file_name="ICRF.fits",
    freq=1e9,
    ra_col="RAJ2000",
    dec_col="DEJ2000",
    name_col="ICRF",
    flux_col="None",
    maj_col="None",
    min_col="None",
    pa_col="None",
    vizier_id="I/323/icrf2",
)
```

- NVSS

```python
Catalogue(
    survey="NVSS",
    file_name="NVSS.fits",
    name_col="NVSS",
    freq=1.4e9,
    ra_col="RAJ2000",
    dec_col="DEJ2000",
    flux_col="S1.4",
    maj_col="MajAxis",
    min_col="MinAxis",
    pa_col="PA",
    vizier_id="VIII/65/nvss",
)
```

- SUMSS

```python
Catalogue(
    survey="SUMSS",
    file_name="SUMSS.fits",
    freq=8.43e8,
    ra_col="RAJ2000",
    dec_col="DEJ2000",
    name_col="Mosaic",
    flux_col="St",
    maj_col="dMajAxis",
    min_col="dMinAxis",
    pa_col="dPA",
    vizier_id="VIII/81B/sumss212",
)
```

- RACS-LOW

```python
Catalogue(
    file_name="racs-low.fits",
    survey="RACS-LOW",
    freq=887.56e6,
    ra_col="RAJ2000",
    dec_col="DEJ2000",
    name_col="GID",
    flux_col="Ftot",
    maj_col="amaj",
    min_col="bmin",
    pa_col="PA",
    vizier_id="J/other/PASA/38.58/gausscut",
)
```

The known filename is used to find the appropriate catalogue and its full path,
and are appropriately named when using the `flint_catalogue download` tool.

## Contributions

Contributions are welcome! Please do submit a pull-request or issue if you spot
something you would like to address.
