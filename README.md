# flint

A pirate themed toy ASKAP-RACS pipeline. 

Yarrrr-Harrrr fiddley-dee!

<img src="docs/logo.jpeg" alt="Capn' Flint - Credit: DALLE 3" style="width:400px;"/>

## About

As a toy, this `flint` package is trying to get a minimum start-to-finish workflow written for `RACS` style data. `python` functions are used to do the work, and `prefect` is used to orchestrate their usage into a larger pipeline. 

Most of the `python` routines have a CLI that can be used to test them in a piecewise sense. These entry points are installed as programs available on the command line. They are listed below with a brief description:
- `flint_skymodel`: derives a sky-model using a reference catalogue suitable to perform bandpass calibration against. Note that it is not "science quality" as it assumes an ideal primary beam response and the reference catalogues do not incorporate spectral information. 
- `flint_aocalibrate`: Performs amplitude and phase calibration against a sky-model, intended for bandpass calibration, and leverage's Andre Offringa's `calibrate` program. 
- `flint_flagger`: Performs basic flagging on an input measurement set. 
- `flint_bandpass`: A small workflow to bandpass calibrate ASKAP measurement sets that have observed PKS B1934-638. 
- `flint_ms`: Utility functions related to inspecting and pre-processing an ASKAP measurement set. 
- `flint_wsclean`: Uses `wsclean` to image and clean an ASKAP measurement set with pre-defined options. 
- `flint_gaincal`: Uses the `casa` task `gaincal` and `applysolutions` to perform self-calibration of an ASKAP measurement set. 
- `flint_convol`: Convols a collection of images to a common resolution. 
- `flint_yandalinmos`: Will co-add a collection of images of a single field together, optionally including holography measurements. 
- `flint_config`: The beginnings of a configuration-based scheme to specify options throughout a workflow. 

The following commands use the `prefect` framework to link together individual tasks together (outlined above) into a single data-processing pipeline. 
- `flint_flow_continuum_pipeline`: Performs bandpass calibration, solution copying, imaging, self-calibration and mosaicing. 


## Sky-model catalogues

The `flint_skymodel` command will attempt to create an in-field sky-model for a particular measurement set using existing source catalogues and an idealised primary beam response. At the moment these catalogue names are hard-coded. Reach out if you need these catalogues. In hopefully the near future this can be relaxed to allow a user-specific catalogue. 

If calibrating a bandpass (i.e. `1934-638`) `flint` will use the packaged source model. At the moment this is only provided for `calibrate`. 

## About ASKAP Measurement Sets

Some of the innovative components of ASKAP and the `yandasoft` package have resulted in measurement sets that are not immediately inline with external tools. Measurement sets should first be processed with [fixms](https://github.com/AlecThomson/FixMS). Be careful -- most (all) `flint` tasks don't currently do this automatically. Be aware, me hearty. 

## Containers 

At the moment this toy pipeline uses `singularity` containers to use compiled software that are outside the `python` ecosystem. For the moment there are no 'supported' container packaged with this repository -- sorry! 

In a nutshell, the containers used throughout are passed in as command line arguments, whose context should be enough to explain what it is expecting. At the time of writing there are two containers for:
- calibration: this should contain `calibrate` and `applysolutions`. These are tools written by Andre Offringa. 
- flagging: this should contain `aoflagger`, which is installable via a `apt install aoflagger` within ubunutu. 
- imaging: this should contain `wsclean`. This should be at least version 3. At the moment a modified version is being used (which implements a `-force-mask-round` option). 


## Contributions

Contributions are welcome! Please do submit a pull-request or issue if you spot something you would like to address. 

