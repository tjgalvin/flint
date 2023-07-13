# flint

A pirate themed toy ASKAP-RACS pipeline. 

Yarrrr-Harrrr fiddley-dee!

## About

As a toy, this `flint` package is trying to get a minimum start-to-finish workflow written for `RACS` style data. `python` functions are used to do the work, and `prefect` is used to orchestrate their usage into a larger pipeline. 

Most of the `python` routines have a CLI that can be used to test them in a piecewise sense. These programs are made available when this package is `pip` installed, and they are listed below:
- `flint_skymodel`
- `flint_aocalibrate`
- `flint_flagger`
- `flint_bandpass`
- `flint_ms`

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

## Contributions

Contributions are welcome! Please do submit a pull-request or issue if you spot something you would like to address. 

