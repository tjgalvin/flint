# Validation

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
