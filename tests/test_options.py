"""Very basic tests to make sure the FieldOptions class is
somewhat tracked, especially when using an argparse object
to create it
"""

import pytest
from pathlib import Path

from flint.options import FieldOptions, dump_field_options_to_yaml
from flint.prefect.flows.continuum_pipeline import get_parser


def test_dump_field_options_to_yaml(tmpdir):
    """See if the field options file can be dumped to an output directory"""
    tmpdir = Path(tmpdir)

    field_options = FieldOptions(
        flagger_container=Path("a"), calibrate_container=Path("b")
    )

    assert not (tmpdir / "Jack").exists()

    path_1 = tmpdir / "field_options.yaml"
    path_2 = tmpdir / "Jack" / "Sparrow" / "field_options.yaml"

    for path in (path_1, path_2):
        output_path = dump_field_options_to_yaml(
            output_path=path, field_options=field_options
        )
        assert output_path.exists()

    with pytest.raises(FileExistsError):
        dump_field_options_to_yaml(output_path=path_2, field_options=field_options)


def test_config_field_options(tmpdir):
    output_file = f"{tmpdir}/example.config"
    contents = """--holofile /scratch3/projects/spiceracs/RACS_Low2_Holography/akpb.iquv.square_6x6.63.887MHz.SB39549.cube.fits
        --calibrate-container /scratch3/gal16b/containers/calibrate.sif
        --flagger-container /scratch3/gal16b/containers/aoflagger.sif
        --wsclean-container /scratch3/projects/spiceracs/singularity_images/wsclean_force_mask.sif
        --yandasoft-container /scratch3/gal16b/containers/yandasoft.sif
        --cluster-config /scratch3/gal16b/split/petrichor.yaml
        --selfcal-rounds 2
        --split-path $(pwd)
        --zip-ms
        --run-aegean
        --aegean-container '/scratch3/gal16b/containers/aegean.sif'
        --reference-catalogue-directory '/scratch3/gal16b/reference_catalogues/'
        --linmos-residuals
    """

    with open(output_file, "w") as out:
        for line in contents.split("\n"):
            out.write(f"{line.lstrip()}\n")

    parser = get_parser()
    args = parser.parse_args(
        f"""/scratch3/gal16b/askap_sbids/112334/
        --calibrated-bandpass-path /scratch3/gal16b/askap_sbids/111/
        --cli-config {str(output_file)}""".split()
    )

    field_options = FieldOptions(
        flagger_container=args.flagger_container,
        calibrate_container=args.calibrate_container,
        holofile=args.holofile,
        expected_ms=args.expected_ms,
        wsclean_container=args.wsclean_container,
        yandasoft_container=args.yandasoft_container,
        rounds=args.selfcal_rounds,
        zip_ms=args.zip_ms,
        run_aegean=args.run_aegean,
        aegean_container=args.aegean_container,
        no_imaging=args.no_imaging,
        reference_catalogue_directory=args.reference_catalogue_directory,
        linmos_residuals=args.linmos_residuals,
        beam_cutoff=args.beam_cutoff,
        pb_cutoff=args.pb_cutoff,
        use_preflagger=args.use_preflagger,
    )

    assert isinstance(field_options, FieldOptions)
    assert field_options.use_preflagger is False
    assert field_options.zip_ms is True
    assert field_options.linmos_residuals is True
    assert field_options.rounds == 2
    assert isinstance(field_options.wsclean_container, Path)


def test_create_field_options():
    parser = get_parser()
    args = parser.parse_args(
        """/scratch3/gal16b/askap_sbids/112334/
        --calibrated-bandpass-path /scratch3/gal16b/askap_sbids/111/
        --holofile /scratch3/projects/spiceracs/RACS_Low2_Holography/akpb.iquv.square_6x6.63.887MHz.SB39549.cube.fits
        --calibrate-container /scratch3/gal16b/containers/calibrate.sif
        --flagger-container /scratch3/gal16b/containers/aoflagger.sif
        --wsclean-container /scratch3/projects/spiceracs/singularity_images/wsclean_force_mask.sif
        --yandasoft-container /scratch3/gal16b/containers/yandasoft.sif
        --cluster-config /scratch3/gal16b/split/petrichor.yaml
        --selfcal-rounds 2
        --split-path $(pwd)
        --zip-ms
        --run-aegean
        --aegean-container '/scratch3/gal16b/containers/aegean.sif'
        --reference-catalogue-directory '/scratch3/gal16b/reference_catalogues/'
        --linmos-residuals
    """.split()
    )

    field_options = FieldOptions(
        flagger_container=args.flagger_container,
        calibrate_container=args.calibrate_container,
        holofile=args.holofile,
        expected_ms=args.expected_ms,
        wsclean_container=args.wsclean_container,
        yandasoft_container=args.yandasoft_container,
        rounds=args.selfcal_rounds,
        zip_ms=args.zip_ms,
        run_aegean=args.run_aegean,
        aegean_container=args.aegean_container,
        no_imaging=args.no_imaging,
        reference_catalogue_directory=args.reference_catalogue_directory,
        linmos_residuals=args.linmos_residuals,
        beam_cutoff=args.beam_cutoff,
        pb_cutoff=args.pb_cutoff,
        use_preflagger=args.use_preflagger,
    )

    assert isinstance(field_options, FieldOptions)
    assert field_options.use_preflagger is False
    assert field_options.zip_ms is True
    assert field_options.linmos_residuals is True
    assert field_options.rounds == 2
    assert isinstance(field_options.wsclean_container, Path)


def test_create_field_options2():
    parser = get_parser()
    args = parser.parse_args(
        """/scratch3/gal16b/askap_sbids/112334/
        --calibrated-bandpass-path /scratch3/gal16b/askap_sbids/111/
        --holofile /scratch3/projects/spiceracs/RACS_Low2_Holography/akpb.iquv.square_6x6.63.887MHz.SB39549.cube.fits
        --calibrate-container /scratch3/gal16b/containers/calibrate.sif
        --flagger-container /scratch3/gal16b/containers/aoflagger.sif
        --wsclean-container /scratch3/projects/spiceracs/singularity_images/wsclean_force_mask.sif
        --yandasoft-container /scratch3/gal16b/containers/yandasoft.sif
        --cluster-config /scratch3/gal16b/split/petrichor.yaml
        --selfcal-rounds 2
        --split-path $(pwd)
        --run-aegean
        --aegean-container '/scratch3/gal16b/containers/aegean.sif'
        --reference-catalogue-directory '/scratch3/gal16b/reference_catalogues/'
        --use-preflagger
    """.split()
    )

    field_options = FieldOptions(
        flagger_container=args.flagger_container,
        calibrate_container=args.calibrate_container,
        holofile=args.holofile,
        expected_ms=args.expected_ms,
        wsclean_container=args.wsclean_container,
        yandasoft_container=args.yandasoft_container,
        rounds=args.selfcal_rounds,
        zip_ms=args.zip_ms,
        run_aegean=args.run_aegean,
        aegean_container=args.aegean_container,
        no_imaging=args.no_imaging,
        reference_catalogue_directory=args.reference_catalogue_directory,
        linmos_residuals=args.linmos_residuals,
        beam_cutoff=args.beam_cutoff,
        pb_cutoff=args.pb_cutoff,
        use_preflagger=args.use_preflagger,
    )

    assert isinstance(field_options, FieldOptions)
    assert field_options.use_preflagger is True
    assert field_options.zip_ms is False
    assert field_options.linmos_residuals is False
    assert field_options.rounds == 2
    assert isinstance(field_options.wsclean_container, Path)


def test_create_field_options3():
    """Make sure that the calibrated-bandpass-path can be left unchecked"""
    parser = get_parser()
    args = parser.parse_args(
        """/scratch3/gal16b/askap_sbids/112334/
        --holofile /scratch3/projects/spiceracs/RACS_Low2_Holography/akpb.iquv.square_6x6.63.887MHz.SB39549.cube.fits
        --calibrate-container /scratch3/gal16b/containers/calibrate.sif
        --flagger-container /scratch3/gal16b/containers/aoflagger.sif
        --wsclean-container /scratch3/projects/spiceracs/singularity_images/wsclean_force_mask.sif
        --yandasoft-container /scratch3/gal16b/containers/yandasoft.sif
        --cluster-config /scratch3/gal16b/split/petrichor.yaml
        --selfcal-rounds 2
        --split-path $(pwd)
        --run-aegean
        --aegean-container '/scratch3/gal16b/containers/aegean.sif'
        --reference-catalogue-directory '/scratch3/gal16b/reference_catalogues/'
        --use-preflagger
    """.split()
    )

    field_options = FieldOptions(
        flagger_container=args.flagger_container,
        calibrate_container=args.calibrate_container,
        holofile=args.holofile,
        expected_ms=args.expected_ms,
        wsclean_container=args.wsclean_container,
        yandasoft_container=args.yandasoft_container,
        rounds=args.selfcal_rounds,
        zip_ms=args.zip_ms,
        run_aegean=args.run_aegean,
        aegean_container=args.aegean_container,
        no_imaging=args.no_imaging,
        reference_catalogue_directory=args.reference_catalogue_directory,
        linmos_residuals=args.linmos_residuals,
        beam_cutoff=args.beam_cutoff,
        pb_cutoff=args.pb_cutoff,
        use_preflagger=args.use_preflagger,
    )

    assert isinstance(field_options, FieldOptions)
    assert field_options.use_preflagger is True
    assert field_options.zip_ms is False
    assert field_options.linmos_residuals is False
    assert field_options.rounds == 2
    assert isinstance(field_options.wsclean_container, Path)
    assert args.calibrated_bandpass_path is None
