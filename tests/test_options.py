"""Very basic tests to make sure the FieldOptions class is
somewhat tracked, especially when using an argparse object
to create it
"""
from pathlib import Path

from flint.options import FieldOptions
from flint.prefect.flows.continuum_pipeline import get_parser


def test_create_field_options():
    parser = get_parser()
    args = parser.parse_args(
        """/scratch3/gal16b/askap_sbids/112334/ 
        /scratch3/gal16b/askap_sbids/111/ 
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
        use_smoothed=args.use_smoothed,
    )

    assert isinstance(field_options, FieldOptions)
    assert field_options.use_preflagger is False
    assert field_options.use_smoothed is False
    assert field_options.zip_ms is True
    assert field_options.linmos_residuals is True
    assert field_options.rounds == 2
    assert isinstance(field_options.wsclean_container, Path)


def test_create_field_options2():
    parser = get_parser()
    args = parser.parse_args(
        """/scratch3/gal16b/askap_sbids/112334/ 
        /scratch3/gal16b/askap_sbids/111/ 
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
        --use-smoothed
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
        use_smoothed=args.use_smoothed,
    )

    assert isinstance(field_options, FieldOptions)
    assert field_options.use_preflagger is True
    assert field_options.use_smoothed is True
    assert field_options.zip_ms is False
    assert field_options.linmos_residuals is False
    assert field_options.rounds == 2
    assert isinstance(field_options.wsclean_container, Path)
