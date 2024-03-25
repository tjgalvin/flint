"""Basic tests for the prefect bandpass flow"""

# this pirate made a promise that if there is a typo and an error
# then a test will be made to not let it happen. Penaalty is walking
# the plank

from flint.prefect.flows.bandpass_pipeline import get_parser


def test_parser():
    parser = get_parser()

    args = parser.parse_args(
        """/51998
        --calibrate-container /scratch3/gal16b/containers/calibrate.sif
        --flagger-container /scratch3/gal16b/containers/aoflagger.sif
        --cluster-config /scratch3/gal16b/bp_test/petrichor.yaml
        --split-path $(pwd)
        --smooth-window-size 8
        --smooth-polynomial-order 3
    """.split()
    )

    assert args.smooth_polynomial_order == 3
    assert args.smooth_window_size == 8
    assert isinstance(args.smooth_window_size, int)
    assert isinstance(args.smooth_polynomial_order, int)
