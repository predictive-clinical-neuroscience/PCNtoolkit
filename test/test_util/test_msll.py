"""
Test to verify that MSLL is computed correctly regardless of data scale.

This test addresses the issue where MSLL was scale-dependent because the baseline
log-likelihood was computed on unscaled Y values while the model log-likelihood
was computed on scaled Y values.

The fix ensures both are computed on the same (scaled) data.
"""

from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR
from test.fixtures.evaluator_fixtures import create_test_data


def test_msll_remains_similar_when_scale_changes(tmp_path):
    """
    Test that MSLL values are similar when data is scaled differently,
    confirming that MSLL is scale-independent.

    Using the same random seed, data at different scales should produce
    similar MSLL values.
    """
    msll_values = {}

    for scale_factor in [1.0, 1e-4]:
        # Create data with specified scale (keep the same seed)
        data = create_test_data(
            n_samples=100, scale_factor=scale_factor, seed=123)

        # Create and fit model
        blr = BLR()
        model = NormativeModel(
            template_regression_model=blr,
            savemodel=False,
            evaluate_model=True,
            saveresults=False,
            saveplots=False,
            save_dir=str(tmp_path / f"scale_{scale_factor}"),
            inscaler="standardize",
            outscaler="standardize",
        )

        model.fit(data)

        msll = float(data.statistics.sel(
            response_vars="test_metric", statistic="MSLL").values)
        msll_values[scale_factor] = msll
        print(f"Scale factor: {scale_factor}, MSLL: {msll}")

    # If both fitted and baseline models are computed on scaled data then the
    # MSLL should remain similar
    msll_diff = abs(msll_values[1.0] - msll_values[1e-4])
    print(f"MSLL difference between scales: {msll_diff}")

    # Allow some numerical tolerance depending on the specs of the PC
    assert msll_diff < 1e-8, (
        f"MSLL differs too much between scales: "
        f"scale=1.0 -> {msll_values[1.0]}, scale=1e-4 -> {msll_values[1e-4]}"
    )
