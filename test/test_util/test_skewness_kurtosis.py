"""
Tests for the Skew and Kurt evaluation metrics.

Verifies that:
- Skewness and excess kurtosis are near 0 for
  standard-normal z-scores.
- Skewness is positive for right-skewed z-scores.
- Excess kurtosis is positive for heavy-tailed z-scores.
- Both metrics handle NaN and Inf values gracefully.
"""

import numpy as np
import xarray as xr

from pcntoolkit.util.evaluator import Evaluator
from test.fixtures.evaluator_fixtures import create_test_data


# ── helpers ──────────────────────────────────────────────────

def _add_z_scores(
    data,
    z_values: np.ndarray,
) -> object:
    """
    Add a Z variable to a NormData object.

    Parameters
    ----------
    data : NormData
        Dataset returned by create_test_data.
    z_values : np.ndarray
        1-D array of z-scores with length equal to
        the number of observations in *data*.

    Returns
    -------
    NormData
        Slice for response variable 'test_metric' with
        Z values assigned.
    """
    # Reshape to (n_observations, n_response_vars=1)
    z_2d = z_values.reshape(-1, 1)
    # Assign Z as an xarray DataArray with matching dims
    data["Z"] = xr.DataArray(
        z_2d,
        dims=("observations", "response_vars"),
        coords={
            "observations": data.coords["observations"],
            "response_vars": data.coords["response_vars"],
        },
    )
    # Select the single response variable so _evaluate_*
    # receives a 1-D-compatible slice
    return data.sel({"response_vars": "test_metric"})


# ── tests for Skew ───────────────────────────────────────────

def test_001_skew_should_beNearZero_when_zscoresAreNormal() -> (
    None
):
    """
    Skewness of standard-normal z-scores should be close to 0.

    Arrange: create normally distributed z-scores (n=5000).
    Act: call _evaluate_skew.
    Assert: result is within 0.1 of zero.
    """
    # Arrange
    rng = np.random.default_rng(0)
    z = rng.standard_normal(5000)
    data = create_test_data(n_samples=5000, seed=0)
    sliced = _add_z_scores(data, z)
    evaluator = Evaluator()

    # Act
    result = evaluator._evaluate_skew(sliced)

    # Assert
    assert abs(result) < 0.1


def test_002_kurt_should_beNearZero_when_zscoresAreNormal() -> (
    None
):
    """
    Excess kurtosis of standard-normal z-scores should be near 0.

    Arrange: create normally distributed z-scores (n=5000).
    Act: call _evaluate_kurt.
    Assert: result is within 0.1 of zero.
    """
    # Arrange
    rng = np.random.default_rng(1)
    z = rng.standard_normal(5000)
    data = create_test_data(n_samples=5000, seed=1)
    sliced = _add_z_scores(data, z)
    evaluator = Evaluator()

    # Act
    result = evaluator._evaluate_kurt(sliced)

    # Assert
    assert abs(result) < 0.1


def test_003_skew_should_bePositive_when_zscoresAreRightSkewed() -> (
    None
):
    """
    Skewness should be positive for right-skewed z-scores.

    Arrange: create right-skewed z-scores using a log-normal
    distribution minus its mean (n=5000).
    Act: call _evaluate_skew.
    Assert: result > 0.
    """
    # Arrange: log-normal samples are right-skewed
    rng = np.random.default_rng(2)
    z = rng.lognormal(mean=0.0, sigma=1.0, size=5000)
    # Centre around 0 to keep it as a z-score analogue
    z = z - z.mean()
    data = create_test_data(n_samples=5000, seed=2)
    sliced = _add_z_scores(data, z)
    evaluator = Evaluator()

    # Act
    result = evaluator._evaluate_skew(sliced)

    # Assert
    assert result > 0.0


def test_004_kurt_should_bePositive_when_zscoresAreLeptokurtic() -> (
    None
):
    """
    Excess kurtosis should be positive for heavy-tailed z-scores.

    Arrange: draw z-scores from a t-distribution with 3 degrees
    of freedom, which is leptokurtic (excess kurtosis = 6).
    Act: call _evaluate_kurt.
    Assert: result > 0.
    """
    # Arrange: t(3) has excess kurtosis = 6 — heavy tails
    rng = np.random.default_rng(3)
    z = rng.standard_t(df=3, size=5000)
    data = create_test_data(n_samples=5000, seed=3)
    sliced = _add_z_scores(data, z)
    evaluator = Evaluator()

    # Act
    result = evaluator._evaluate_kurt(sliced)

    # Assert
    assert result > 0.0


def test_005_skew_should_handleNanAndInf_when_zscoresContainInvalidValues() -> (
    None
):
    """
    _evaluate_skew should return a finite float when z-scores
    contain Inf and NaN values.

    Arrange: normal z-scores with 5 Inf and 5 NaN entries.
    Act: call _evaluate_skew.
    Assert: result is finite.
    """
    # Arrange
    rng = np.random.default_rng(4)
    z = rng.standard_normal(500)
    # Inject invalid values
    z[0:5] = np.inf
    z[5:10] = np.nan
    data = create_test_data(n_samples=500, seed=4)
    sliced = _add_z_scores(data, z)
    evaluator = Evaluator()

    # Act
    result = evaluator._evaluate_skew(sliced)

    # Assert
    assert np.isfinite(result)


def test_006_kurt_should_handleNanAndInf_when_zscoresContainInvalidValues() -> (
    None
):
    """
    _evaluate_kurt should return a finite float when z-scores
    contain Inf and NaN values.

    Arrange: normal z-scores with 5 Inf and 5 NaN entries.
    Act: call _evaluate_kurt.
    Assert: result is finite.
    """
    # Arrange
    rng = np.random.default_rng(5)
    z = rng.standard_normal(500)
    # Inject invalid values
    z[0:5] = np.inf
    z[5:10] = np.nan
    data = create_test_data(n_samples=500, seed=5)
    sliced = _add_z_scores(data, z)
    evaluator = Evaluator()

    # Act
    result = evaluator._evaluate_kurt(sliced)

    # Assert
    assert np.isfinite(result)
