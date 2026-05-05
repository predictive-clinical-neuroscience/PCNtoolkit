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
from test.fixtures.plotter_fixtures import create_test_data_with_z


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
    # Arrange: create_test_data_with_z already injects
    # standard-normal Z-scores, so no overwrite needed
    data = create_test_data_with_z(n=5000, seed=0)
    # Slice to the single response variable
    sliced = data.sel({"response_vars": "metric"})
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
    # Arrange: create_test_data_with_z already injects
    # standard-normal Z-scores, so no overwrite needed
    data = create_test_data_with_z(n=5000, seed=1)
    # Slice to the single response variable
    sliced = data.sel({"response_vars": "metric"})
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
    data = create_test_data_with_z(n=5000, seed=2)
    # Overwrite Z with the right-skewed distribution
    data["Z"] = xr.DataArray(
        z.reshape(-1, 1),
        dims=("observations", "response_vars"),
        coords={
            "observations": data.coords["observations"],
            "response_vars": data.coords["response_vars"],
        },
    )
    # Slice to the single response variable
    sliced = data.sel({"response_vars": "metric"})
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
    data = create_test_data_with_z(n=5000, seed=3)
    # Overwrite Z with the heavy-tailed t-distribution
    data["Z"] = xr.DataArray(
        z.reshape(-1, 1),
        dims=("observations", "response_vars"),
        coords={
            "observations": data.coords["observations"],
            "response_vars": data.coords["response_vars"],
        },
    )
    # Slice to the single response variable
    sliced = data.sel({"response_vars": "metric"})
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
    data = create_test_data_with_z(n=500, seed=4)
    # Overwrite Z with the array containing Inf and NaN
    data["Z"] = xr.DataArray(
        z.reshape(-1, 1),
        dims=("observations", "response_vars"),
        coords={
            "observations": data.coords["observations"],
            "response_vars": data.coords["response_vars"],
        },
    )
    # Slice to the single response variable
    sliced = data.sel({"response_vars": "metric"})
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
    data = create_test_data_with_z(n=500, seed=5)
    # Overwrite Z with the array containing Inf and NaN
    data["Z"] = xr.DataArray(
        z.reshape(-1, 1),
        dims=("observations", "response_vars"),
        coords={
            "observations": data.coords["observations"],
            "response_vars": data.coords["response_vars"],
        },
    )
    # Slice to the single response variable
    sliced = data.sel({"response_vars": "metric"})
    evaluator = Evaluator()

    # Act
    result = evaluator._evaluate_kurt(sliced)

    # Assert
    assert np.isfinite(result)
