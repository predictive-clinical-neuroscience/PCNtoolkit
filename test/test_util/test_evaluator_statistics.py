"""
Tests for how Evaluator.evaluate handles the requested statistics.

Verifies that:
- Computing MLL does not emit a spurious "NLL renamed" warning.
- The old name NLL is still accepted, with a deprecation warning.
- Unknown statistic names produce a warning that lists valid names.
- Rho and Rho_p are always computed together.
"""

import warnings

import numpy as np
import pytest
import xarray as xr

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.util.evaluator import Evaluator
from test.fixtures.evaluator_fixtures import create_test_data


def _make_data(n: int = 60, seed: int = 0) -> NormData:
    """Create a NormData with the Yhat and logp variables the evaluator needs."""
    data = create_test_data(n_samples=n, seed=seed, n_batch_effects=0)
    rng = np.random.default_rng(seed)
    shape = data["Y"].shape
    dims = data["Y"].dims
    data["Yhat"] = xr.DataArray(data["Y"].values + rng.normal(0, 1, shape), dims=dims)
    data["logp"] = xr.DataArray(rng.normal(-1, 0.1, shape), dims=dims)
    return data


def test_001_mll_should_notWarnAboutRename_when_mllRequested() -> None:
    """
    Requesting MLL must not trigger the "NLL has been renamed" warning.

    Arrange: data with Yhat and logp.
    Act: evaluate with statistics=["MLL"] while recording warnings.
    Assert: no DeprecationWarning is raised and MLL is filled in.
    """
    # Arrange
    data = _make_data()

    # Act
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Evaluator().evaluate(data, statistics=["MLL"])

    # Assert
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert np.isfinite(result.statistics.sel(statistic="MLL").values).all()


def test_002_nll_should_mapToMll_when_oldNameRequested() -> None:
    """
    The old name NLL still works, but warns that it was renamed.

    Arrange: data with Yhat and logp.
    Act: evaluate with statistics=["NLL"].
    Assert: a DeprecationWarning names both NLL and MLL, and MLL is computed.
    """
    # Arrange
    data = _make_data()

    # Act
    with pytest.warns(DeprecationWarning, match="'NLL' has been renamed to 'MLL'"):
        result = Evaluator().evaluate(data, statistics=["NLL"])

    # Assert
    assert list(result.statistics.statistic.values) == ["MLL"]
    assert np.isfinite(result.statistics.sel(statistic="MLL").values).all()


def test_003_unknownStatistic_should_warn_when_nameNotValid() -> None:
    """
    Unknown names are dropped with a warning that lists the valid names.

    Arrange: data with Yhat.
    Act: evaluate with one valid and one unknown statistic.
    Assert: the warning names the typo, and the valid statistic is computed.
    """
    # Arrange
    data = _make_data()

    # Act
    with pytest.warns(UserWarning, match=r"unknown statistic\(s\): R3.*Valid"):
        result = Evaluator().evaluate(data, statistics=["R2", "R3"])

    # Assert
    assert list(result.statistics.statistic.values) == ["R2"]


def test_004_rhoP_should_beComputed_when_onlyRhoPRequested() -> None:
    """
    Requesting only Rho_p must not leave it as NaN.

    Arrange: data with Yhat.
    Act: evaluate with statistics=["Rho_p"].
    Assert: both Rho and Rho_p are present and finite.
    """
    # Arrange
    data = _make_data()

    # Act
    result = Evaluator().evaluate(data, statistics=["Rho_p"])

    # Assert
    assert set(result.statistics.statistic.values) == {"Rho", "Rho_p"}
    assert np.isfinite(result.statistics.values).all()


@pytest.mark.parametrize("requested", [None, []])
def test_005_resolveStatistics_should_returnAll_when_nothingRequested(
    requested: list[str] | None,
) -> None:
    """
    No explicit request means every statistic is computed.

    Arrange: an Evaluator.
    Act: resolve None or an empty list.
    Assert: the full list of statistics is returned, as a fresh copy.
    """
    # Arrange
    evaluator = Evaluator()

    # Act
    resolved = evaluator.resolve_statistics(requested)

    # Assert
    assert resolved == Evaluator.ALL_STATISTICS
    assert resolved is not Evaluator.ALL_STATISTICS
