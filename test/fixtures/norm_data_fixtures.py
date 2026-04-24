import pytest

from pcntoolkit.dataio.norm_data import NormData
from test.fixtures.data_fixtures import *
"""
This file contains pytest fixtures for generating NormData objects in the PCNtoolkit.

The fixtures defined here include:
1. NormData objects from numpy arrays
2. NormData objects from pandas DataFrames

These fixtures are used to create consistent and controlled datasets for testing
"""


@pytest.fixture(scope="module")
def norm_data_from_arrays(train_arrays):
    X, y, batch_effects = train_arrays
    return NormData.from_ndarrays("from_arrays", X, y, batch_effects)


@pytest.fixture(scope="module")
def test_norm_data_from_arrays(test_arrays):
    X, y, batch_effects = test_arrays
    return NormData.from_ndarrays("from_arrays_test", X, y, batch_effects)


@pytest.fixture(scope="module")
def transfer_norm_data_from_arrays(transfer_arrays) -> NormData:
    """Build transfer NormData from the transfer_arrays fixture.

    Parameters
    ----------
    transfer_arrays : tuple
        ``(X, y, batch_effects)`` arrays from data_fixtures.

    Returns
    -------
    NormData
        Transfer dataset.
    """
    # Unpack the tuple returned by the transfer_arrays fixture
    X, y, batch_effects = transfer_arrays
    return NormData.from_ndarrays(
        "from_arrays_transfer", X, y, batch_effects
    )


@pytest.fixture(scope="module")
def norm_data_from_dataframe(train_dataframe, n_covariates, batch_effect_values, n_response_vars):
    return NormData.from_dataframe(
        "from_dataframe",
        train_dataframe,
        covariates=[f"covariate_{i}" for i in range(n_covariates)],
        batch_effects=[f"batch_effect_{i}" for i in range(len(batch_effect_values))],
        response_vars=[f"response_var_{i}" for i in range(n_response_vars)],
    )
