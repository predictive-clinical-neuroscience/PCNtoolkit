import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.util.output import Output, Warnings
from test.fixtures.blr_model_fixtures import *  # noqa: F401,F403
from test.fixtures.data_fixtures import *  # noqa: F401,F403
from test.fixtures.norm_data_fixtures import *  # noqa: F401,F403
from test.fixtures.path_fixtures import *  # noqa: F401,F403

"""
Tests for NormativeModel.transfer()
"""


# ------- Helpers -------


@pytest.fixture(scope="module")
def transfer_norm_data_1be(
    n_transfer_subjects: int,
    n_covariates: int,
    n_response_vars: int,
    batch_effect_values: list[list[int]],
) -> NormData:
    """Build a transfer NormData with 1 batch effect column which is fewer 
    than the training data that has 2.

    Parameters
    ----------
    n_transfer_subjects : int
        Number of subjects for the transfer dataset.
    n_covariates : int
        Number of covariate columns.
    n_response_vars : int
        Number of response variable columns.
    batch_effect_values : list[list[int]]
        Full batch effect value lists; only the first entry is
        used.

    Returns
    -------
    NormData
        Transfer dataset with exactly 1 batch effect column.
    """
    n_batch_effects = 1

    # Use only the first batch effect column from the full list
    X, y, be = np_arrays(
        n_transfer_subjects,
        n_covariates,
        n_response_vars,
        batch_effect_values[:n_batch_effects],
    )
    # Wrap into a NormData container
    return NormData.from_ndarrays("transfer_1be", X, y, be)


# ------- Tests -------


def test_001_transfer_should_fit(
    fitted_norm_blr_model: NormativeModel,
    transfer_norm_data_from_arrays: NormData,
) -> None:
    """Transfer must succeed.

    Parameters
    ----------
    fitted_norm_blr_model : NormativeModel
        Pre-fitted BLR normative model.
    transfer_norm_data_from_arrays : NormData
        Transfer dataset
    """
    # Act: transfer
    transferred = fitted_norm_blr_model.transfer(
        transfer_norm_data_from_arrays,
    )
    # Assert: the model is fitted
    assert transferred.is_fitted


def test_002_transfer_should_fit_when_fewerBatchEffects(
    fitted_norm_blr_model: NormativeModel,
    transfer_norm_data_1be: NormData,
) -> None:
    """Transfer must complete and return a fitted model.

    Parameters
    ----------
    fitted_norm_blr_model : NormativeModel
        Pre-fitted BLR normative model.
    transfer_norm_data_1be : NormData
        Transfer dataset with 1 batch effect column.
    """
    # Act: transfer with fewer batch effects than training data
    transferred = fitted_norm_blr_model.transfer(
        transfer_norm_data_1be,
    )
    #  Assert: the model is fitted
    assert transferred.is_fitted


def test_003_transfer_should_warn_when_fewerBatchEffects(
    fitted_norm_blr_model: NormativeModel,
    transfer_norm_data_1be: NormData,
) -> None:
    """Transfer must emit a UserWarning about fewer batch effects.

    Parameters
    ----------
    fitted_norm_blr_model : NormativeModel
        Pre-fitted BLR normative model.
    transfer_norm_data_1be : NormData
        Transfer dataset with 1 batch effect column.
    """
    # Ensure warnings are not suppressed for this test
    Output.set_show_warnings(True)
    # Assert: the warning appeared
    with pytest.warns(
        UserWarning,
        match=Warnings.TRANSFER_DATA_FEWER_BATCH_EFFECTS,
    ):
        fitted_norm_blr_model.transfer(
            transfer_norm_data_1be,
        )
