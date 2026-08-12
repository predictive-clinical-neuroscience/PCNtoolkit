from math import log

import pytest
from typing import Any, Callable

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math_functions.basis_function import (
    BasisFunction,
    BsplineBasisFunction,
    LinearBasisFunction,
)
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import BLR
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *
import os

# Default keyword arguments shared by all BLR tests.
BLR_BASE_CONFIG: dict[str, Any] = {
    "n_iter": 1000,
    "tol": 1e-3,
    "ard": False,
    "optimizer": "l-bfgs-b",
    "l_bfgs_b_l": 0.1,
    "l_bfgs_b_epsilon": 0.1,
    "l_bfgs_b_norm": "l1",
    "heteroskedastic": True,
    "fixed_effect": True,
    "warp_name": "WarpSinhArcsinh",
}


@pytest.fixture
def cvfolds():
    return 4


@pytest.fixture
def savemodel():
    return True


@pytest.fixture
def blr_model_factory() -> Callable:
    """Allow tests to build BLR models with custom overrides. By default,
    the factory builds a BLR with the settings in BLR_BASE_CONFIG.

    Examples
    --------
    .. code-block:: python

        def test_cg_optimizer(blr_model_factory):
            model = blr_model_factory(optimizer="cg")
    """
    # Return a function that builds a BLR with the specified overrides.
    return lambda **overrides: BLR(
        "test_blr",
        **{
            "basis_function_mean": BsplineBasisFunction(
                basis_column=0, degree=3, nknots=5
            ),
            **BLR_BASE_CONFIG,
            **overrides,
        },
    )



@pytest.fixture
def fitted_blr_model(
    blr_model_factory: Callable,
    norm_data_from_arrays: NormData,
    fitted_norm_blr_model: NormativeModel,
) -> BLR:
    # Build a default BLR model via the factory.
    blr_model = blr_model_factory()
    if os.path.exists(fitted_norm_blr_model.save_dir):
        shutil.rmtree(fitted_norm_blr_model.save_dir)
    os.makedirs(fitted_norm_blr_model.save_dir, exist_ok=True)
    be_maps = fitted_norm_blr_model.batch_effects_maps
    response_var = norm_data_from_arrays.response_vars[0]
    X, be, be_maps, Y, _ = fitted_norm_blr_model.extract_data(
        norm_data_from_arrays.sel(response_vars=response_var)
    )
    blr_model.fit(X, be, be_maps, Y)
    return blr_model


@pytest.fixture
def norm_blr_model(
    blr_model_factory: Callable,
    save_dir_blr
) -> NormativeModel:
    # Build a default BLR model via the factory.
    blr_model = blr_model_factory()
    if os.path.exists(save_dir_blr):
        shutil.rmtree(save_dir_blr)
    os.makedirs(save_dir_blr, exist_ok=True)
    return NormativeModel(
        blr_model,
        save_dir=save_dir_blr,
        inscaler="standardize",
        outscaler="standardize",
    )


@pytest.fixture
def fitted_norm_blr_model(norm_blr_model: NormativeModel,
                          norm_data_from_arrays: NormData
                          ) -> NormativeModel:
    print("removing items")
    if os.path.exists(norm_blr_model.save_dir):
        shutil.rmtree(norm_blr_model.save_dir)
    os.makedirs(norm_blr_model.save_dir, exist_ok=True)
    norm_blr_model.fit(norm_data_from_arrays)
    return norm_blr_model


@pytest.fixture
def log1p_transform_norm_blr_model(
    save_dir_test_model: str
) -> NormativeModel:
    """Create a NormativeModel using BLR with log1p.

    Returns
    -------
    NormativeModel
        Un-fitted normative model with log1p transform.
    """
    # Build a fresh save directory
    log_dir = os.path.join(save_dir_test_model, "log1p")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    # Create  test regression model
    blr_model = BLR("test_model_log1p")
    # Return a NormativeModel with the log1p transform
    return NormativeModel(
        template_regression_model=blr_model,
        savemodel=False,
        saveresults=False,
        evaluate_model=False,
        saveplots=False,
        save_dir=log_dir,
        inscaler="standardize",
        outscaler="standardize",
        name="test_model_log1p",
        y_transform="log1p",
    )


@pytest.fixture
def log_transform_norm_blr_model(
    save_dir_test_model: str,
) -> NormativeModel:
    """Create a NormativeModel using BLR with natural log y_transform.

    Returns
    -------
    NormativeModel
        Un-fitted normative model with natural-log transform.
    """
    # Build a fresh save directory for this fixture
    log_dir = os.path.join(save_dir_test_model, "log")
    if os.path.exists(log_dir):
        # Remove stale directory from previous test runs
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    # Create a BLR regression model for the natural-log transform test
    blr_model = BLR("test_model_log")
    # Return NormativeModel with natural-log transform enabled
    return NormativeModel(
        template_regression_model=blr_model,
        savemodel=False,
        saveresults=False,
        evaluate_model=False,
        saveplots=False,
        save_dir=log_dir,
        inscaler="standardize",
        outscaler="standardize",
        name="test_model_log",
        y_transform="log",
    )

