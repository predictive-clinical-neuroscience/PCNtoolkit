import copy

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math_functions.likelihood import NormalLikelihood
from pcntoolkit.math_functions.prior import *
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import *
from pcntoolkit.regression_model.hbr import *
from test.fixtures.blr_model_fixtures import *
from test.fixtures.data_fixtures import *
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *
from test.fixtures.test_model_fixtures import *

"""
This file contains tests for the NormBase class in the PCNtoolkit.

The tests cover the following aspects:
1. Fitting the model with valid data
2. Handling invalid data for fitting
3. Scaling methods for data
4. Polynomial basis expansion
5. B-spline basis expansion
"""


def test_fit(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    if os.path.exists(new_norm_test_model.save_dir):
        shutil.rmtree(new_norm_test_model.save_dir)
    os.makedirs(new_norm_test_model.save_dir, exist_ok=True)
    new_norm_test_model.fit(norm_data_from_arrays)
    assert new_norm_test_model.is_fitted


def test_fit_predict(new_norm_test_model: NormativeModel, norm_data_from_arrays: NormData, test_norm_data_from_arrays: NormData):
    if os.path.exists(new_norm_test_model.save_dir):
        shutil.rmtree(new_norm_test_model.save_dir)
    os.makedirs(new_norm_test_model.save_dir, exist_ok=True)
    new_norm_test_model.fit_predict(norm_data_from_arrays, test_norm_data_from_arrays)
    assert new_norm_test_model.is_fitted


def test_predict(fitted_norm_test_model: NormativeModel, test_norm_data_from_arrays: NormData):
    os.makedirs(fitted_norm_test_model.save_dir, exist_ok=True)
    fitted_norm_test_model.predict(test_norm_data_from_arrays)
    assert fitted_norm_test_model.is_fitted


def test_harmonize(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    fitted_norm_test_model.harmonize(norm_data_from_arrays)
    assert fitted_norm_test_model.is_fitted


def test_compute_zscores(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    fitted_norm_test_model.compute_zscores(norm_data_from_arrays)
    assert fitted_norm_test_model.is_fitted


def test_compute_centiles(fitted_norm_test_model: NormativeModel, norm_data_from_arrays: NormData):
    fitted_norm_test_model.compute_centiles(norm_data_from_arrays)
    assert fitted_norm_test_model.is_fitted


# Test the fit method of the NormBase class with invalid data
def test_norm_base_fit_invalid_data(new_norm_test_model: NormativeModel):
    with pytest.raises(AttributeError):
        new_norm_test_model.fit(None)

    with pytest.raises(AttributeError):
        new_norm_test_model.fit("invalid_data")


# Scaling tests
@pytest.mark.parametrize("scaler", ["standardize", "minmax"])
def test_scaling(
    norm_data_from_arrays,
    new_norm_test_model: NormativeModel,
    n_covariates,
    n_response_vars,
    scaler,
):
    object.__setattr__(new_norm_test_model, "inscaler", scaler)
    object.__setattr__(new_norm_test_model, "outscaler", scaler)
    copydata = copy.deepcopy(norm_data_from_arrays)
    copydata.attrs["is_scaled"] = False
    for s in new_norm_test_model.inscalers:
        s.is_fitted = False
    for s in new_norm_test_model.outscalers:
        s.is_fitted = False
    X_bak = norm_data_from_arrays.X.data.copy()
    y_bak = norm_data_from_arrays.Y.data.copy()
    new_norm_test_model.scale_forward(copydata)

    if scaler == "standardize":
        assert_standardized(copydata, n_covariates, n_response_vars)
    elif scaler == "minmax":
        assert_minmax_scaled(copydata)

    new_norm_test_model.scale_backward(copydata)
    assert np.allclose(copydata.X.data, X_bak)
    assert np.allclose(copydata.Y.data, y_bak)


def assert_standardized(data, n_covariates, n_response_vars):
    assert data.X.data.mean(axis=0) == pytest.approx(n_covariates * [0])
    assert data.X.data.std(axis=0) == pytest.approx(n_covariates * [1])
    assert data.Y.data.mean(axis=0) == pytest.approx(n_response_vars * [0])
    assert data.Y.data.std(axis=0) == pytest.approx(n_response_vars * [1])


def assert_minmax_scaled(data):
    assert np.allclose(data.X.data.min(axis=0), 0)
    assert np.allclose(data.X.data.max(axis=0), 1)
    assert np.allclose(data.Y.data.min(axis=0), 0)
    assert np.allclose(data.Y.data.max(axis=0), 1)


def test_test_model_to_and_from_dict_and_args(test_model_args: dict, norm_data_from_arrays: NormData, save_dir_test_model):
    model = NormativeModel.from_args(**test_model_args)
    model_dict = model.to_dict()
    assert model_dict["template_regression_model"]["success_ratio"] == test_model_args["success_ratio"]
    model.fit(norm_data_from_arrays)
    assert model.is_fitted
    model_dict = model.to_dict()
    assert model_dict["template_regression_model"]["success_ratio"] == test_model_args["success_ratio"]
    model.predict(norm_data_from_arrays)
    assert hasattr(norm_data_from_arrays, "Z")
    Z_bak = copy.deepcopy(norm_data_from_arrays["Z"])
    del norm_data_from_arrays["Z"]
    assert not hasattr(norm_data_from_arrays, "Z")
    model.save(save_dir_test_model)
    loaded_model = NormativeModel.load(save_dir_test_model)
    assert loaded_model.is_fitted
    loaded_model.predict(norm_data_from_arrays)
    assert np.allclose(norm_data_from_arrays["Z"], Z_bak)


@pytest.fixture
def hbr_model_args(save_dir_hbr):
    return {
        "savemodel": False,
        "saveresults": False,
        "evaluate_model": False,
        "saveplots": False,
        "save_dir": save_dir_hbr,
        "inscaler": "standardize",
        "outscaler": "standardize",
        "name": "hbr_test_model",
        "alg": "hbr",
        "likelihood": "Normal",
        "linear_mu": True,
        "random_mu": False,
        "random_slope_mu": False,
        "random_intercept_mu": True,
        "linear_sigma": False,
        "random_sigma": True,
        "dist_name_sigma_sigma": "LogNormal",
        "dist_params_sigma_sigma": (2.5, 1.3),
        "draws": 10,
        "tune": 10,
        "cores": 1,
    }


def test_hbr_model_to_and_from_dict_and_args(hbr_model_args: dict, norm_data_from_arrays: NormData, save_dir_hbr):
    model = NormativeModel.from_args(**hbr_model_args)
    model_dict = model.to_dict()
    for k in ["savemodel", "saveresults", "evaluate_model", "saveplots", "inscaler", "outscaler", "name"]:
        assert model_dict[k] == hbr_model_args[k]
    tmplt = model.template_regression_model
    assert isinstance(tmplt, HBR)
    assert isinstance(tmplt.likelihood, NormalLikelihood)
    assert isinstance(tmplt.likelihood.mu, LinearPrior)
    assert isinstance(tmplt.likelihood.mu.intercept, RandomPrior)
    assert isinstance(tmplt.likelihood.sigma, RandomPrior)
    assert isinstance(tmplt.likelihood.sigma.sigma, Prior)
    assert tmplt.likelihood.sigma.sigma.dist_name == "LogNormal"
    assert tmplt.likelihood.sigma.sigma.dist_params == (2.5, 1.3)
    model.fit(norm_data_from_arrays)
    assert model.is_fitted

    model1 = model[model.response_vars[0]]
    assert isinstance(model1, HBR)
    assert isinstance(model1.likelihood, NormalLikelihood)
    assert isinstance(model1.likelihood.mu, LinearPrior)
    assert isinstance(model1.likelihood.mu.intercept, RandomPrior)
    assert isinstance(model1.likelihood.sigma, RandomPrior)
    assert isinstance(model1.likelihood.sigma.sigma, Prior)
    assert model1.likelihood.sigma.sigma.dist_name == "LogNormal"
    assert model1.likelihood.sigma.sigma.dist_params == (2.5, 1.3)

    model.predict(norm_data_from_arrays)
    assert hasattr(norm_data_from_arrays, "Z")
    Z_bak = copy.deepcopy(norm_data_from_arrays["Z"])
    del norm_data_from_arrays["Z"]
    assert not hasattr(norm_data_from_arrays, "Z")
    model.save(save_dir_hbr)
    del model
    loaded_model = NormativeModel.load(save_dir_hbr)
    loaded_model.predict(norm_data_from_arrays)
    assert np.allclose(norm_data_from_arrays["Z"], Z_bak)
    assert loaded_model.is_fitted
    model1 = loaded_model[loaded_model.response_vars[0]]
    assert isinstance(model1, HBR)
    assert isinstance(model1.likelihood, NormalLikelihood)
    assert isinstance(model1.likelihood.mu, LinearPrior)
    assert isinstance(model1.likelihood.mu.intercept, RandomPrior)
    assert isinstance(model1.likelihood.sigma, RandomPrior)
    assert isinstance(model1.likelihood.sigma.sigma, Prior)
    assert model1.likelihood.sigma.sigma.dist_name == "LogNormal"
    assert model1.likelihood.sigma.sigma.dist_params == [2.5, 1.3]


@pytest.fixture
def blr_model_args(save_dir_blr):
    return {
        "savemodel": False,
        "saveresults": False,
        "evaluate_model": False,
        "saveplots": False,
        "save_dir": save_dir_blr,
        "inscaler": "standardize",
        "outscaler": "standardize",
        "name": "blr_test_model",
        "alg": "blr",
        "n_iter": 10,
        "tol": 1e-3,
        "ard": False,
        "optimizer": "l-bfgs-b",
        "l_bfgs_b_l": 0.7,
        "l_bfgs_b_epsilon": 0.1,
        "l_bfgs_b_norm": "l2",
        "fixed_effect": True,
        "heteroskedastic": False,
        "fixed_effect_var": False,
    }


def test_blr_model_to_and_from_dict_and_args(blr_model_args: dict, norm_data_from_arrays: NormData, save_dir_blr):
    model = NormativeModel.from_args(**blr_model_args)
    model_dict = model.to_dict()
    for k in ["savemodel", "saveresults", "evaluate_model", "saveplots", "inscaler", "outscaler", "name"]:
        assert model_dict[k] == blr_model_args[k]
    tmplt = model.template_regression_model
    assert isinstance(tmplt, BLR)
    assert tmplt.n_iter == 10
    assert tmplt.tol == 1e-3
    assert not tmplt.ard
    assert tmplt.optimizer == "l-bfgs-b"
    assert tmplt.l_bfgs_b_l == 0.7
    assert tmplt.l_bfgs_b_epsilon == 0.1
    assert tmplt.l_bfgs_b_norm == "l2"
    assert tmplt.fixed_effect
    assert not tmplt.heteroskedastic
    assert not tmplt.fixed_effect_var
    print("Fitting")
    model.fit(norm_data_from_arrays)
    print("fitted")
    assert model.is_fitted

    model1 = model[model.response_vars[0]]
    assert isinstance(model1, BLR)
    assert model1.n_iter == 10
    assert model1.tol == 1e-3
    assert not model1.ard
    assert model1.optimizer == "l-bfgs-b"
    assert model1.l_bfgs_b_l == 0.7
    assert model1.l_bfgs_b_epsilon == 0.1
    assert model1.l_bfgs_b_norm == "l2"
    assert model1.fixed_effect
    assert not model1.fixed_effect_var
    model.predict(norm_data_from_arrays)
    assert hasattr(norm_data_from_arrays, "Z")
    Z_bak = copy.deepcopy(norm_data_from_arrays["Z"])
    del norm_data_from_arrays["Z"]
    assert not hasattr(norm_data_from_arrays, "Z")
    model.save(save_dir_blr)
    del model
    loaded_model = NormativeModel.load(save_dir_blr)
    model1 = loaded_model[loaded_model.response_vars[0]]
    loaded_model.predict(norm_data_from_arrays)
    assert np.allclose(norm_data_from_arrays["Z"], Z_bak)

    assert loaded_model.is_fitted
    assert isinstance(model1, BLR)
    assert model1.n_iter == 10
    assert model1.tol == 1e-3
    assert not model1.ard
    assert model1.optimizer == "l-bfgs-b"
    assert model1.l_bfgs_b_l == 0.7
    assert model1.l_bfgs_b_epsilon == 0.1
    assert model1.l_bfgs_b_norm == "l2"
    assert model1.fixed_effect
    assert not model1.fixed_effect_var


def test_log1p_transform(
    log1p_transform_norm_blr_model: NormativeModel,
    norm_data_from_arrays: NormData,
    test_norm_data_from_arrays: NormData,
) -> None:
    log1p_transform_norm_blr_model.fit_predict(
        norm_data_from_arrays, test_norm_data_from_arrays
    )

    # Check that Y are non-negative in the original data. Y has to be > -1
    # for the log(Y+1) transform to work.
    assert bool(
        np.all(test_norm_data_from_arrays["Y"].values >= 0)
    )

    # Both training and test centiles should be bigger than -1 (not 0) due to
    # the exp(Y) - 1 transform
    assert bool(
        np.all(norm_data_from_arrays["centiles"].values > -1)
    )
    assert bool(
        np.all(test_norm_data_from_arrays["centiles"].values > -1)
    )

    # Yhat is a mean predicted in log1p space and mapped back with expm1, so it
    # should be bigger than -1 (not 0). A predicted mean below 0 in log1p space
    # maps to a Y between -1 and 0 due to  the exp(Y) - 1 transform
    assert bool(
        np.all(norm_data_from_arrays["Yhat"].values > -1)
    )
    assert bool(
        np.all(test_norm_data_from_arrays["Yhat"].values > -1)
    )


def test_log_transformed(
    log_transform_norm_blr_model: NormativeModel,
    norm_data_from_arrays: NormData,
    test_norm_data_from_arrays: NormData,
) -> None:
    # Force the BOTH test and train data to be >= 1e-6. 
    # We do that as Y has to be > 0 for the log(Y) transform to work.
    norm_data_from_arrays["Y"].values = (
        norm_data_from_arrays["Y"].values.clip(min=1e-6))
    test_norm_data_from_arrays["Y"].values = (
        test_norm_data_from_arrays["Y"].values.clip(min=1e-6))

    log_transform_norm_blr_model.fit_predict(
        norm_data_from_arrays, test_norm_data_from_arrays
    )

    # Check that Y are positive in the original data
    assert bool(
        np.all(test_norm_data_from_arrays["Y"].values > 0)
    )

    # Both training and test centiles should be bigger than 0 due to the exp(Y)
    # transform
    assert bool(
        np.all(norm_data_from_arrays["centiles"].values > 0)
    )
    assert bool(
        np.all(test_norm_data_from_arrays["centiles"].values > 0)
    )

    # We dont expect any negative yhat values in the train and test dataset
    assert bool(
        np.all(norm_data_from_arrays["Yhat"].values > 0)
    )
    assert bool(
        np.all(test_norm_data_from_arrays["Yhat"].values > 0)
    )
