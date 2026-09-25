import json

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.regression_model.blr import BLR
from test.fixtures.blr_model_fixtures import *
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *
from pcntoolkit.normative_model import NormativeModel



@pytest.mark.parametrize("n_iter,tol,ard", [(100, 1e-3, False), (1, 1e-6, True)])
def test_blr_to_and_from_dict_and_args(n_iter, tol, ard):
    args = {"n_iter": n_iter, "tol": tol, "ard": ard}
    blr1 = BLR.from_args("test_blr", args)
    assert blr1.n_iter == n_iter
    assert blr1.tol == tol
    assert blr1.ard == ard
    assert blr1.optimizer == "l-bfgs-b"
    assert blr1.l_bfgs_b_l == 0.1
    assert blr1.l_bfgs_b_epsilon == 0.1
    assert blr1.l_bfgs_b_norm == "l2"

    dict2 = blr1.to_dict()
    assert dict2["n_iter"] == n_iter
    assert dict2["tol"] == tol
    assert dict2["ard"] == ard
    assert dict2["optimizer"] == "l-bfgs-b"
    assert dict2["l_bfgs_b_l"] == 0.1
    assert dict2["l_bfgs_b_epsilon"] == 0.1
    assert dict2["l_bfgs_b_norm"] == "l2"

    blr2 = BLR.from_dict(dict2)
    assert blr2.n_iter == n_iter
    assert blr2.tol == tol
    assert blr2.ard == ard
    assert blr2.optimizer == "l-bfgs-b"
    assert blr2.l_bfgs_b_l == 0.1
    assert blr2.l_bfgs_b_epsilon == 0.1
    assert blr2.l_bfgs_b_norm == "l2"


def test_loglik_after_to_and_from_dict(
    fitted_blr_model: BLR,
    norm_data_from_arrays: NormData,
    fitted_norm_blr_model: NormativeModel,
) -> None:
    # Round-trip through JSON text, as save() and load() do.
    my_dict = json.loads(json.dumps(fitted_blr_model.to_dict()))
    assert "Sigma_a" not in my_dict
    assert "Lambda_a" not in my_dict
    blr = BLR.from_dict(my_dict)
    response_var = norm_data_from_arrays.response_vars[0]
    resp_data = norm_data_from_arrays.sel(response_vars=response_var)
    X, be, _, Y, _ = fitted_norm_blr_model.extract_data(resp_data)
    Phi, Phi_var = blr.Phi_Phi_var(X.values, be.values)
    # At the stored hyp, loglik must rebuild Sigma_a and Lambda_a (not in the file).
    assert np.isfinite(blr.loglik(blr.hyp, Phi, Y.values, Phi_var))


def test_fit(blr_model_factory, norm_data_from_arrays: NormData, fitted_norm_blr_model: NormativeModel):
    blr_model = blr_model_factory()
    print("fitting")
    response_var = norm_data_from_arrays.response_vars[0]
    X, be, be_maps, Y, _ = fitted_norm_blr_model.extract_data(norm_data_from_arrays.sel(response_vars=response_var))
    blr_model.fit(X, be, be_maps, Y)
    assert blr_model.is_fitted


def test_forward_backward(fitted_blr_model: BLR, norm_data_from_arrays: NormData, fitted_norm_blr_model: NormativeModel):
    response_var = norm_data_from_arrays.response_vars[0]
    X, be, _, Y, _ = fitted_norm_blr_model.extract_data(norm_data_from_arrays.sel(response_vars=response_var))
    Z = fitted_blr_model.forward(X, be, Y)
    assert Z.shape == Y.shape
    Y_prime = fitted_blr_model.backward(X, be, Z)
    assert Y_prime.shape == Y.shape
    assert np.allclose(Y_prime, Y)


def test_parse_hyps(norm_data_from_arrays: NormData):
    X = norm_data_from_arrays.X.to_numpy()
    var_X = norm_data_from_arrays.X.to_numpy()
    blr = BLR("test_blr")
    blr.D = X.shape[1]
    blr.var_D = var_X.shape[1]
    hyp = blr.init_hyp()
    alpha, beta, gamma = blr.parse_hyps(hyp, X, var_X)
    assert True


def test_cg_ard_fit(
    blr_model_factory,
    norm_data_from_arrays: NormData,
    fitted_norm_blr_model: NormativeModel,
):
    """Test that CG optimizer with ARD (non-heteroskedastic) fits successfully.
    
    Parameters
    ----------
    blr_model_factory: Callable
        Fixture that builds BLR models with optional overrides.
    norm_data_from_arrays: NormData
        Fixture that provides NormData from arrays.
    fitted_norm_blr_model: NormativeModel
        Fixture that provides a fitted NormativeModel 
    for testing.
    """
    # Build a BLR variant with ARD enabled and CG optimizer;
    # all other settings are inherited from BLR_BASE_CONFIG.
    blr_cg = blr_model_factory(ard=True, optimizer="cg")
    # Extract data for the first response variable.
    response_var = norm_data_from_arrays.response_vars[0]
    # Unpack design matrix, batch effects, maps, and targets.
    X, be, be_maps, Y, _ = fitted_norm_blr_model.extract_data(
        norm_data_from_arrays.sel(response_vars=response_var)
    )
    blr_cg.fit(X, be, be_maps, Y)
    assert blr_cg.is_fitted
