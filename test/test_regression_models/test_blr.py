import json
import os
import shutil
from typing import Callable

import numpy as np
import pytest

from pcntoolkit.dataio.norm_data import NormData
from pcntoolkit.math_functions.basis_function import (
    BsplineBasisFunction,
)
from pcntoolkit.normative_model import NormativeModel
from pcntoolkit.regression_model.blr import (
    BLR,
    create_design_matrix,
)
from pcntoolkit.util.migration import registry
from test.fixtures.blr_model_fixtures import *
from test.fixtures.norm_data_fixtures import *
from test.fixtures.path_fixtures import *


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


@pytest.mark.parametrize("ard", [False, True])
def test_loaded_model_rebuilds_prior_and_posterior(
    ard: bool,
    blr_model_factory,
    norm_data_from_arrays: NormData,
    fitted_norm_blr_model: NormativeModel,
) -> None:
    """A loaded BLR rebuilds Sigma_a and Lambda_a, which are not saved.

    The prior (Sigma_a, Lambda_a), posterior (A, m) and log likelihood of
    the loaded model must match the fitted model. With ARD, each column
    has its own prior, so Sigma_a is not a multiple of the identity.
    """
    response_var = norm_data_from_arrays.response_vars[0]
    resp_data = norm_data_from_arrays.sel(response_vars=response_var)
    X, be, be_maps, Y, _ = fitted_norm_blr_model.extract_data(resp_data)
    fitted_blr_model = blr_model_factory(ard=ard)
    fitted_blr_model.fit(X, be, be_maps, Y)
    # Round-trip through JSON text, as save() and load() do.
    my_dict = json.loads(json.dumps(fitted_blr_model.to_dict()))
    assert "Sigma_a" not in my_dict
    assert "Lambda_a" not in my_dict
    blr = BLR.from_dict(my_dict)
    Phi, Phi_var = blr.Phi_Phi_var(X.values, be.values)
    # At the stored hyp, loglik must rebuild Sigma_a and Lambda_a (not in the file).
    loaded_nlZ = blr.loglik(blr.hyp, Phi, Y.values, Phi_var)
    # The fitted model must hold the posterior of its final hyp (issue #550),
    # so the loaded model gives the same value.
    fitted_nlZ = fitted_blr_model.loglik(fitted_blr_model.hyp, Phi, Y.values, Phi_var)
    np.testing.assert_allclose(loaded_nlZ, fitted_nlZ, rtol=1e-10)
    np.testing.assert_allclose(blr.Sigma_a, fitted_blr_model.Sigma_a, rtol=1e-10)
    np.testing.assert_allclose(blr.Lambda_a, fitted_blr_model.Lambda_a, rtol=1e-10)
    if ard:
        # Guard that the ARD case really has a different prior per column.
        assert np.ptp(np.diag(blr.Sigma_a)) > 0
    np.testing.assert_allclose(blr.A, fitted_blr_model.A, rtol=1e-10)
    np.testing.assert_allclose(blr.m, fitted_blr_model.m, rtol=1e-10)


def test_fit(
    blr_model_factory,
    norm_data_from_arrays: NormData,
    fitted_norm_blr_model: NormativeModel,
):
    blr_model = blr_model_factory()
    print("fitting")
    response_var = norm_data_from_arrays.response_vars[0]
    X, be, be_maps, Y, _ = fitted_norm_blr_model.extract_data(
        norm_data_from_arrays.sel(response_vars=response_var)
    )
    blr_model.fit(X, be, be_maps, Y)
    assert blr_model.is_fitted


def test_forward_backward(
    fitted_blr_model: BLR,
    norm_data_from_arrays: NormData,
    fitted_norm_blr_model: NormativeModel,
):
    response_var = norm_data_from_arrays.response_vars[0]
    X, be, _, Y, _ = fitted_norm_blr_model.extract_data(
        norm_data_from_arrays.sel(response_vars=response_var)
    )
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


# ----
# Fixed effect slopes must be on the raw covariate (e.g. age), even after the B-spline
# basis dropped its redundant linear term (include_linear=False, issue #542)
# ----


def _slope_design_inputs() -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    """Two covariates (age first) and one batch effect with three sites."""
    rng = np.random.default_rng(0)
    covs = np.column_stack([rng.uniform(-2, 2, 30), rng.normal(size=30)])
    be = rng.integers(0, 3, size=(30, 1))
    be_maps = {"site": {"a": 0, "b": 1, "c": 2}}
    return covs, be, be_maps


def _slope_columns(Phi: np.ndarray, n_levels: int) -> np.ndarray:
    """The batch effect slope columns are appended last."""
    return Phi[:, -n_levels:]


@pytest.mark.parametrize("include_linear", [True, False])
def test_fixed_effect_slope_uses_raw_covariate(include_linear: bool) -> None:
    """
    Fixed effect slopes must be on the raw covariate (e.g. age), whatever the basis.
    """
    covs, be, be_maps = _slope_design_inputs()
    basis = BsplineBasisFunction(basis_column=0, include_linear=include_linear)
    basis.fit(covs)
    Phi = create_design_matrix(
        basis.transform(covs),
        be,
        be_maps,
        linear=True,
        intercept=True,
        fixed_effect_slope=True,
        fixed_effect_slope_X=covs,
    )
    expected = covs[:, [0]] * np.eye(3)[be[:, 0]]
    assert np.allclose(_slope_columns(Phi, 3), expected)


def test_fixed_effect_slope_follows_basis_column() -> None:
    """When user specifies BsplineBasisFunction(basis_column=1), the per-site slope is
    on covariate 1, unless fixed_effect_slope_indices overrides it."""
    covs, be, be_maps = _slope_design_inputs()
    blr = BLR(
        basis_function_mean=BsplineBasisFunction(basis_column=1),
        fixed_effect_slope=True,
    )
    blr.be_maps = be_maps
    Phi, _ = blr.Phi_Phi_var(covs, be)
    onehot = np.eye(3)[be[:, 0]]
    assert np.allclose(_slope_columns(Phi, 3), covs[:, [1]] * onehot)

    # Explicit fixed_effect_slope_indices override the default
    blr.fixed_effect_slope_indices = [0]
    Phi, _ = blr.Phi_Phi_var(covs, be)
    assert np.allclose(_slope_columns(Phi, 3), covs[:, [0]] * onehot)


def test_migration_keeps_legacy_slope_on_covariate_0() -> None:
    """Models saved before v1.4.0 with indices None keep the slope on covariate 0."""
    d = {"fixed_effect_slope_indices": None, "fixed_effect_var_slope_indices": None}
    d = registry.migrate("BLR", d, version="1.3.0")
    assert d["fixed_effect_slope_indices"] == [0]
    assert d["fixed_effect_var_slope_indices"] == [0]


def test_migration_loads_model_when_slopes_are_off() -> None:
    """Old models without per-site slopes load, whatever their slope indices.

    The migration rejects old slope indices on a B-spline basis, e.g. [3],
    because they changed meaning in v1.4.0. But with fixed_effect_slope=False
    the indices are never used, so the model must still load.
    """
    d = {
        "fixed_effect_slope": False,
        "fixed_effect_slope_indices": [3],
        "basis_function_mean": {"basis_function": "BsplineBasisFunction"},
    }
    # Pretend the model was saved with v1.3.0 so that the migration logic for pre-1.4.0
    # models is triggered.
    registry.migrate("BLR", d, version="1.3.0")


def test_init_hyp_accepts_array_hyp0(blr_model_factory: Callable) -> None:
    """init_hyp() must accept hyp0 as np.ndarray, the type the docstring
    declares. Regression test for #557."""
    blr_model = blr_model_factory(hyp0=np.array([0.0, 0.0]))
    hyp0 = blr_model.init_hyp()
    assert isinstance(hyp0, np.ndarray)
    assert hyp0.shape == (2,)


def test_init_hyp_accepts_list_hyp0(blr_model_factory: Callable) -> None:
    """init_hyp() must accept hyp0 as a list and return it as an array.
    Regression test for #557."""
    blr_model = blr_model_factory(hyp0=[0.0, 0.0])
    hyp0 = blr_model.init_hyp()
    assert isinstance(hyp0, np.ndarray)
    assert hyp0.shape == (2,)


def test_init_hyp_after_to_from_dict(blr_model_factory: Callable) -> None:
    """from_dict converts saved list values to arrays; init_hyp() must still
    work on the result (the save -> load -> refit path). Regression test for #557."""
    blr_model = blr_model_factory(hyp0=[0.0, 0.0])
    restored = BLR.from_dict(blr_model.to_dict())
    assert isinstance(restored.hyp0, np.ndarray)
    hyp0 = restored.init_hyp()
    assert isinstance(hyp0, np.ndarray)
    assert hyp0.shape == (2,)


def test_fit_with_array_hyp0(
    norm_data_from_arrays: NormData,
    fitted_norm_blr_model: NormativeModel,
) -> None:
    """fit() must start the optimizer from an array hyp0, the type the
    docstring declares. Regression test for #557."""
    blr_model = BLR("test_blr_array_hyp0", hyp0=np.array([0.0, 0.0]))
    response_var = norm_data_from_arrays.response_vars[0]
    X, be, be_maps, Y, _ = fitted_norm_blr_model.extract_data(
        norm_data_from_arrays.sel(response_vars=response_var)
    )
    blr_model.fit(X, be, be_maps, Y)
    assert blr_model.is_fitted


def test_refit_after_save_and_load(
    norm_data_from_arrays: NormData,
    save_dir_blr: str,
) -> None:
    """A model created with hyp0 as a list must be refittable after save and
    load, since from_dict brings hyp0 back as an array. Regression test for #557."""
    if os.path.exists(save_dir_blr):
        shutil.rmtree(save_dir_blr)
    os.makedirs(save_dir_blr, exist_ok=True)
    blr_model = BLR("test_blr_hyp0_reload", hyp0=[0.0, 0.0])
    model = NormativeModel(
        blr_model,
        save_dir=save_dir_blr,
        saveresults=False,
        saveplots=False,
        evaluate_model=False,
    )
    model.fit(norm_data_from_arrays)
    response_var = norm_data_from_arrays.response_vars.values[0]
    loaded = NormativeModel.load(save_dir_blr)
    assert isinstance(loaded[response_var].hyp0, np.ndarray)
    loaded.fit(norm_data_from_arrays)
    assert loaded[response_var].is_fitted
