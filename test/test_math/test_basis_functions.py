import importlib.metadata
import json
from unittest.mock import patch

import numpy as np
import pytest

from pcntoolkit.math_functions.basis_function import (
    BasisFunction,
    BsplineBasisFunction,
    CompositeBasisFunction,
    LinearBasisFunction,
    create_basis_function,
)
from test.fixtures.norm_data_fixtures import *


@pytest.mark.parametrize(
    "basis_function_name, basis_function_class",
    [
        ("polynomial", "PolynomialBasisFunction"),
        ("bspline", "BsplineBasisFunction"),
        ("linear", "LinearBasisFunction"),
    ],
)
def test_to_and_from_dict(basis_function_name, basis_function_class):
    basis_function = create_basis_function(basis_function_name, degree=3, nknots=5)
    basis_function_dict = basis_function.to_dict()
    basis_function_from_dict = create_basis_function(basis_function_dict)
    assert basis_function_dict == basis_function_from_dict.__dict__ | {
        "basis_function": basis_function_class
    }


@pytest.mark.parametrize(
    "basis_column, degree", [(0, 3), (1, 4)]
)  # None is for all columns
def test_poly_fit_and_transform(norm_data_from_arrays, basis_column, degree):
    basis_function = create_basis_function(
        "polynomial", degree=degree, basis_column=basis_column
    )
    X = norm_data_from_arrays.X.values
    basis_function.fit(X)
    Phi = basis_function.transform(X)
    assert basis_function.is_fitted
    assert basis_function.basis_name == "poly"
    assert Phi.shape == (
        norm_data_from_arrays.X.data.shape[0],
        basis_function.dimension + X.shape[1] - 1,
    )


@pytest.mark.parametrize("nknots, degree", [(8, 4), (10, 4), (10, 3)])
def test_bspline_fit_and_transform(norm_data_from_arrays, nknots, degree):
    basis_function = create_basis_function(
        "bspline",
        source_array_name="X",
        degree=degree,
        nknots=nknots,
    )
    X = norm_data_from_arrays.X.values
    basis_function.fit(X)
    Phi = basis_function.transform(X)

    assert basis_function.is_fitted
    assert basis_function.basis_name == "bspline"
    assert basis_function.dimension == nknots + degree - 1
    assert Phi.shape == (
        norm_data_from_arrays.X.data.shape[0],
        basis_function.dimension + X.shape[1] - 1,
    )


@pytest.mark.parametrize("nknots, degree", [(8, 4), (10, 3)])
def test_bspline_with_linear_term(norm_data_from_arrays, nknots, degree):
    basis_function = create_basis_function(
        "bspline",
        source_array_name="X",
        degree=degree,
        nknots=nknots,
        include_linear=True,
    )
    X = norm_data_from_arrays.X.values
    basis_function.fit(X)
    Phi = basis_function.transform(X)

    assert basis_function.dimension == nknots + degree
    assert Phi.shape == (
        norm_data_from_arrays.X.data.shape[0],
        basis_function.dimension + X.shape[1] - 1,
    )


def test_composite_from_dict_passes_version_to_parts() -> None:
    """Loading a saved composite basis must not migrate its parts as if they
    were saved with v0.0.0 (the parts have no ptk_version of their own)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(50, 2))
    composite = CompositeBasisFunction(
        [BsplineBasisFunction(basis_column=0), LinearBasisFunction(basis_column=1)]
    )
    composite.fit(X)

    current_version = importlib.metadata.version("pcntoolkit")
    with patch("pcntoolkit.util.migration.Output.warning") as mock_warning:
        BasisFunction.from_dict(composite.to_dict(), version=current_version)

    assert mock_warning.call_count == 0

def test_composite_bspline_quantile_preserves_fit_on_save_load() -> None:
    """Regression test for issue #554.

    A fitted CompositeBasisFunction must preserve its own `is_fitted` state,
    and that of a quantile-knot B-spline part, across the documented
    `to_dict()` -> JSON -> `from_dict()` round trip. Otherwise code that
    conditionally refits a basis it believes is unfitted (e.g.
    `BLR.Phi_Phi_var`'s `if not basis_function.is_fitted: basis_function.fit(X)`)
    will refit the loaded composite -- and therefore recompute quantile
    knots -- on prediction data, producing different results than the
    original fitted model.
    """
    rng = np.random.default_rng(554)
    # Skewed training distribution so quantile knots would clearly differ
    # if recomputed on a very differently distributed prediction set.
    X_train = np.column_stack(
        [rng.exponential(scale=2.0, size=200), rng.uniform(0, 1, size=200)]
    )

    composite = CompositeBasisFunction(
        [
            BsplineBasisFunction(
                basis_column=0, degree=3, nknots=5, knot_method="quantile"
            ),
            LinearBasisFunction(basis_column=1),
        ]
    )
    composite.fit(X_train)
    assert composite.is_fitted
    original_knots = list(composite.parts[0].knots)

    # Round-trip through the documented serialization flow.
    current_version = importlib.metadata.version("pcntoolkit")
    serialized = json.dumps(composite.to_dict())
    loaded = BasisFunction.from_dict(json.loads(serialized), version=current_version)

    assert loaded.is_fitted
    assert loaded.parts[0].is_fitted
    assert loaded.parts[1].is_fitted
    np.testing.assert_array_equal(loaded.parts[0].knots, original_knots)

    # Prediction data drawn from a very different distribution than the
    # training data. If the loaded basis were refitted here, quantile knots
    # (and therefore the design matrix) would differ from the original.
    X_predict = np.column_stack(
        [rng.normal(loc=50.0, scale=10.0, size=30), rng.uniform(0, 1, size=30)]
    )

    # Mirror the "refit only if not already fitted" pattern regression
    # models use when consuming a basis function (see BLR.Phi_Phi_var).
    for basis in (composite, loaded):
        if not basis.is_fitted:
            basis.fit(X_predict)

    Phi_original = composite.transform(X_predict)
    Phi_loaded = loaded.transform(X_predict)
    np.testing.assert_array_equal(Phi_original, Phi_loaded)