import pytest

from pcntoolkit.math_functions.basis_function import create_basis_function
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
