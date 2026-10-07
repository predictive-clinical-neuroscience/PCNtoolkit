import importlib.metadata
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
def test_composite_basis_interaction_linear():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    basis_function = CompositeBasisFunction(
        parts=[
            LinearBasisFunction(basis_column=0),
            LinearBasisFunction(basis_column=1),
        ],
        interactions=[(0, 1)],
    )

    basis_function.fit(X)
    Phi = basis_function.transform(X)

    expected = np.column_stack(
        [
            X[:, 0],
            X[:, 1],
            X[:, 0] * X[:, 1],
        ]
    )

    np.testing.assert_allclose(Phi, expected)

    assert basis_function.is_fitted
    assert basis_function.dimension == 3


def test_composite_basis_without_interactions():
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    basis_function = CompositeBasisFunction(
        parts=[
            LinearBasisFunction(basis_column=0),
            LinearBasisFunction(basis_column=1),
        ]
    )

    basis_function.fit(X)
    Phi = basis_function.transform(X)

    np.testing.assert_allclose(Phi, X)

    assert basis_function.interactions == []
    assert basis_function.dimension == 2


def test_composite_basis_bspline_by_linear_interaction():
    age = np.linspace(10, 80, 20)
    sex = np.tile([0.0, 1.0], 10)
    X = np.column_stack([age, sex])

    age_basis = BsplineBasisFunction(
        basis_column=0,
        degree=3,
        nknots=5,
    )
    sex_basis = LinearBasisFunction(basis_column=1)

    basis_function = CompositeBasisFunction(
        parts=[age_basis, sex_basis],
        interactions=[(0, 1)],
    )

    basis_function.fit(X)
    Phi = basis_function.transform(X)

    age_basis.basis_column = 0
    age_Phi = age_basis.transform(age)
    sex_basis.basis_column = 0
    sex_Phi = sex_basis.transform(sex)

    expected_interaction = age_Phi[:, :, None] * sex_Phi[:, None, :]
    expected_interaction = expected_interaction.reshape(len(X), -1)

    interaction_Phi = Phi[:, -expected_interaction.shape[1]:]

    np.testing.assert_allclose(
        interaction_Phi,
        expected_interaction,
    )

    # For sex == 0, all interaction terms should vanish.
    np.testing.assert_allclose(
        interaction_Phi[sex == 0],
        0.0,
    )

    # For sex == 1, the interaction equals the age basis.
    np.testing.assert_allclose(
        interaction_Phi[sex == 1],
        age_Phi[sex == 1],
    )


def test_composite_basis_interaction_uses_parts_order():
    X = np.column_stack(
        [
            np.linspace(10, 80, 20),
            np.linspace(1, 2, 20),
        ]
    )

    linear_basis = LinearBasisFunction(basis_column=1)
    bspline_basis = BsplineBasisFunction(
        basis_column=0,
        degree=3,
        nknots=5,
    )

    basis_function = CompositeBasisFunction(
        parts=[
            linear_basis,   # parts[0], despite basis_column=1
            bspline_basis,  # parts[1], despite basis_column=0
        ],
        interactions=[(0, 1)],
    )

    basis_function.fit(X)
    Phi = basis_function.transform(X)

    linear_basis.basis_column = 0
    linear_Phi = linear_basis.transform(X[:, 1])

    bspline_basis.basis_column = 0
    bspline_Phi = bspline_basis.transform(X[:, 0])

    expected_interaction = (
        linear_Phi[:, :, None] * bspline_Phi[:, None, :]
    ).reshape(len(X), -1)

    np.testing.assert_allclose(
        Phi[:, -expected_interaction.shape[1]:],
        expected_interaction,
    )


def test_composite_basis_interaction_to_and_from_dict():
    basis_function = CompositeBasisFunction(
        parts=[
            BsplineBasisFunction(
                basis_column=0,
                degree=3,
                nknots=5,
            ),
            LinearBasisFunction(basis_column=1),
        ],
        interactions=[(0, 1)],
    )

    basis_function_dict = basis_function.to_dict()
    restored = create_basis_function(basis_function_dict)

    assert restored.interactions == [(0, 1)]
    assert len(restored.parts) == 2
    assert isinstance(restored.parts[0], BsplineBasisFunction)
    assert isinstance(restored.parts[1], LinearBasisFunction)


def test_composite_basis_old_dict_without_interactions():
    basis_function_dict = {
        "basis_function": "CompositeBasis",
        "parts": [
            {
                "basis_function": "LinearBasisFunction",
                "basis_column": 0,
            },
            {
                "basis_function": "LinearBasisFunction",
                "basis_column": 1,
            },
        ],
    }

    restored = create_basis_function(basis_function_dict)

    assert restored.interactions == []


def test_composite_basis_multidimensional_interaction():
    X = np.column_stack(
        [
            np.linspace(10, 80, 20),
            np.linspace(0, 1, 20),
        ]
    )

    basis_1 = BsplineBasisFunction(
        basis_column=0,
        degree=3,
        nknots=5,
    )
    basis_2 = BsplineBasisFunction(
        basis_column=1,
        degree=2,
        nknots=4,
    )

    basis_function = CompositeBasisFunction(
        parts=[basis_1, basis_2],
        interactions=[(0, 1)],
    )

    basis_function.fit(X)
    Phi = basis_function.transform(X)

    # Transform the two components independently.
    basis_1.basis_column = 0
    left = basis_1.transform(X[:, 0])

    basis_2.basis_column = 0
    right = basis_2.transform(X[:, 1])

    # Expected Cartesian product of the two basis expansions.
    expected = (
        left[:, :, None] * right[:, None, :]
    ).reshape(len(X), -1)

    interaction_Phi = Phi[:, -expected.shape[1]:]

    np.testing.assert_allclose(
        interaction_Phi,
        expected,
    )

    assert expected.shape[1] == left.shape[1] * right.shape[1]
    assert basis_function.dimension == (
        basis_1.dimension
        + basis_2.dimension
        + basis_1.dimension * basis_2.dimension
    )


def test_composite_basis_preserves_untransformed_columns():
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )

    basis_function = CompositeBasisFunction(
        parts=[LinearBasisFunction(basis_column=0)]
    )

    basis_function.fit(X)
    Phi = basis_function.transform(X)

    np.testing.assert_allclose(Phi, X)
    
